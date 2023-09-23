use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use flume::Sender;
use itertools::Itertools;
use qp_trie::Trie;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use web_rwkv::{
    model::{BackedState, Model, ModelState},
    tokenizer::Tokenizer,
};

use crate::{sampler::Sampler, FinishReason, Token, TokenCounter};

#[derive(Debug)]
pub enum SlotResult {
    /// There is an idle slot ready to be picked up.
    Success(usize),
    /// An idle slot is swapped.
    Fault(usize),
    /// There is no idle slot left.
    Failure(Box<GenerateContext>),
}

#[derive(Debug)]
enum SlotState {
    /// The slot might be either picked up or swapped.
    Idle(Tokens),
    /// The slot is locked and is waiting for processing.
    Wait(Box<GenerateContext>),
    /// The slot is currently under processing.
    Busy,
}

impl Default for SlotState {
    fn default() -> Self {
        Self::Idle(Default::default())
    }
}

#[repr(transparent)]
#[derive(Debug, Default, Clone)]
pub struct Tokens(pub Vec<u16>);

impl std::ops::Deref for Tokens {
    type Target = TokenSlice;

    fn deref(&self) -> &Self::Target {
        self.0.as_token_slice()
    }
}

impl Borrow<[u8]> for Tokens {
    fn borrow(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

impl Borrow<[u16]> for Tokens {
    fn borrow(&self) -> &[u16] {
        &self.0
    }
}

impl Borrow<TokenSlice> for Tokens {
    fn borrow(&self) -> &TokenSlice {
        self.0[..].as_token_slice()
    }
}

impl qp_trie::Break for Tokens {
    type Split = TokenSlice;

    fn empty<'a>() -> &'a Self::Split {
        Default::default()
    }

    fn find_break(&self, loc: usize) -> &Self::Split {
        self.0[..loc].as_token_slice()
    }
}

#[repr(transparent)]
pub struct TokenSlice([u16]);

impl std::ops::Deref for TokenSlice {
    type Target = [u16];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Borrow<[u8]> for TokenSlice {
    fn borrow(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

impl Default for &TokenSlice {
    fn default() -> Self {
        <&[u16]>::default().as_token_slice()
    }
}

pub trait AsTokenSlice {
    fn as_token_slice(&self) -> &TokenSlice;
}

impl AsTokenSlice for [u16] {
    fn as_token_slice(&self) -> &TokenSlice {
        let ptr = self as *const [u16] as *const TokenSlice;
        unsafe { &*ptr }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateContext {
    /// Tokens that are provided at first.
    pub prompt_tokens: Vec<u16>,
    /// Tokens that have been computed and cached.
    pub prefix: Tokens,
    /// Tokens to be computed.
    pub suffix: Tokens,
    /// The accumulated penalties for model-output tokens.
    pub penalties: HashMap<u16, f32>,
    /// Texts that are output by the model.
    pub model_text: Vec<u8>,
    /// Model may output partial utf-8. This makes sure the output is always valid.
    pub output_buffer: Vec<u8>,
    /// Tokens that are output by the model.
    pub model_tokens: Vec<u16>,

    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub sampler: Sampler,
    pub logit_bias: HashMap<u16, f32>,
    pub embed: bool,
    pub sender: Sender<Token>,
}

pub struct Runtime<'a> {
    model: Arc<Model<'a>>,
    slots: Vec<SlotState>,
    state: ModelState,
    backed: Trie<Tokens, BackedState>,
    max_runtime_batch: usize,
}

impl<'a> Runtime<'a> {
    pub fn new(model: Arc<Model<'a>>, state: ModelState, max_runtime_batch: usize) -> Self {
        Self {
            model,
            slots: Default::default(),
            state,
            backed: Default::default(),
            max_runtime_batch,
        }
    }

    /// Queue a generation task.
    pub fn queue(&mut self, context: GenerateContext) -> SlotResult {
        let tokens = Tokens([context.prefix, context.suffix].concat());
        let choice = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| match slot {
                SlotState::Idle(content) => {
                    if tokens.starts_with(content) {
                        Some((index, true, content.len()))
                    } else {
                        Some((index, false, 0))
                    }
                }
                _ => None,
            })
            .max_by(|(_, _, lhs), (_, _, rhs)| lhs.cmp(rhs));
        match choice {
            None => SlotResult::Failure(
                GenerateContext {
                    prefix: Default::default(),
                    suffix: tokens,
                    ..context
                }
                .into(),
            ),
            Some((batch, false, _)) => {
                let prefix = self.backed.longest_common_prefix(&tokens);
                let len = match self.backed.contains_key(prefix) {
                    true => prefix.len(),
                    false => 0,
                };
                let mut state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        ..context
                    }
                    .into(),
                );

                let prefix = prefix.to_vec();
                let reload = self
                    .backed
                    .remove(prefix[..].as_token_slice())
                    .unwrap_or_else(|| BackedState::new(self.model.info(), 1));

                std::mem::swap(&mut state, &mut self.slots[batch]);
                match state {
                    SlotState::Idle(content) => {
                        let backed = self.state.back_batch(batch).expect("back state");
                        self.backed.insert(content, backed);
                        self.state.load_batch(&reload, batch).expect("load state");
                        SlotResult::Fault(batch)
                    }
                    _ => unreachable!(),
                }
            }
            Some((id, true, len)) => {
                let state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        ..context
                    }
                    .into(),
                );
                let _ = std::mem::replace(&mut self.slots[id], state);
                SlotResult::Success(id)
            }
        }
    }
}

pub fn run(runtime: Arc<Mutex<Option<Runtime>>>, tokenizer: Tokenizer) {
    let penalty_free_tokens = (0..u16::MAX)
        .filter(|token| {
            let word = tokenizer.decode(&[*token]).unwrap_or_default();
            let word = String::from_utf8(word).unwrap_or_default();
            word.contains('\n')
        })
        .collect::<HashSet<_>>();

    let mut payload: Vec<Option<Box<GenerateContext>>> = Default::default();
    let mut done: Vec<Option<Box<GenerateContext>>> = Default::default();

    let mut model: Option<Arc<Model>> = Default::default();
    let mut state: Option<ModelState> = Default::default();

    let mut process = || -> Result<()> {
        if let Some(runtime) = &mut *runtime.lock().unwrap() {
            payload.resize(runtime.slots.len(), None);
            done.resize(runtime.slots.len(), None);
            model.replace(runtime.model.clone());
            state.replace(runtime.state.clone());

            // take data from some waiting slots
            let occupied = payload.iter().filter(|x| x.is_some()).count();
            let remain = runtime.max_runtime_batch - runtime.max_runtime_batch.min(occupied);
            let batches = runtime
                .slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| matches!(slot, SlotState::Wait(_)))
                .take(remain)
                .map(|(batch, _)| batch)
                .collect_vec();
            for batch in batches {
                let mut slot = SlotState::Busy;
                std::mem::swap(&mut runtime.slots[batch], &mut slot);
                match slot {
                    SlotState::Wait(context) => {
                        let _ = context.sender.send(Token::Start);
                        payload[batch].replace(context)
                    }
                    _ => unreachable!(),
                };
            }

            // reset all finished slots to idle
            for (done, slot) in done.iter_mut().zip_eq(runtime.slots.iter_mut()) {
                if let Some(done) = done.take() {
                    assert!(matches!(slot, SlotState::Busy));
                    *slot = SlotState::Idle(done.prefix);
                }
            }
        }

        if let (Some(model), Some(state)) = (&model, &state) {
            let mut tokens = payload
                .iter()
                .map(|context| match context {
                    Some(context) => context.suffix.0.clone(),
                    None => vec![],
                })
                .collect_vec();

            // run the model until there is at least one slot finished
            let logits = loop {
                let logits = model.run(&mut tokens, state)?;
                if logits.iter().any(Option::is_some) {
                    break logits;
                }
            };
            // post-process logits
            let logits = payload
                .par_iter()
                .zip_eq(logits.into_par_iter())
                .map(|(context, x)| (context.as_ref(), x))
                .map(|(context, logits)| {
                    context.zip(logits).map(|(context, mut logits)| {
                        context
                            .penalties
                            .iter()
                            .filter(|(token, _)| !penalty_free_tokens.contains(token))
                            .for_each(|(token, penalty)| logits[*token as usize] -= penalty);
                        context
                            .logit_bias
                            .iter()
                            .for_each(|(token, bias)| logits[*token as usize] += *bias);
                        logits
                    })
                })
                .collect::<Vec<_>>();

            let probs = model.softmax(logits)?;
            let output_tokens = payload
                .par_iter()
                .zip_eq(probs.into_par_iter())
                .map(|(context, x)| (context.as_ref(), x))
                .map(|(context, probs)| {
                    context
                        .zip(probs)
                        .map(|(context, probs)| context.sampler.sample(probs))
                })
                .collect::<Vec<_>>();

            for (context, done, token, tokens) in itertools::multizip((
                payload.iter_mut(),
                done.iter_mut(),
                output_tokens.into_iter(),
                tokens.into_iter(),
            )) {
                let mut finished = false;

                if let Some(context) = context {
                    let prefix = std::mem::take(&mut context.prefix);
                    let suffix = std::mem::take(&mut context.suffix);
                    let model_tokens = [prefix.0, suffix.0].concat();

                    // compute new prefix and suffix using the current remaining tokens
                    assert!(model_tokens.len() >= tokens.len());
                    let len = model_tokens.len() - tokens.len();
                    context.prefix = Tokens(model_tokens[..len].to_vec());
                    context.suffix = Tokens(model_tokens[len..].to_vec());
                    context
                        .penalties
                        .iter_mut()
                        .for_each(|(_, penalty)| *penalty *= context.sampler.penalty_decay);

                    if let Some(token) = token {
                        assert_eq!(context.suffix.len(), 0);
                        context.suffix.0.push(token);

                        let penalty = match context.penalties.get(&token) {
                            Some(penalty) => penalty + context.sampler.frequency_penalty,
                            None => context.sampler.presence_penalty,
                        };
                        context.penalties.insert(token, penalty);

                        let mut word = tokenizer.decode(&[token])?;
                        context.model_text.append(&mut word.clone());
                        context.output_buffer.append(&mut word);
                        context.model_tokens.push(token);

                        if let Ok(word) = String::from_utf8(context.output_buffer.clone()) {
                            let _ = context.sender.send(Token::Token(word));
                            context.output_buffer.clear();
                        }

                        let model_text = String::from_utf8_lossy(&context.model_text);
                        let count_tokens = || {
                            let prompt_tokens = context.prompt_tokens.len();
                            let completion_tokens = context.model_tokens.len();
                            let total_tokens = prompt_tokens + completion_tokens;
                            TokenCounter {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens,
                            }
                        };
                        let mut finish = |reason| {
                            let _ = context.sender.send(Token::Stop(reason, count_tokens()));
                            let _ = context.sender.send(Token::Done);
                            finished = true;
                        };

                        if context.stop.iter().any(|stop| model_text.contains(stop)) {
                            finish(FinishReason::Stop);
                        } else if context.model_tokens.len() >= context.max_tokens {
                            finish(FinishReason::Length);
                        }
                    }
                }

                if finished {
                    std::mem::swap(context, done);
                }
            }
        }

        Ok(())
    };

    loop {
        if let Err(err) = process() {
            log::error!("{err}");
        }
    }
}
