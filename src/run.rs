use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::Result;
use flume::{Receiver, Sender};
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
    Idle(Tokens, Instant),
    /// The slot is locked and is waiting for processing.
    Wait(Box<GenerateContext>),
    /// The slot is currently under processing.
    Busy,
}

impl Default for SlotState {
    fn default() -> Self {
        Self::Idle(Default::default(), Instant::now())
    }
}

#[derive(Debug, Default)]
enum Payload {
    #[default]
    Empty,
    Busy(Box<GenerateContext>),
    Done(Box<GenerateContext>),
}

impl Payload {
    /// Takes out the value if `self` is [`Payload::Done`], and reset `self` to [`Payload::Empty`].
    fn take(&mut self) -> Option<Box<GenerateContext>> {
        match std::mem::take(self) {
            Payload::Done(context) => Some(context),
            payload => {
                *self = payload;
                None
            }
        }
    }

    /// Set `self` to [`Payload::Done`] if `self` is [`Payload::Busy`].
    fn finalize(&mut self) {
        *self = match std::mem::take(self) {
            Payload::Busy(context) => Payload::Done(context),
            payload => payload,
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
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
        self.0[..loc >> 1].as_token_slice()
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
    embed_layer: usize,
}

impl<'a> Runtime<'a> {
    pub fn new(
        model: Arc<Model<'a>>,
        state: ModelState,
        max_runtime_batch: usize,
        embed_layer: usize,
    ) -> Self {
        let slots = (0..state.max_batch())
            .map(|_| SlotState::default())
            .collect();

        Self {
            model,
            slots,
            state,
            backed: Trie::new(),
            max_runtime_batch,
            embed_layer,
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
                SlotState::Idle(content, time) => match tokens.starts_with(content) {
                    true => Some((index, true, content.len(), time.elapsed().as_micros())),
                    false => Some((index, false, 0, time.elapsed().as_micros())),
                },
                _ => None,
            })
            .max_by(|&lhs, &rhs| lhs.2.cmp(&rhs.2).then(lhs.3.cmp(&rhs.3)));
        match choice {
            None => SlotResult::Failure(
                GenerateContext {
                    prefix: Default::default(),
                    suffix: tokens,
                    ..context
                }
                .into(),
            ),
            Some((batch, false, _, _)) => {
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
                    SlotState::Idle(content, _) => {
                        let backed = self.state.back_batch(batch).expect("back state");
                        self.backed.insert(content, backed);
                        self.state.load_batch(&reload, batch).expect("load state");
                        SlotResult::Fault(batch)
                    }
                    _ => unreachable!(),
                }
            }
            Some((id, true, len, _)) => {
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

    /// Manually back an idle slot.
    pub fn back(&mut self, batch: usize) -> Option<BackedState> {
        match &self.slots[batch] {
            SlotState::Idle(content, _) => {
                let backed = self.state.back_batch(batch).expect("back state");
                self.backed.insert(content.clone(), backed.clone());
                Some(backed)
            }
            _ => None,
        }
    }
}

pub fn run(runtime: Arc<Mutex<Option<Runtime>>>, tokenizer: Tokenizer, receiver: Receiver<()>) {
    let penalty_free_tokens = (0..u16::MAX)
        .filter(|token| {
            let word = tokenizer.decode(&[*token]).unwrap_or_default();
            let word = String::from_utf8(word).unwrap_or_default();
            word.contains('\n')
        })
        .collect::<HashSet<_>>();

    let mut payloads: Vec<Payload> = Default::default();
    let mut model: Option<Arc<Model>> = Default::default();
    let mut state: Option<ModelState> = Default::default();

    let payloads_occupancy = |payloads: &[Payload]| {
        payloads
            .iter()
            .filter(|x| matches!(x, Payload::Busy(_)))
            .count()
    };

    let mut process = |payloads: &mut Vec<Payload>| -> Result<()> {
        if let Some(runtime) = &mut *runtime.lock().unwrap() {
            payloads.resize_with(runtime.slots.len(), Default::default);
            model.replace(runtime.model.clone());
            state.replace(runtime.state.clone());

            // reset all finished slots to idle
            payloads
                .iter_mut()
                .enumerate()
                .for_each(|(batch, payload)| {
                    if let Some(context) = payload.take() {
                        assert!(matches!(runtime.slots[batch], SlotState::Busy));
                        runtime.slots[batch] = SlotState::Idle(context.prefix, Instant::now());

                        if let Some(backed) = runtime.back(batch) {
                            log::info!("manually backed slot {}", batch);
                            if context.embed {
                                let num_emb = backed.shape[0];
                                let num_layer = backed.shape[1] / 5;
                                let embed_layer = runtime.embed_layer;

                                let start = ((num_layer - embed_layer) * 5 + 4) * num_emb;
                                let end = start + num_emb;
                                let _ = context
                                    .sender
                                    .send(Token::Embed(backed.data[start..end].to_vec()));
                            }
                        }
                    }
                });

            // take data from some waiting slots
            let occupancy = payloads_occupancy(payloads);
            let remain = runtime.max_runtime_batch - runtime.max_runtime_batch.min(occupancy);
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
                        assert!(matches!(payloads[batch], Payload::Empty));
                        payloads[batch] = Payload::Busy(context);
                    }
                    _ => unreachable!(),
                };
            }
        } else {
            payloads.clear();
            model = None;
            state = None;
        }

        if let (Some(model), Some(state)) = (&model, &state) {
            let mut input_tokens = payloads
                .iter()
                .map(|payload| match payload {
                    Payload::Busy(context) => context.suffix.0.clone(),
                    _ => vec![],
                })
                .collect_vec();

            // run the model until there is at least one slot finished
            let occupancy = payloads_occupancy(payloads);
            let logits = match occupancy {
                0 => vec![None; payloads.len()],
                _ => loop {
                    let logits = model.run(&mut input_tokens, state)?;
                    if logits.iter().any(Option::is_some) {
                        break logits;
                    }
                },
            };
            // post-process logits
            let logits = payloads
                .par_iter()
                .zip_eq(logits.into_par_iter())
                .map(|(payload, logits)| match payload {
                    Payload::Busy(context) => logits.map(|mut logits| {
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
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>();

            let probs = match occupancy {
                0 => vec![None; payloads.len()],
                _ => model.softmax(logits)?,
            };
            let output_tokens = payloads
                .par_iter()
                .zip_eq(probs.into_par_iter())
                .map(|(payload, probs)| match payload {
                    Payload::Busy(context) => probs.map(|probs| context.sampler.sample(probs)),
                    _ => None,
                })
                .collect::<Vec<_>>();

            for (payload, token, tokens) in itertools::multizip((
                payloads.iter_mut(),
                output_tokens.into_iter(),
                input_tokens.into_iter(),
            )) {
                let mut done = false;

                if let Payload::Busy(context) = payload {
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
                            done = true;
                        };

                        if context.sender.is_disconnected() {
                            done = true;
                        } else if context.stop.iter().any(|stop| model_text.contains(stop)) {
                            finish(FinishReason::Stop);
                        } else if context.model_tokens.len() >= context.max_tokens {
                            finish(FinishReason::Length);
                        }
                    }
                }

                if done {
                    payload.finalize();
                }
            }
        }

        Ok(())
    };

    loop {
        let _ = receiver.recv();

        'run: loop {
            if let Err(err) = process(&mut payloads) {
                log::error!("{err}");
            }
            if payloads.iter().all(Payload::is_empty) {
                break 'run;
            }
        }
    }
}
