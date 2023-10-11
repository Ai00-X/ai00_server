use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    convert::Infallible,
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
    model::{v4, v5, BackedState, FromBuilder, Model, ModelState, StateBuilder},
    tokenizer::Tokenizer,
};

use crate::{sampler::Sampler, FinishReason, Token, TokenCounter, STATE_CHUNK_SIZE};

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

pub struct Runtime<M, S, B>
where
    B: BackedState,
    S: ModelState<BackedState = B>,
    M: Model<ModelState = S>,
{
    model: M,
    state: S,
    slots: Mutex<Vec<SlotState>>,
    backed: Mutex<Trie<Tokens, B>>,
    max_runtime_batch: usize,
    embed_layer: usize,
}

impl<M, S, B> Runtime<M, S, B>
where
    for<'a> B: BackedState + Clone + FromBuilder<Builder<'a> = StateBuilder, Error = Infallible>,
    S: ModelState<BackedState = B>,
    M: Model<ModelState = S>,
{
    pub fn new(model: M, state: S, max_runtime_batch: usize, embed_layer: usize) -> Self {
        let slots = (0..state.max_batch())
            .map(|_| SlotState::default())
            .collect();

        Self {
            model,
            state,
            slots: Mutex::new(slots),
            backed: Mutex::new(Trie::new()),
            max_runtime_batch,
            embed_layer,
        }
    }

    /// Queue a generation task.
    pub fn queue(&self, context: GenerateContext) -> SlotResult {
        let mut slots = self.slots.lock().unwrap();
        let mut cache = self.backed.lock().unwrap();

        let tokens = Tokens([context.prefix, context.suffix].concat());
        let choice = slots
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
                let prefix = cache.longest_common_prefix(&tokens);
                let len = match cache.contains_key(prefix) {
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
                let reload = cache
                    .remove(prefix[..].as_token_slice())
                    .unwrap_or_else(|| {
                        let context = self.model.context();
                        let info = self.model.info();
                        StateBuilder::new(context, info)
                            .with_max_batch(1)
                            .with_chunk_size(STATE_CHUNK_SIZE)
                            .build_backed()
                    });

                std::mem::swap(&mut state, &mut slots[batch]);
                match state {
                    SlotState::Idle(content, _) => {
                        let backed = self.state.back_batch(batch).expect("back state");
                        cache.insert(content, backed);
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
                let _ = std::mem::replace(&mut slots[id], state);
                SlotResult::Success(id)
            }
        }
    }

    fn process(
        &self,
        tokenizer: &Tokenizer,
        penalty_free_tokens: &HashSet<u16>,
        payloads: &mut Vec<Payload>,
    ) -> Result<()> {
        {
            let mut slots = self.slots.lock().unwrap();
            let mut cache = self.backed.lock().unwrap();

            payloads.resize_with(self.state.max_batch(), Default::default);

            // reset all finished slots to idle
            payloads
                .iter_mut()
                .enumerate()
                .for_each(|(batch, payload)| {
                    if let Some(context) = payload.take() {
                        assert!(matches!(slots[batch], SlotState::Busy));
                        slots[batch] = SlotState::Idle(context.prefix, Instant::now());

                        if let Some(backed) = match &slots[batch] {
                            SlotState::Idle(content, _) => {
                                log::info!("manually backed slot {}", batch);
                                let backed = self.state.back_batch(batch).expect("back state");
                                cache.insert(content.clone(), backed.clone());
                                Some(backed)
                            }
                            _ => None,
                        } {
                            if context.embed {
                                let embed = backed.embed(0, self.embed_layer);
                                let _ = context.sender.send(Token::Embed(embed));
                            }
                        }
                    }
                });

            // take data from some waiting slots
            let occupancy = payloads
                .iter()
                .filter(|x| matches!(x, Payload::Busy(_)))
                .count();
            let remain = self.max_runtime_batch - self.max_runtime_batch.min(occupancy);
            let batches = slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| matches!(slot, SlotState::Wait(_)))
                .take(remain)
                .map(|(batch, _)| batch)
                .collect_vec();
            for batch in batches {
                let mut slot = SlotState::Busy;
                std::mem::swap(&mut slots[batch], &mut slot);
                match slot {
                    SlotState::Wait(context) => {
                        let _ = context.sender.send(Token::Start);
                        assert!(matches!(payloads[batch], Payload::Empty));
                        payloads[batch] = Payload::Busy(context);
                    }
                    _ => unreachable!(),
                };
            }
        }

        let mut input_tokens = payloads
            .iter()
            .map(|payload| match payload {
                Payload::Busy(context) => context.suffix.0.clone(),
                _ => vec![],
            })
            .collect_vec();

        // run the model until there is at least one slot finished
        let occupancy = payloads
            .iter()
            .filter(|x| matches!(x, Payload::Busy(_)))
            .count();
        let logits = match occupancy {
            0 => vec![None; payloads.len()],
            _ => loop {
                let logits = self.model.run(&mut input_tokens, &self.state)?;
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
            _ => self.model.softmax(logits)?,
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

        Ok(())
    }
}

pub enum RuntimeUntyped<'a> {
    V4(Runtime<v4::Model<'a>, v4::ModelState, v4::BackedState>),
    V5(Runtime<v5::Model<'a>, v5::ModelState, v5::BackedState>),
}

impl RuntimeUntyped<'_> {
    /// Queue a generation task.
    pub fn queue(&self, context: GenerateContext) -> SlotResult {
        match self {
            RuntimeUntyped::V4(runtime) => runtime.queue(context),
            RuntimeUntyped::V5(runtime) => runtime.queue(context),
        }
    }

    fn process(
        &self,
        tokenizer: &Tokenizer,
        penalty_free_tokens: &HashSet<u16>,
        payloads: &mut Vec<Payload>,
    ) -> Result<()> {
        match self {
            RuntimeUntyped::V4(runtime) => {
                runtime.process(tokenizer, penalty_free_tokens, payloads)
            }
            RuntimeUntyped::V5(runtime) => {
                runtime.process(tokenizer, penalty_free_tokens, payloads)
            }
        }
    }
}

pub fn run<'a>(tokenizer: Tokenizer, receiver: Receiver<Option<Arc<RuntimeUntyped<'a>>>>) {
    let penalty_free_tokens = (0..u16::MAX)
        .filter(|token| {
            let word = tokenizer.decode(&[*token]).unwrap_or_default();
            let word = String::from_utf8(word).unwrap_or_default();
            word.contains('\n')
        })
        .collect::<HashSet<_>>();

    let mut runtime: Option<Arc<RuntimeUntyped<'a>>>;
    let mut payloads = Vec::new();

    loop {
        runtime = receiver.recv().unwrap();

        if let Some(runtime) = &runtime {
            'run: loop {
                if let Err(err) = runtime.process(&tokenizer, &penalty_free_tokens, &mut payloads) {
                    log::error!("{}", err);
                }
                if payloads.iter().all(Payload::is_empty) {
                    break 'run;
                }
            }
        }
    }
}
