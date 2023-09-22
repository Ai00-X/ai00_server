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

use crate::{sampler::Sampler, Token};

#[derive(Debug)]
pub enum SlotResult {
    /// There is an idle slot ready to be picked up.
    Success(usize),
    /// An idle slot is swapped.
    Fault(usize),
    /// There is no idle slot left.
    Failure(Box<GenerateTask>),
}

#[derive(Debug)]
enum SlotState {
    /// The slot might be either picked up or swapped.
    Idle(Tokens),
    /// The slot is locked and is waiting for processing.
    Wait(Box<GenerateTask>),
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

impl<'a> Borrow<TokenSlice> for Tokens {
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
pub struct GenerateTask {
    /// Tokens that have been computed and cached.
    pub prefix: Tokens,
    /// Tokens to be computed.
    pub suffix: Tokens,
    /// Tokens that are output by the model, for calculating penalties.
    pub model_tokens: Tokens,

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
    pub fn queue(&mut self, task: GenerateTask) -> SlotResult {
        let tokens = Tokens([task.prefix, task.suffix].concat());
        let choice = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| match slot {
                SlotState::Idle(content) => {
                    if tokens.starts_with(&content) {
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
                GenerateTask {
                    prefix: Default::default(),
                    suffix: tokens,
                    ..task
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
                    GenerateTask {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        ..task
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
                    GenerateTask {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        ..task
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
        .into_iter()
        .filter(|token| {
            let word = tokenizer.decode(&[*token]).unwrap_or_default();
            let word = String::from_utf8(word).unwrap_or_default();
            word.contains('\n')
        })
        .collect::<HashSet<_>>();

    let mut payload: Vec<Option<GenerateTask>> = Default::default();
    let mut model: Option<Arc<Model>> = Default::default();
    let mut state: Option<ModelState> = Default::default();

    let mut process = || -> Result<()> {
        // take data from some waiting slots
        if let Some(runtime) = &mut *runtime.lock().unwrap() {
            payload.resize(runtime.slots.len(), None);
            model.replace(runtime.model.clone());
            state.replace(runtime.state.clone());

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
                    SlotState::Wait(task) => payload[batch].replace(*task),
                    _ => unreachable!(),
                };
            }
        }

        if let (Some(model), Some(state)) = (&model, &state) {
            let mut tokens = payload
                .iter()
                .map(|task| match task {
                    Some(task) => task.suffix.0.clone(),
                    None => vec![],
                })
                .collect_vec();

            let eliminate = |(task, x): (_, Vec<_>)| match x.len() {
                0 => (task, None),
                _ => (task, Some(x)),
            };

            let logits = model.run(&mut tokens, state)?;
            let logits =
                payload
                    .par_iter()
                    .zip_eq(logits.into_par_iter())
                    .map(|(task, x)| (task.as_ref(), x))
                    .map(eliminate)
                    .map(|(task, logits)| {
                        task.zip(logits).map(|(task, mut logits)| {
                            let mut occurrence: HashMap<u16, f32> = HashMap::new();
                            task.model_tokens.iter().rev().enumerate().for_each(
                                |(index, token)| {
                                    let ap = task.sampler.presence_penalty;
                                    let af = task.sampler.frequency_penalty;
                                    let ad = task.sampler.penalty_decay;
                                    let mut penalty = occurrence.remove(token).unwrap_or(ap);
                                    penalty += af * ad.powf(index as f32);
                                    occurrence.insert(*token, penalty);
                                },
                            );
                            occurrence
                                .into_iter()
                                .filter(|(token, _)| !penalty_free_tokens.contains(token))
                                .for_each(|(token, penalty)| logits[token as usize] -= penalty);
                            task.logit_bias
                                .iter()
                                .for_each(|(token, bias)| logits[*token as usize] += *bias);
                            logits
                        })
                    })
                    .map(|x| x.unwrap_or(vec![]))
                    .collect::<Vec<_>>();

            let probs = model.softmax(logits)?;
            let output_tokens = payload
                .par_iter()
                .zip_eq(probs.into_par_iter())
                .map(|(task, x)| (task.as_ref(), x))
                .map(eliminate)
                .map(|(task, probs)| {
                    task.zip(probs)
                        .map(|(task, probs)| task.sampler.sample(probs))
                })
                .collect::<Vec<_>>();
            for (batch, token) in output_tokens.into_iter().enumerate() {
                if let Some(token) = token {
                    tokens[batch].push(token);
                    todo!();
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
