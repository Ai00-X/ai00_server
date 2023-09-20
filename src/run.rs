use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use flume::{Receiver, Sender};
use qp_trie::Trie;
use web_rwkv::{
    model::{BackedState, Model, ModelState},
    tokenizer::Tokenizer,
};

use crate::{sampler::Sampler, GenerateRequest, Token};

#[derive(Debug)]
pub enum SlotResult {
    /// There is an idle slot ready to be picked up.
    Success(usize),
    /// An idle slot is swapped.
    Fault(usize),
    /// There is no idle slot left.
    Failure(GenerateTask),
}

#[derive(Debug)]
enum SlotState {
    /// The slot might be either picked up or swapped.
    Idle(Vec<u16>),
    /// The slot is locked and is waiting for processing.
    Wait(GenerateTask),
    /// The slot is currently under processing.
    Busy,
}

impl Default for SlotState {
    fn default() -> Self {
        Self::Idle(Vec::default())
    }
}

struct Tokens(Vec<u16>);

impl Borrow<[u8]> for Tokens {
    fn borrow(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
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
struct TokenSlice([u16]);

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

trait AsTokenSlice {
    fn as_token_slice(&self) -> &TokenSlice;
}

impl AsTokenSlice for [u16] {
    fn as_token_slice(&self) -> &TokenSlice {
        let ptr = self as *const [u16] as *const TokenSlice;
        unsafe { &*ptr }
    }
}

#[derive(Debug)]
pub struct GenerateTask {
    pub tokens: Vec<u16>,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub sampler: Sampler,
    pub occurrence: HashMap<u16, usize>,
    pub logit_bias: HashMap<u16, f32>,
    pub embed: bool,
    pub sender: Sender<Token>,
}

pub struct Runtime<'a> {
    model: Model<'a>,
    slots: Vec<SlotState>,
    state: ModelState,
    backed: Trie<Tokens, BackedState>,
}

impl<'a> Runtime<'a> {
    pub fn new(model: Model<'a>, state: ModelState) -> Self {
        Self {
            model,
            slots: Default::default(),
            state,
            backed: Default::default(),
        }
    }

    /// Queue a generation task.
    pub fn queue(&mut self, task: GenerateTask) -> SlotResult {
        let tokens = task.tokens;
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
            None => SlotResult::Failure(GenerateTask { tokens, ..task }),
            Some((batch, false, _)) => {
                let prefix = self.backed.longest_common_prefix(tokens.as_token_slice());
                let len = match self.backed.contains_key(prefix) {
                    true => prefix.len(),
                    false => 0,
                };
                let mut state = SlotState::Wait(GenerateTask {
                    tokens: tokens[len..].to_vec(),
                    ..task
                });

                let prefix = prefix.to_vec();
                let reload = self
                    .backed
                    .remove(prefix[..].as_token_slice())
                    .unwrap_or_else(|| BackedState::new(self.model.info(), 1));

                std::mem::swap(&mut state, &mut self.slots[batch]);
                match state {
                    SlotState::Idle(content) => {
                        let backed = self.state.back_batch(batch).unwrap();
                        self.backed.insert(Tokens(content), backed);
                        self.state.load_batch(&reload, batch).unwrap();
                        SlotResult::Fault(batch)
                    }
                    _ => unreachable!(),
                }
            }
            Some((id, true, len)) => {
                let state = SlotState::Wait(GenerateTask {
                    tokens: tokens[len..].to_vec(),
                    ..task
                });
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
}
