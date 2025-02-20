use std::{
    cmp::Ordering,
    collections::HashMap,
    error::Error,
    sync::{Arc, Weak},
    time::Duration,
};

use anyhow::Result;
use derivative::Derivative;
use flume::{Receiver, Sender};
use itertools::Itertools;
use qp_trie::Trie;
use salvo::oapi::ToSchema;
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{Mutex, RwLock},
    time::Instant,
};
use web_rwkv::{
    context::Context,
    runtime::{
        infer::{InferInput, InferOutput},
        model::{ModelInfo, State},
        Runtime, TokioRuntime,
    },
    tensor::TensorCpu,
    tokenizer::Tokenizer,
};

use crate::{
    sampler::{bnf::BnfSampler, Formatter},
    GenerateRequest, ReloadRequest, ThreadRequest, Token,
};

const MIN_PROMPT_CACHE_TOKENS: usize = 32;
const MAX_CACHE_ITEMS: usize = 256;

/// The result of trying to queuing a task.
#[derive(Debug)]
enum SlotResult {
    /// There is an idle slot ready to be picked up.
    Success(usize),
    /// An idle slot is swapped.
    Fault(usize),
    /// There is no idle slot left.
    Failure(Box<GenerateContext>),
    /// An error occurred.
    Error(Box<dyn Error>),
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

#[derive(Debug, PartialEq, Eq)]
enum SlotChoice {
    Continue(usize, usize),
    Back(usize),
    Empty(usize),
}

impl std::cmp::Ord for SlotChoice {
    fn cmp(&self, other: &Self) -> Ordering {
        // priority: continue > empty > back
        use SlotChoice::{Back, Continue, Empty};
        match (self, other) {
            (Continue(_, x), Continue(_, y)) => x.cmp(y),
            (Continue(_, _), _) => Ordering::Greater,
            (_, Continue(_, _)) => Ordering::Less,
            (Empty(_), Empty(_)) => Ordering::Equal,
            (Empty(_), Back(_)) => Ordering::Greater,
            (Back(_), Empty(_)) => Ordering::Less,
            (Back(_), Back(_)) => Ordering::Equal,
        }
    }
}

impl std::cmp::PartialOrd for SlotChoice {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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

impl std::borrow::Borrow<[u8]> for Tokens {
    fn borrow(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

impl std::borrow::Borrow<[u16]> for Tokens {
    fn borrow(&self) -> &[u16] {
        &self.0
    }
}

impl std::borrow::Borrow<TokenSlice> for Tokens {
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

impl std::borrow::Borrow<[u8]> for TokenSlice {
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

#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct GenerateContext {
    /// Tokens that are provided at first.
    pub prompt_tokens: Vec<u16>,
    /// Whether the prompt has already been processed and cached.
    pub prompt_cached: CachedPrompt,
    /// Tokens that have been computed and cached.
    pub prefix: Tokens,
    /// Tokens to be computed.
    pub suffix: Tokens,
    /// The output of the model from the last run.
    pub output: Option<TensorCpu<f32>>,
    /// Tokens to be chosen if this is a choose request.
    pub choices: Vec<Tokens>,
    /// Texts that are output by the model.
    pub model_text: Vec<u8>,
    /// Model may output partial utf-8. This makes sure the output is always valid.
    pub buffer: Vec<u8>,
    /// Tokens that are output by the model.
    pub model_tokens: Vec<u16>,
    /// Compiled BNF schema, if any.
    #[derivative(Debug = "ignore")]
    pub formatters: Vec<Arc<RwLock<dyn Formatter + Send + Sync>>>,
    /// For measuring time used.
    pub instant: Option<Instant>,
    /// Generate request provided by the caller.
    pub request: GenerateRequest,
    /// To send back generated tokens.
    pub sender: Sender<Token>,
}

#[derive(Debug, Default, Clone)]
pub enum CachedPrompt {
    #[default]
    None,
    Future(tokio::sync::watch::Sender<Option<CachedItem>>),
    Done,
}

/// An item that a cache slot holds, including a state, last model output and a timestamp.
#[derive(Debug, Clone)]
pub struct CachedItem {
    state: TensorCpu<f32>,
    output: TensorCpu<f32>,
    instant: Instant,
}

impl CachedItem {
    pub fn new(state: TensorCpu<f32>, output: TensorCpu<f32>) -> Self {
        Self {
            state,
            output,
            instant: Instant::now(),
        }
    }

    /// Update an existing cache item's timestamp.
    pub fn update(cached: CachedItem) -> Self {
        Self {
            instant: Instant::now(),
            ..cached
        }
    }
}

struct CacheCheckout {
    prefix: Vec<u16>,
    state: TensorCpu<f32>,
    output: Option<TensorCpu<f32>>,
}

#[derive(Debug, Default)]
struct Cache {
    state: Option<InitState>,
    cache: Trie<Tokens, tokio::sync::watch::Sender<Option<CachedItem>>>,
}

impl Cache {
    fn maintain(&mut self) {
        let cache = &mut self.cache;
        if cache.count() <= MAX_CACHE_ITEMS {
            return;
        }

        let mut remove = vec![];
        for (tokens, _) in cache
            .iter()
            .filter_map(|(tokens, item)| item.borrow().clone().map(|item| (tokens, item)))
            .sorted_unstable_by_key(|(_, item)| item.instant.elapsed())
            .skip(MAX_CACHE_ITEMS)
        {
            remove.push(tokens.to_owned());
        }

        for tokens in remove.into_iter() {
            cache.remove(&tokens);
        }
    }
}

#[derive(Debug, Default)]
struct CacheHub {
    backed: HashMap<StateId, Cache>,
    default: Cache,
}

impl CacheHub {
    fn fetch(&mut self, id: StateId) -> &mut Cache {
        match self.backed.get_mut(&id) {
            Some(item) => item,
            None => &mut self.default,
        }
    }
}

#[derive(
    Derivative, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema,
)]
#[derivative(Debug = "transparent")]
#[serde(transparent)]
pub struct StateId(uuid::Uuid);

impl StateId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct InitState {
    pub name: String,
    pub id: StateId,
    pub default: bool,
    #[derivative(Debug = "ignore")]
    pub data: TensorCpu<f32>,
}

struct Model<M>(M);

trait ModelSerialize {
    fn serialize(&self, file: std::fs::File) -> Result<()>;
}

impl<M: Serialize> ModelSerialize for Model<M> {
    fn serialize(&self, file: std::fs::File) -> Result<()> {
        use cbor4ii::{core::enc::Write, serde::Serializer};
        use std::{fs::File, io::Write as _};

        struct FileWriter(File);
        impl Write for FileWriter {
            type Error = std::io::Error;
            fn push(&mut self, input: &[u8]) -> Result<(), Self::Error> {
                self.0.write_all(input)
            }
        }

        let file = FileWriter(file);
        let mut serializer = Serializer::new(file);
        self.0.serialize(&mut serializer)?;

        Ok(())
    }
}

async fn enqueue(
    context: GenerateContext,
    tokenizer: Arc<Tokenizer>,
    slots: Vec<Arc<RwLock<SlotState>>>,
) -> Result<SlotResult> {
    let tokens = match [context.prefix, context.suffix].concat() {
        tokens if tokens.is_empty() => vec![0u16],
        tokens => tokens,
    };

    // compile the BNF schema.
    let mut formatters = Vec::<Arc<RwLock<dyn Formatter + Send + Sync>>>::new();
    if let Some(schema) = context.request.bnf_schema.clone() {
        match BnfSampler::new(&tokenizer, &schema) {
            Ok(bnf) => formatters.push(Arc::new(RwLock::new(bnf))),
            Err(err) => return Ok(SlotResult::Error(err.into())),
        }
    }

    // find the best idle slot by:
    // 1. find the slot that matches the context (continue)
    // 2. find an empty slot
    // 3. find the oldest non-empty slot
    let choice = slots
        .iter()
        .enumerate()
        .filter_map(|(batch, slot)| {
            let slot = slot.try_read().ok()?;
            match &*slot {
                SlotState::Idle(content, instant) => {
                    let instant = instant.clone();
                    match (content.is_empty(), tokens.starts_with(content)) {
                        (true, _) => Some((SlotChoice::Empty(batch), instant)),
                        (_, true) => Some((SlotChoice::Continue(batch, content.len()), instant)),
                        (_, false) => Some((SlotChoice::Back(batch), instant)),
                    }
                }
                _ => None,
            }
        })
        .max_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)));

    todo!()
}

pub async fn serve(
    context: Context,
    info: ModelInfo,
    reload: Arc<ReloadRequest>,
    runtime: Weak<dyn Runtime>,
    state: Arc<dyn State + Send + Sync>,
    tokenizer: Arc<Tokenizer>,
    receiver: Receiver<GenerateRequest>,
) {
    let queue = Arc::new(Mutex::new(Vec::<GenerateContext>::new()));
    let slots = vec![Arc::new(RwLock::new(SlotState::default())); reload.max_batch];

    let enqueue = {
        let (queue, slots) = (queue, slots).clone();
        move || async move {}
    };
    tokio::spawn(async move {
        loop {
            enqueue().await;
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    while let Ok(request) = receiver.recv_async().await {}
}
