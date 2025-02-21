use std::{
    cmp::Ordering,
    collections::HashMap,
    error::Error,
    ops::Deref,
    sync::{Arc, Weak},
    time::Duration,
};

use anyhow::Result;
use derivative::Derivative;
use flume::{Receiver, Sender, TryRecvError};
use itertools::Itertools;
use qp_trie::Trie;
use tokio::{
    sync::{Mutex, RwLock},
    task::JoinHandle,
    time::Instant,
};
use web_rwkv::{
    context::Context,
    runtime::{
        model::{ModelInfo, State},
        Runtime,
    },
    tensor::TensorCpu,
    tokenizer::Tokenizer,
};

use crate::{
    sampler::{bnf::BnfSampler, Formatter},
    GenerateKind, GenerateRequest, InitState, ReloadRequest, StateId, Token,
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
    /// The slot is currently under processing.
    Busy(JoinHandle<GenerateContext>),
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

impl GenerateContext {
    pub async fn new(
        request: GenerateRequest,
        sender: Sender<Token>,
        tokenizer: &Tokenizer,
    ) -> Result<Self> {
        let tokens = Tokens(tokenizer.encode(request.prompt.as_bytes())?);
        let model_tokens = Tokens(tokenizer.encode(request.model_text.as_bytes())?);

        // init sampler state here
        request.sampler.write().await.init(&model_tokens);

        let choices = match &request.kind {
            GenerateKind::Choose { choices, .. } => {
                let choices: Vec<_> = choices
                    .iter()
                    .map(|prompt| tokenizer.encode(prompt.as_bytes()))
                    .try_collect()?;
                choices.into_iter().map(Tokens).collect()
            }
            _ => Vec::new(),
        };
        Ok(Self {
            prompt_tokens: tokens.to_vec(),
            prompt_cached: Default::default(),
            prefix: Default::default(),
            suffix: tokens,
            output: None,
            choices,
            model_text: Vec::new(),
            buffer: Vec::new(),
            model_tokens: Vec::new(),
            formatters: Vec::new(),
            instant: None,
            request,
            sender,
        })
    }
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

#[derive(Derivative, Clone)]
#[derivative(Debug)]
struct CoreRuntime {
    context: Context,
    reload: Arc<ReloadRequest>,
    info: ModelInfo,
    #[derivative(Debug = "ignore")]
    state: Arc<dyn State + Send + Sync>,
    #[derivative(Debug = "ignore")]
    runtime: Weak<dyn Runtime + Send + Sync>,
    tokenizer: Arc<Tokenizer>,
    slots: Arc<Mutex<Vec<SlotState>>>,
    caches: Arc<Mutex<CacheHub>>,
}

impl CoreRuntime {
    /// Search for the longest common prefix in the memory cache and checkout the state from that point.
    /// Should there be a cache miss, an initial state is returned.
    async fn checkout(&self, id: StateId, tokens: &[u16]) -> CacheCheckout {
        let mut caches = self.caches.lock().await;

        let Cache { state, cache } = caches.fetch(id);
        let prefix = cache.longest_common_prefix(tokens.as_token_slice());
        let len = (1..=prefix.len())
            .rev()
            .find(|len| cache.contains_key(prefix[0..*len].as_token_slice()))
            .unwrap_or_default();
        let prefix = prefix[0..len].to_vec();

        let state = state.clone().map(|state| state.data);
        let item = cache.get(prefix[..].as_token_slice()).cloned();
        drop(caches);

        match item {
            Some(sender) => {
                let mut receiver = sender.subscribe();
                let item = loop {
                    if let Some(item) = receiver.borrow_and_update().deref().clone() {
                        break item;
                    }
                    let _ = receiver.changed().await;
                };
                let item = CachedItem::update(item);
                sender.send_replace(Some(item.clone()));
                CacheCheckout {
                    prefix,
                    state: item.state,
                    output: Some(item.output),
                }
            }
            None => {
                let prefix = vec![];
                let state = state.unwrap_or_else(|| self.state.init());
                CacheCheckout {
                    prefix,
                    state,
                    output: None,
                }
            }
        }
    }

    /// Queue an inference task.
    async fn queue(&self, context: GenerateContext) -> Result<SlotResult> {
        let tokens = match [context.prefix, context.suffix].concat() {
            tokens if tokens.is_empty() => vec![0u16],
            tokens => tokens,
        };

        // compile the BNF schema.
        let mut formatters = Vec::<Arc<RwLock<dyn Formatter + Send + Sync>>>::new();
        if let Some(schema) = context.request.bnf_schema.clone() {
            match BnfSampler::new(&self.tokenizer, &schema) {
                Ok(bnf) => formatters.push(Arc::new(RwLock::new(bnf))),
                Err(err) => return Ok(SlotResult::Error(err.into())),
            }
        }

        // find the best idle slot by:
        // 1. find the slot that matches the context (continue)
        // 2. find an empty slot
        // 3. find the oldest non-empty slot
        let mut slots = self.slots.lock().await;
        let choice = slots
            .iter()
            .enumerate()
            .filter_map(|(batch, slot)| match slot {
                SlotState::Idle(content, time) => {
                    let delta = time.elapsed().as_millis();
                    match (content.is_empty(), tokens.starts_with(content)) {
                        (true, _) => Some((SlotChoice::Empty(batch), delta)),
                        (false, true) => Some((SlotChoice::Continue(batch, content.len()), delta)),
                        (false, false) => Some((SlotChoice::Back(batch), delta)),
                    }
                }
                _ => None,
            })
            .max_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)))
            .map(|(x, _)| x);

        match choice {
            // we cannot find a slot because all slots are occupied
            // in this case, we hand the request back to the caller
            None => Ok(SlotResult::Failure(
                GenerateContext {
                    prefix: Default::default(),
                    suffix: Tokens(tokens),
                    formatters,
                    ..context
                }
                .into(),
            )),
            // back a non-relative and non-empty slot and use it for our new context
            Some(SlotChoice::Back(batch)) => {
                log::info!("[queue][back][slot: {}]", batch);
                let checkout = self.checkout(context.request.state, &tokens).await;
                self.state.load(checkout.state, batch)?;

                let len = checkout.prefix.len();
                assert!(len == 0 || (len > 0 && checkout.output.is_some()));
                log::info!("[cache][slot: {}][len: {}]", batch, len);

                let context = GenerateContext {
                    prefix: Tokens(tokens[..len].to_vec()),
                    suffix: Tokens(tokens[len..].to_vec()),
                    output: checkout.output,
                    formatters,
                    ..context
                };
                let runtime = self.clone();
                slots[batch] = SlotState::Busy(tokio::spawn(runtime.process(context)));
                Ok(SlotResult::Fault(batch))
            }
            // directly occupy an empty slot so no need backing
            Some(SlotChoice::Empty(batch)) => {
                log::info!("[queue][empty][slot: {}]", batch);
                let checkout = self.checkout(context.request.state, &tokens).await;
                self.state.load(checkout.state, batch)?;

                let len = checkout.prefix.len();
                assert!(len == 0 || (len > 0 && checkout.output.is_some()));
                log::info!("[cache][slot: {}][len: {}]", batch, len);

                let context = GenerateContext {
                    prefix: Tokens(tokens[..len].to_vec()),
                    suffix: Tokens(tokens[len..].to_vec()),
                    output: checkout.output,
                    formatters,
                    ..context
                };
                let runtime = self.clone();
                slots[batch] = SlotState::Busy(tokio::spawn(runtime.process(context)));
                Ok(SlotResult::Success(batch))
            }
            Some(SlotChoice::Continue(batch, ..)) => {
                log::info!("[queue][continue][slot: {}]", batch);
                let checkout = self.checkout(context.request.state, &tokens).await;
                self.state.load(checkout.state, batch)?;

                let len = checkout.prefix.len();
                assert!(len == 0 || (len > 0 && checkout.output.is_some()));
                log::info!("[cache][slot: {}][len: {}]", batch, len);

                let context = GenerateContext {
                    prefix: Tokens(tokens[..len].to_vec()),
                    suffix: Tokens(tokens[len..].to_vec()),
                    output: checkout.output,
                    formatters,
                    ..context
                };
                let runtime = self.clone();
                slots[batch] = SlotState::Busy(tokio::spawn(runtime.process(context)));
                Ok(SlotResult::Success(batch))
            }
        }
    }

    async fn process(self, context: GenerateContext) -> GenerateContext {
        tokio::time::sleep(Duration::from_secs_f32(1.0)).await;
        context
    }
}

async fn enqueue(runtime: CoreRuntime, receiver: Receiver<GenerateContext>, timer: Duration) {
    let mut queue = Vec::<GenerateContext>::new();
    loop {
        match receiver.try_recv() {
            Ok(request) => queue.push(request),
            Err(TryRecvError::Empty) => tokio::time::sleep(timer).await,
            Err(TryRecvError::Disconnected) => break,
        }

        let mut temp = Vec::new();
        for context in queue.drain(..) {
            match runtime.queue(context).await {
                Ok(SlotResult::Failure(context)) => temp.push(*context),
                Ok(SlotResult::Success(batch)) => log::info!("[enqueue][success][slot: {batch}]"),
                Ok(SlotResult::Fault(batch)) => log::info!("[enqueue][fault][slot: {batch}]"),
                Ok(SlotResult::Error(err)) => log::error!("[enqueue][error] {err}"),
                Err(err) => log::error!("[enqueue][error] {err}"),
            }
        }
        std::mem::swap(&mut queue, &mut temp);
    }
}

pub async fn run(
    context: Context,
    info: ModelInfo,
    reload: Arc<ReloadRequest>,
    runtime: Weak<dyn Runtime + Send + Sync>,
    state: Arc<dyn State + Send + Sync>,
    tokenizer: Arc<Tokenizer>,
    receiver: Receiver<GenerateContext>,
) {
    let slots = std::iter::repeat_with(Default::default)
        .take(reload.max_batch)
        .collect();
    let slots = Arc::new(Mutex::new(slots));
    let caches = Arc::new(Mutex::new(Default::default()));

    let runtime = CoreRuntime {
        context,
        reload,
        info,
        state,
        runtime,
        tokenizer,
        slots,
        caches,
    };
    tokio::spawn(enqueue(runtime, receiver, Duration::from_secs_f32(1.0)));
}
