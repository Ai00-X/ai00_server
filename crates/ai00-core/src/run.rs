use std::{
    cmp::Ordering,
    collections::HashMap,
    ops::Deref,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use derivative::Derivative;
use flume::{Receiver, Sender};
use itertools::Itertools;
use qp_trie::Trie;
use salvo::oapi::ToSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use web_rwkv::{
    context::Context,
    runtime::{
        infer::{InferInfo, InferInput, InferInputBatch, InferOption, InferOutput},
        model::{Bundle, ModelInfo, State},
        softmax::softmax,
        Dispatcher, Job, TokioRuntime,
    },
    tensor::{TensorCpu, TensorInit},
    tokenizer::Tokenizer,
};

use crate::{
    sampler::{bnf::BnfSampler, Formatter},
    Environment, FinishReason, GenerateKind, GenerateRequest, ReloadRequest, Token, TokenCounter,
};

const END_OF_LINE_TOKEN: u16 = 261;
const MIN_PROMPT_CACHE_TOKENS: usize = 32;
const MAX_CACHE_ITEMS: usize = 256;

#[derive(Debug)]
pub enum SlotResult {
    /// There is an idle slot ready to be picked up.
    Success(usize),
    /// An idle slot is swapped.
    Fault(usize),
    /// There is no idle slot left.
    Failure(Box<GenerateContext>),
    /// An error occurred.
    Error(String),
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

#[derive(Debug, Clone, Default)]
pub enum Payload {
    #[default]
    Empty,
    Busy(GenerateContext),
    Done(GenerateContext),
}

impl Payload {
    /// Takes out the value if `self` is [`Payload::Done`], and reset `self` to [`Payload::Empty`].
    pub fn take(&mut self) -> Option<GenerateContext> {
        match std::mem::take(self) {
            Payload::Done(context) => Some(context),
            payload => {
                *self = payload;
                None
            }
        }
    }

    /// Set `self` to [`Payload::Done`] if `self` is [`Payload::Busy`].
    pub fn finalize(&mut self) {
        *self = match std::mem::take(self) {
            Payload::Busy(context) => Payload::Done(context),
            payload => payload,
        }
    }

    /// Returns `true` if the payload is [`Empty`].
    ///
    /// [`Empty`]: Payload::Empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
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

impl Environment {
    pub async fn enqueue(&self, context: GenerateContext) -> Vec<GenerateContext> {
        let mut queue = vec![];
        match self {
            Environment::Loaded(runtime) => {
                match runtime.queue(context).await.expect("queue task error") {
                    SlotResult::Success(batch) => log::info!("queued task at slot {batch}"),
                    SlotResult::Fault(batch) => log::info!("swapped task at slot {batch}"),
                    SlotResult::Failure(context) => queue.push(*context),
                    SlotResult::Error(reason) => log::warn!("queue task failed: {}", reason),
                }
            }
            Environment::None => queue.push(context),
        };
        queue
    }
}

pub struct Runtime {
    context: Context,
    reload: ReloadRequest,
    info: ModelInfo,
    state: Arc<dyn State + Send + Sync>,
    model: Arc<dyn ModelSerialize + Send + Sync>,
    runtime: TokioRuntime<InferInput, InferOutput>,
    tokenizer: Arc<Tokenizer>,
    slots: Mutex<Vec<SlotState>>,
    caches: Mutex<CacheHub>,
}

impl Runtime {
    pub async fn new<J, B>(
        context: Context,
        bundle: B,
        reload: ReloadRequest,
        states: Vec<InitState>,
        tokenizer: Tokenizer,
    ) -> Self
    where
        J: Job<Input = InferInput, Output = InferOutput> + Send + 'static,
        B: Dispatcher<J, Info = InferInfo> + Bundle + Clone + Send + 'static,
    {
        let slots = (0..reload.max_batch)
            .map(|_| SlotState::default())
            .collect();

        let mut caches = CacheHub::default();

        // set up default initial state
        if let Some(state) = states.iter().find(|state| state.default) {
            caches.default.state = Some(state.clone());
        }
        for state in states {
            let id = state.id;
            let item = Cache {
                state: Some(state),
                cache: Trie::new(),
            };
            caches.backed.insert(id, item);
        }

        let info = bundle.info();
        let state = Arc::new(bundle.state());
        let model = Arc::new(Model(bundle.model()));
        let runtime = TokioRuntime::<InferInput, InferOutput>::new(bundle).await;

        Self {
            context,
            reload,
            info,
            state,
            model,
            runtime,
            tokenizer: Arc::new(tokenizer),
            slots: Mutex::new(slots),
            caches: Mutex::new(caches),
        }
    }

    #[inline]
    pub fn context(&self) -> &Context {
        &self.context
    }

    #[inline]
    pub fn reload(&self) -> &ReloadRequest {
        &self.reload
    }

    #[inline]
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    #[inline]
    pub fn num_batch(&self) -> usize {
        self.state.num_batch()
    }

    #[inline]
    pub fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    pub async fn states(&self) -> Vec<(StateId, InitState)> {
        let caches = self.caches.lock().await;
        let mut states = vec![];

        if let Some(state) = &caches.default.state {
            states.push((state.id, state.clone()));
        }
        for item in caches.backed.values() {
            if let Some(state) = &item.state {
                states.push((state.id, state.clone()));
            }
        }

        states
    }

    pub async fn load_init_state(&self, state: InitState) {
        let mut caches = self.caches.lock().await;
        caches.backed.insert(
            state.id,
            Cache {
                state: Some(state),
                cache: Trie::new(),
            },
        );
    }

    pub async fn unload_init_state(&self, id: StateId) {
        let mut caches = self.caches.lock().await;
        caches.backed.remove(&id);
    }

    pub async fn serialize_model(&self, path: PathBuf) -> Result<()> {
        let model = self.model.clone();
        let handle = tokio::task::spawn_blocking(move || {
            let file = std::fs::File::create(path)?;
            model.serialize(file)
        });
        handle.await?
    }

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

    /// Compile and cache the given schema into a BNF sampler.
    async fn compile_bnf_schema(&self, schema: String) -> Result<BnfSampler> {
        BnfSampler::new(&self.tokenizer, &schema)
    }

    /// Queue an inference task.
    pub async fn queue(&self, context: GenerateContext) -> Result<SlotResult> {
        let mut slots = self.slots.lock().await;

        let mut tokens = [context.prefix, context.suffix].concat();
        if tokens.is_empty() {
            tokens.push(0);
        }

        // compile the BNF schema.
        let mut formatters = Vec::<Arc<RwLock<dyn Formatter + Send + Sync>>>::new();
        if let Some(schema) = context.request.bnf_schema.clone() {
            match self.compile_bnf_schema(schema).await {
                Ok(bnf) => formatters.push(Arc::new(RwLock::new(bnf))),
                Err(err) => return Ok(SlotResult::Error(err.to_string())),
            }
        }

        // find the best idle slot by:
        // 1. find the slot that matches the context (continue)
        // 2. find an empty slot
        // 3. find the oldest non-empty slot
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
                log::info!("start at non-empty slot {}", batch);
                let checkout = self.checkout(context.request.state, &tokens).await;
                self.state.load(checkout.state, batch)?;

                let len = checkout.prefix.len();
                assert!(len == 0 || (len > 0 && checkout.output.is_some()));
                log::info!("slot {} checks out cache of length {}", batch, len);

                let mut state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        output: checkout.output,
                        formatters,
                        ..context
                    }
                    .into(),
                );

                std::mem::swap(&mut state, &mut slots[batch]);
                Ok(SlotResult::Fault(batch))
            }
            // directly occupy an empty slot so no need backing
            Some(SlotChoice::Empty(batch)) => {
                log::info!("start at empty slot {}", batch);
                let checkout = self.checkout(context.request.state, &tokens).await;
                self.state.load(checkout.state, batch)?;

                let len = checkout.prefix.len();
                assert!(len == 0 || (len > 0 && checkout.output.is_some()));
                log::info!("slot {} checks out cache of length {}", batch, len);

                let state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        output: checkout.output,
                        formatters,
                        ..context
                    }
                    .into(),
                );
                slots[batch] = state;
                Ok(SlotResult::Success(batch))
            }
            // continue from an existing slot
            Some(SlotChoice::Continue(batch, ..)) => {
                log::info!("continue at slot {}", batch);
                let checkout = self.checkout(context.request.state, &tokens).await;
                self.state.load(checkout.state, batch)?;

                let len = checkout.prefix.len();
                assert!(len == 0 || (len > 0 && checkout.output.is_some()));
                log::info!("slot {} checks out cache of length {}", batch, len);

                let state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        output: checkout.output,
                        formatters,
                        ..context
                    }
                    .into(),
                );
                slots[batch] = state;
                Ok(SlotResult::Success(batch))
            }
        }
    }

    /// This critical section synchronizes `slots` and fills `payloads`.
    async fn synchronize(&self, payloads: &mut [Payload]) -> Result<()> {
        let mut slots = self.slots.lock().await;

        // synchronize payloads and slots: kill dead payloads
        for (slot, payload) in slots.iter().zip(payloads.iter_mut()) {
            if !(payload.is_empty() || matches!(slot, SlotState::Busy)) {
                log::warn!("payload should either be empty or slot should be busy");
                *payload = Payload::Empty;
            }
        }

        // reset all finished slots to idle
        for (batch, payload) in payloads.iter_mut().enumerate() {
            let Some(context) = payload.take() else {
                continue;
            };

            let backed = self.state.back(batch).await?;
            if let GenerateKind::Embed { layer } = context.request.kind {
                let layer = layer.clamp(0, self.info.num_layer - 1);
                let backed = backed.clone();
                let embed = self.state.embed(layer, backed)?.to_vec();
                let _ = context.sender.send(Token::Embed(embed));
            }

            if let Some(output) = context.output {
                let mut caches = self.caches.lock().await;
                let cache = &mut caches.fetch(context.request.state).cache;
                let item = CachedItem::new(backed, output);
                let (item, _) = tokio::sync::watch::channel(Some(item));
                cache.insert(context.prefix.clone(), item);
                log::info!(
                    "backed completed slot {} of length {}",
                    batch,
                    context.prefix.len()
                );
            }

            assert!(matches!(slots[batch], SlotState::Busy));
            slots[batch] = SlotState::Idle(context.prefix, Instant::now());
        }

        // take data from some pending slots
        let occupancy = payloads
            .iter()
            .filter(|x| matches!(x, Payload::Busy(_)))
            .count();
        let remain = self.reload.max_batch - self.reload.max_batch.min(occupancy);
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

            let SlotState::Wait(context) = slot else {
                unreachable!()
            };
            let mut context = *context;

            // allocate a future cache slot
            let mut caches = self.caches.lock().await;
            let cache = &mut caches.fetch(context.request.state).cache;

            let enable = context.prompt_tokens.len() > MIN_PROMPT_CACHE_TOKENS;
            let enable = enable && !cache.contains_key(context.prompt_tokens.as_token_slice());
            if enable {
                let (sender, _) = tokio::sync::watch::channel(None);
                context.prompt_cached = CachedPrompt::Future(sender.clone());
                cache.insert(Tokens(context.prompt_tokens.clone()), sender);

                log::info!(
                    "slot {} schedules future back of length {}",
                    batch,
                    context.prompt_tokens.len()
                );
            }

            let _ = context.sender.send(Token::Start);
            assert!(matches!(payloads[batch], Payload::Empty));
            payloads[batch] = Payload::Busy(context);
        }

        Ok(())
    }

    async fn sample(&self, payloads: &mut [Payload]) -> Result<HashMap<usize, (u16, Vec<f32>)>> {
        // update raw outputs
        let mut set = tokio::task::JoinSet::new();
        for (batch, payload) in payloads.iter().enumerate() {
            let Payload::Busy(context) = payload else {
                continue;
            };

            // in case that we have not yet read the whole prompt but still gets the output (from the cache)
            if !context.suffix.is_empty() {
                continue;
            }

            let Some(output) = context.output.clone() else {
                continue;
            };

            let num_vocab = self.info.num_vocab;
            let formatters = context.formatters.clone();
            let sampler = context.request.sampler.clone();
            let bias = context.request.bias.clone();
            set.spawn(async move {
                let mut data = output.to_vec();
                assert_eq!(data.len(), num_vocab);

                sampler.read().await.transform(&mut data);
                for (token, bias) in bias.iter() {
                    data[*token as usize] += *bias;
                }
                for formatter in formatters {
                    formatter.read().await.transform(&mut data);
                }

                (batch, data)
            });
        }
        let mut outputs = HashMap::new();
        while let Some(Ok((batch, data))) = set.join_next().await {
            outputs.insert(batch, data);
        }
        let outputs = (0..payloads.len())
            .map(|batch| outputs.remove(&batch))
            .map(|data| match data {
                Some(data) => TensorCpu::from_data([self.info.num_vocab, 1, 1, 1], data),
                None => TensorCpu::from_data([self.info.num_vocab, 0, 1, 1], vec![]),
            })
            .try_collect()?;

        // compute probabilities
        let outputs = softmax(&self.context, outputs).await?;

        // sample tokens
        let mut set = tokio::task::JoinSet::new();
        for (batch, (payload, output)) in payloads.iter_mut().zip(outputs.into_iter()).enumerate() {
            let Payload::Busy(context) = payload else {
                continue;
            };

            if output.is_empty() {
                continue;
            }

            let num_vocab = self.info.num_vocab;
            let sampler = context.request.sampler.clone();
            set.spawn(async move {
                let data = output.to_vec();
                assert_eq!(data.len(), num_vocab);
                let token = sampler.write().await.sample(&data);
                (batch, token, data)
            });
        }
        let mut tokens = HashMap::new();
        while let Some(Ok((batch, token, data))) = set.join_next().await {
            tokens.insert(batch, (token, data));
        }

        Ok(tokens)
    }

    async fn finalize(
        &self,
        payloads: &mut [Payload],
        tokens: HashMap<usize, (u16, Vec<f32>)>,
    ) -> Result<()> {
        for (batch, payload) in payloads.iter_mut().enumerate() {
            let Payload::Busy(context) = payload else {
                continue;
            };

            // in case that we have not yet read the whole prompt but still gets the output (from the cache)
            if !context.suffix.is_empty() {
                continue;
            }

            let Some((token, data)) = tokens.get(&batch) else {
                continue;
            };

            // cache the prompt if it is too long.
            if let (CachedPrompt::Future(sender), Some(output)) =
                (context.prompt_cached.clone(), context.output.clone())
            {
                assert_eq!(context.prefix.len(), context.prompt_tokens.len());
                let backed = self.state.back(batch).await?;
                sender.send_replace(Some(CachedItem::new(backed, output)));
                context.prompt_cached = CachedPrompt::Done;

                log::info!(
                    "backed prompt of slot {} of length {}",
                    batch,
                    context.prefix.len()
                );
            }

            // map token 0 output to "\n\n"
            let token = match token {
                0 => END_OF_LINE_TOKEN,
                _ => *token,
            };

            assert_eq!(context.suffix.len(), 0);
            context.suffix.0.push(token);

            let mut word = self.tokenizer.decode(&[token])?;
            context.model_text.append(&mut word.clone());
            context.buffer.append(&mut word);
            context.model_tokens.push(token);

            let instant = context.instant.get_or_insert(Instant::now());
            let mut done = false;
            let mut stop = |reason| {
                let counter = {
                    let prompt = context.prompt_tokens.len();
                    let completion = context.model_tokens.len();
                    let total = prompt + completion;
                    let duration = instant.elapsed();
                    TokenCounter {
                        prompt,
                        completion,
                        total,
                        duration,
                    }
                };

                let _ = context.sender.send(Token::Stop(reason, counter));
                let _ = context.sender.send(Token::Done);
                done = true;
            };

            // update the formatter (BNF) state
            let mut halt = false;
            for formatter in context.formatters.iter() {
                let mut formatter = formatter.write().await;
                halt |= formatter.update(token);
            }

            // here we detect if there is a stop word in our buffer
            let ((head, tail), stop_matched) = context
                .request
                .stop
                .iter()
                .map(|stop| {
                    let stop = stop.as_bytes();
                    let mut index_safe = 0;
                    let mut index_unsafe = 0;
                    while index_unsafe < context.buffer.len() {
                        // the maximum match of the current stop string
                        let index_stop = index_unsafe - index_safe;
                        if index_stop >= stop.len() {
                            // we have a total match
                            return (index_safe, true);
                        }

                        let output = context.buffer[index_unsafe];
                        let stop = stop[index_stop];

                        index_unsafe += 1;
                        if output != stop {
                            index_safe = index_unsafe;
                        }
                    }
                    (index_safe, index_unsafe - index_safe >= stop.len())
                })
                .min_by(|x, y| match (x.1, y.1) {
                    (true, false) => Ordering::Less,
                    (false, true) => Ordering::Greater,
                    _ => x.0.cmp(&y.0),
                })
                .map(|(mid, matched)| (context.buffer.split_at(mid), matched))
                .unwrap_or(((&context.buffer[..], &[]), false));

            if context.sender.is_disconnected() {
                done = true;
            } else if matches!(context.request.kind, GenerateKind::Choose { .. }) {
                // calculate perplexities for choose request
                let backed = self.state.read(batch)?;
                let mut perplexities = Vec::with_capacity(context.choices.len());
                for choice in &context.choices {
                    if choice.is_empty() {
                        perplexities.push(f32::INFINITY);
                        continue;
                    }

                    let mut probabilities = Vec::with_capacity(choice.len());
                    probabilities.push(data[choice[0] as usize]);

                    // construct an inference session with only one batch
                    let mut batches = vec![InferInputBatch::default(); self.num_batch()];
                    batches[batch] = InferInputBatch {
                        tokens: choice.0.clone(),
                        option: InferOption::Full,
                    };
                    let inference = InferInput::new(batches, self.reload.token_chunk_size);
                    let mut inference = Some(inference);

                    let mut index = 1;
                    loop {
                        let input = inference.take().unwrap();
                        if input.batches[batch].tokens.is_empty() {
                            break;
                        }
                        let (input, InferOutput(output)) = self.runtime.infer(input).await?;
                        inference.replace(input);

                        let output = output[batch].0.clone().split(1)?;
                        for data in output {
                            if index < choice.len() {
                                let data = data.map(|x| x.exp()).to_vec();
                                let sum: f32 = data.iter().sum();
                                let token = choice[index] as usize;
                                probabilities.push(data[token] / sum);
                            }
                            index += 1;
                        }
                    }

                    let perplexity: f32 = probabilities.into_iter().map(|x| x.ln()).sum::<f32>();
                    let perplexity = -perplexity / choice.len() as f32;
                    perplexities.push(perplexity);

                    // recover the state
                    self.state.write(backed.clone(), batch)?;
                }
                let _ = context.sender.send(Token::Choose(perplexities));
                done = true;
            } else if halt || stop_matched {
                let output = String::from_utf8_lossy(head);
                let _ = context.sender.send(Token::Content(output.into()));
                stop(FinishReason::Stop);
            } else if context.model_tokens.len() >= context.request.max_tokens {
                stop(FinishReason::Length);
            } else if let Ok(word) = String::from_utf8(head.to_vec()) {
                let _ = context.sender.send(Token::Content(word));
                context.buffer = tail.to_vec();
            }

            done.then(|| payload.finalize());
        }

        Ok(())
    }

    async fn process(&self, payloads: &mut [Payload]) -> Result<()> {
        let tokens = self.sample(payloads).await?;
        self.finalize(payloads, tokens).await?;
        self.synchronize(payloads).await?;

        let option = InferOption::Last;
        let batches = payloads
            .iter()
            .map(|payload| match payload {
                Payload::Busy(context) => context.suffix.0.clone(),
                _ => vec![],
            })
            .map(|tokens| InferInputBatch { tokens, option })
            .collect();
        let inference = InferInput::new(batches, self.reload.token_chunk_size);
        if inference.num_token() == 0 {
            return Ok(());
        }
        let mut inference = Some(inference);

        // run the model until there is at least one slot finished
        let outputs = loop {
            let input = inference.take().unwrap();
            let (input, output) = self.runtime.infer(input).await?;
            inference.replace(input);

            if output.iter().any(|batch| batch.len() > 0) {
                break output;
            }
        };

        for (payload, output) in payloads.iter_mut().zip(outputs.iter()) {
            let Payload::Busy(context) = payload else {
                continue;
            };

            // if the suffix is empty, the output is read from the cache, and we don't want to override it.
            if context.suffix.is_empty() {
                continue;
            }

            context.output = match output.len() {
                0 => None,
                x if x == self.info.num_vocab => Some(output.0.clone()),
                x => unreachable!("output size should not be {x}"),
            };
        }

        let inference = inference.unwrap();
        for (payload, input) in payloads.iter_mut().zip(inference.batches.into_iter()) {
            let Payload::Busy(context) = payload else {
                continue;
            };

            let prefix = std::mem::take(&mut context.prefix);
            let suffix = std::mem::take(&mut context.suffix);
            let model_tokens = [prefix.0, suffix.0].concat();

            // compute new prefix and suffix using the current remaining tokens
            assert!(model_tokens.len() >= input.tokens.len());
            let len = model_tokens.len() - input.tokens.len();
            context.prefix = Tokens(model_tokens[..len].to_vec());
            context.suffix = Tokens(model_tokens[len..].to_vec());
        }

        Ok(())
    }

    /// Keep the items in the cache less then [`MAX_CACHE_ITEMS`].
    async fn maintain_cache(&self) {
        let mut caches = self.caches.lock().await;
        caches.default.maintain();
        caches.backed.iter_mut().for_each(|(_, x)| x.maintain());
    }
}

pub async fn run(receiver: Receiver<()>, env: Arc<RwLock<Environment>>) {
    {
        // this task constantly runs, cleaning up state cache
        let env = env.clone();
        tokio::spawn(async move {
            loop {
                if let Environment::Loaded(runtime) = &*env.read().await {
                    runtime.maintain_cache().await;
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
    }

    while let Ok(()) = receiver.recv_async().await {
        if let Environment::Loaded(runtime) = &*env.read().await {
            let mut payloads = vec![Payload::default(); runtime.num_batch()];
            'run: loop {
                if let Err(err) = runtime.process(&mut payloads).await {
                    log::error!("{}", err);
                    break 'run;
                }
                if payloads.iter().all(Payload::is_empty) {
                    break 'run;
                }
            }
        }
    }
}
