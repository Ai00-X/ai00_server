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
        infer::{InferInput, InferInputBatch, InferOption, InferOutputBatch},
        model::{ModelInfo, State},
        Runtime,
    },
    tensor::{kind::ReadWrite, TensorCpu, TensorError, TensorGpu, TensorInto},
    tokenizer::Tokenizer,
};

use crate::{
    sampler::{bnf::BnfSampler, Formatter, Sampler},
    FinishReason, GenerateKind, GenerateRequest, InitState, ReloadRequest, RuntimeInfo, StateId,
    Token, TokenCounter,
};

const MIN_PROMPT_CACHE_TOKENS: usize = 32;
const MAX_CACHE_ITEMS: usize = 256;

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
    Busy(JoinHandle<Result<GenerateContext>>),
    /// The slot is locked for updating.
    Locked,
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

#[derive(Debug, Clone)]
enum InferBatch {
    Run {
        batch: usize,
        tokens: Vec<u16>,
        option: InferOption,
        sender: Sender<TensorCpu<f32>>,
    },
    Load {
        batch: usize,
        tensor: TensorCpu<f32>,
    },
    Back {
        batch: usize,
        sender: Sender<TensorCpu<f32>>,
    },
    Write {
        batch: usize,
        tensor: TensorGpu<f32, ReadWrite>,
    },
    Read {
        batch: usize,
        sender: Sender<TensorGpu<f32, ReadWrite>>,
    },
}

#[derive(Debug, Clone)]
struct SoftmaxBatch {
    input: TensorCpu<f32>,
    sender: Sender<TensorCpu<f32>>,
}

#[derive(Debug, Clone)]
struct RuntimeSender {
    infer: Sender<InferBatch>,
    softmax: Sender<SoftmaxBatch>,
}

#[derive(Derivative, Clone)]
#[derivative(Debug)]
struct CoreRuntime {
    context: Context,
    info: ModelInfo,
    reload: Arc<ReloadRequest>,
    #[derivative(Debug = "ignore")]
    state: Arc<dyn State + Send + Sync>,
    sender: RuntimeSender,
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
    async fn queue(&self, context: GenerateContext) -> Result<SlotResult, TensorError> {
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
        let choice = {
            let mut slots = self.slots.lock().await;
            let choice = slots
                .iter()
                .enumerate()
                .filter_map(|(batch, slot)| match slot {
                    SlotState::Idle(content, instant) => {
                        let delta = instant.elapsed();
                        match (content.is_empty(), tokens.starts_with(content)) {
                            (true, _) => Some((SlotChoice::Empty(batch), delta)),
                            (_, true) => Some((SlotChoice::Continue(batch, content.len()), delta)),
                            (_, false) => Some((SlotChoice::Back(batch), delta)),
                        }
                    }
                    _ => None,
                })
                .max_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)))
                .map(|(x, _)| x);
            match choice {
                None => (),
                Some(SlotChoice::Empty(batch))
                | Some(SlotChoice::Back(batch))
                | Some(SlotChoice::Continue(batch, _)) => slots[batch] = SlotState::Locked,
            }
            choice
        };

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
                log::info!("[queue][back][slot: {batch}]");
                let checkout = self.checkout(context.request.state, &tokens).await;
                self.load(batch, checkout.state).await;

                let len = checkout.prefix.len();
                assert!(len == 0 || (len > 0 && checkout.output.is_some()));
                log::info!("[cache][checkout[[slot: {batch}][len: {len}]");

                let context = GenerateContext {
                    prefix: Tokens(tokens[..len].to_vec()),
                    suffix: Tokens(tokens[len..].to_vec()),
                    output: checkout.output,
                    formatters,
                    ..context
                };
                let handle = tokio::spawn(self.clone().process(batch, context));
                let mut slots = self.slots.lock().await;
                slots[batch] = SlotState::Busy(handle);
                Ok(SlotResult::Fault(batch))
            }
            // directly occupy an empty slot so no need backing
            Some(SlotChoice::Empty(batch)) => {
                log::info!("[queue][empty][slot: {batch}]");
                let checkout = self.checkout(context.request.state, &tokens).await;
                self.load(batch, checkout.state).await;

                let len = checkout.prefix.len();
                assert!(len == 0 || (len > 0 && checkout.output.is_some()));
                log::info!("[cache][checkout][slot: {batch}][len: {len}]");

                let context = GenerateContext {
                    prefix: Tokens(tokens[..len].to_vec()),
                    suffix: Tokens(tokens[len..].to_vec()),
                    output: checkout.output,
                    formatters,
                    ..context
                };
                let handle = tokio::spawn(self.clone().process(batch, context));
                let mut slots = self.slots.lock().await;
                slots[batch] = SlotState::Busy(handle);
                Ok(SlotResult::Success(batch))
            }
            Some(SlotChoice::Continue(batch, ..)) => {
                log::info!("[queue][continue][slot: {batch}]");
                let checkout = self.checkout(context.request.state, &tokens).await;
                self.load(batch, checkout.state).await;

                let len = checkout.prefix.len();
                assert!(len == 0 || (len > 0 && checkout.output.is_some()));
                log::info!("[cache][checkout[[slot: {batch}][len: {len}]");

                let context = GenerateContext {
                    prefix: Tokens(tokens[..len].to_vec()),
                    suffix: Tokens(tokens[len..].to_vec()),
                    output: checkout.output,
                    formatters,
                    ..context
                };
                let handle = tokio::spawn(self.clone().process(batch, context));
                let mut slots = self.slots.lock().await;
                slots[batch] = SlotState::Busy(handle);
                Ok(SlotResult::Success(batch))
            }
        }
    }

    /// Reset finished slots to `idle`. Cache current states of finished slots.
    async fn update(&self) {
        let update = |handle: JoinHandle<_>| async move {
            if !handle.is_finished() {
                return Ok(SlotState::Busy(handle));
            }

            let context = handle.await??;
            Ok::<_, Box<dyn Error + Send + Sync>>(SlotState::Idle(context.prefix, Instant::now()))
        };

        for batch in 0..self.reload.max_batch {
            let handle = {
                let mut slots = self.slots.lock().await;
                let slot = std::mem::replace(&mut slots[batch], SlotState::Locked);
                let SlotState::Busy(handle) = slot else {
                    slots[batch] = slot;
                    continue;
                };
                handle
            };

            let updated = match update(handle).await {
                Ok(updated) => updated,
                Err(err) => {
                    log::error!("[update][error][slot: {batch}] {err}");
                    let mut slots = self.slots.lock().await;
                    slots[batch] = Default::default();
                    continue;
                }
            };

            let mut slots = self.slots.lock().await;
            slots[batch] = updated;
        }
    }

    async fn sample(
        &self,
        output: TensorCpu<f32>,
        sampler: Arc<RwLock<dyn Sampler + Send + Sync>>,
        formatters: Vec<Arc<RwLock<dyn Formatter + Send + Sync>>>,
        bias: Arc<HashMap<u16, f32>>,
    ) -> Result<(u16, TensorCpu<f32>)> {
        // process raw model outputs
        let num_vocab = self.info.num_vocab;
        let input = {
            let mut data = output.to_vec();
            assert_eq!(data.len(), num_vocab);

            sampler.read().await.transform(&mut data);
            for formatter in formatters {
                formatter.read().await.transform(&mut data);
            }
            for (token, bias) in bias.iter() {
                data[*token as usize] += *bias;
            }

            self.context.tensor_from_data([num_vocab, 1, 1, 1], data)?
        };

        // compute probabilities
        let (sender, receiver) = flume::bounded(1);
        let _ = self.sender.softmax.send(SoftmaxBatch { input, sender });
        let output = receiver.recv_async().await?;

        // sample tokens
        assert_eq!(output.len(), num_vocab);
        let token = sampler.write().await.sample(&output);
        Ok((token, output))
    }

    async fn perplexity(&self, batch: usize, tokens: &[u16], head: Option<f32>) -> Result<f32> {
        let mut p = Vec::with_capacity(tokens.len().max(1));
        let len = tokens.len();
        let tokens = match head {
            Some(head) => {
                p.push(head);
                tokens.to_vec()
            }
            None => [&[0], tokens].concat(),
        };

        let (sender, receiver) = flume::unbounded();
        let _ = self
            .sender
            .infer
            .send_async({
                let tokens = tokens.clone();
                let option = InferOption::Full;
                InferBatch::Run {
                    batch,
                    tokens,
                    option,
                    sender,
                }
            })
            .await;

        let index = Arc::new(Mutex::new(1));
        while p.len() < len {
            let tokens = tokens.clone();
            let output = receiver.recv_async().await?;
            let output = output.split(1)?;
            let f = {
                let index = index.clone();
                move || {
                    let mut index = index.blocking_lock();
                    let mut p = Vec::with_capacity(output.len());
                    for data in output {
                        if *index < tokens.len() {
                            let data = data.map(|x| x.exp()).to_vec();
                            let sum: f32 = data.iter().sum();
                            let token = tokens[*index] as usize;
                            p.push(data[token] / sum);
                        }
                        *index += 1;
                    }
                    p
                }
            };
            let mut q = tokio::task::spawn_blocking(f).await?;
            p.append(&mut q);
        }

        let ppl: f32 = p.into_iter().map(|x| x.ln()).sum();
        let ppl = -ppl / tokens.len() as f32;
        Ok(ppl)
    }

    async fn load(&self, batch: usize, tensor: TensorCpu<f32>) {
        let _ = self
            .sender
            .infer
            .send_async(InferBatch::Load { batch, tensor })
            .await;
    }

    async fn back(&self, batch: usize) -> Result<TensorCpu<f32>> {
        let (sender, receiver) = flume::bounded(1);
        let _ = self.sender.infer.send(InferBatch::Back { batch, sender });
        let tensor = receiver.recv_async().await?;
        Ok(tensor)
    }

    async fn write(&self, batch: usize, tensor: TensorGpu<f32, ReadWrite>) {
        let _ = self
            .sender
            .infer
            .send_async(InferBatch::Write { batch, tensor })
            .await;
    }

    async fn read(&self, batch: usize) -> Result<TensorGpu<f32, ReadWrite>> {
        let (sender, receiver) = flume::bounded(1);
        let _ = self.sender.infer.send(InferBatch::Read { batch, sender });
        let tensor = receiver.recv_async().await?;
        Ok(tensor)
    }

    /// Read in the prompt of a batch and continuously sample it until it is done.
    async fn process(self, batch: usize, mut context: GenerateContext) -> Result<GenerateContext> {
        // schedule a future cache slot for the prompt
        {
            let mut caches = self.caches.lock().await;
            let cache = &mut caches.fetch(context.request.state).cache;

            let enable = context.prompt_tokens.len() > MIN_PROMPT_CACHE_TOKENS;
            let enable = enable && !cache.contains_key(context.prompt_tokens.as_token_slice());
            if enable {
                let (sender, _) = tokio::sync::watch::channel(None);
                context.prompt_cached = CachedPrompt::Future(sender.clone());
                cache.insert(Tokens(context.prompt_tokens.clone()), sender);

                let len = context.prompt_tokens.len();
                log::info!("[cache][future][slot: {batch}][len: {len}]");
            }
        }

        let _ = context.sender.send(Token::Start);

        loop {
            let output = match (context.suffix.len(), context.output.clone()) {
                (0, Some(output)) => output,
                _ => {
                    let (sender, receiver) = flume::bounded(1);
                    let _ = self
                        .sender
                        .infer
                        .send_async(InferBatch::Run {
                            batch,
                            tokens: context.suffix.to_vec(),
                            option: InferOption::Last,
                            sender,
                        })
                        .await;

                    let prefix = std::mem::take(&mut context.prefix);
                    let suffix = std::mem::take(&mut context.suffix);

                    context.prefix = Tokens([prefix.0, suffix.0].concat());
                    context.suffix = Tokens(vec![]);

                    receiver.recv_async().await?
                }
            };

            // cache the prompt if being asked
            if let CachedPrompt::Future(sender) = context.prompt_cached.clone() {
                assert_eq!(context.prefix.len(), context.prompt_tokens.len());

                let backed = self.back(batch).await?;
                let output = output.clone();
                sender.send_replace(Some(CachedItem::new(backed, output)));
                context.prompt_cached = CachedPrompt::Done;

                let len = context.prefix.len();
                log::info!("[cache][insert][slot: {batch}][len: {len}]");
            }

            let (token, output) = {
                let output = output.clone();
                let sampler = context.request.sampler.clone();
                let formatters = context.formatters.clone();
                let bias = context.request.bias.clone();
                self.sample(output, sampler, formatters, bias).await?
            };

            let mut stop_token = token == 0;
            let mut word = match self.tokenizer.decode(&[token]) {
                Ok(word) => word,
                Err(err) => {
                    log::warn!("[process][error] {err}");
                    stop_token = true;
                    Vec::new()
                }
            };

            context.output = Some(output.clone());
            context.suffix.0.push(token);
            context.model_tokens.push(token);
            context.model_text.extend(&word);
            context.buffer.append(&mut word);

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
            } else if let GenerateKind::Choose { calibrate, .. } = context.request.kind {
                let backed = self.read(batch).await?;
                let mut ppl = vec![f32::INFINITY; context.choices.len()];

                if calibrate {
                    // compute perplexities of the choices themselves and calibrate their effects
                    let init: TensorGpu<_, _> = self.state.init().to(&self.context);
                    for (index, choice) in context
                        .choices
                        .iter()
                        .enumerate()
                        .filter(|(_, choice)| !choice.is_empty())
                    {
                        self.write(batch, init.clone()).await;
                        ppl[index] = -self.perplexity(batch, choice, None).await?;
                    }
                    // recover the state
                    self.write(batch, backed.clone()).await;
                }

                for (index, choice) in context
                    .choices
                    .iter()
                    .enumerate()
                    .filter(|(_, choice)| !choice.is_empty())
                {
                    let output = output.clone().to_vec();
                    let head = Some(output[choice[0] as usize]);
                    let p = self.perplexity(batch, choice, head).await?;
                    ppl[index] = match calibrate {
                        true => ppl[index] + p,
                        false => p,
                    };
                    // recover the state
                    self.write(batch, backed.clone()).await;
                }

                let _ = context.sender.send(Token::Choose(ppl));
                done = true;
            } else if let GenerateKind::Embed { .. } = context.request.kind {
                let backed = self.back(batch).await?;
                let embed = backed.to_vec();
                let _ = context.sender.send(Token::Embed(embed));
                done = true;
            } else if halt || stop_matched || stop_token {
                let output = String::from_utf8_lossy(head);
                let _ = context.sender.send(Token::Content(output.into()));
                stop(FinishReason::Stop);

                if let Some(output) = context.output.clone() {
                    let backed = self.back(batch).await?;
                    let mut caches = self.caches.lock().await;
                    let cache = &mut caches.fetch(context.request.state).cache;
                    let item = CachedItem::new(backed, output);
                    let (item, _) = tokio::sync::watch::channel(Some(item));
                    cache.insert(context.prefix.clone(), item);

                    let len = context.prefix.len();
                    log::info!("[cache][insert][slot: {batch}][len: {len}]");
                }
            } else if context.model_tokens.len() >= context.request.max_tokens {
                stop(FinishReason::Length);
            } else if let Ok(word) = String::from_utf8(head.to_vec()) {
                let _ = context.sender.send(Token::Content(word));
                context.buffer = tail.to_vec();
            }

            if done {
                log::info!("[process][done][slot: {batch}]");
                break;
            }
        }

        Ok(context)
    }

    /// Keep the items in the cache less then [`MAX_CACHE_ITEMS`].
    async fn maintain_cache(&self) {
        let mut caches = self.caches.lock().await;
        caches.default.maintain();
        caches.backed.iter_mut().for_each(|(_, x)| x.maintain());
    }
}

async fn enqueue(runtime: CoreRuntime, receiver: Receiver<GenerateContext>, timer: Duration) {
    let mut queue = Vec::<GenerateContext>::new();

    'outer: while let Ok(context) = receiver.recv_async().await {
        queue.push(context);

        'inner: loop {
            runtime.maintain_cache().await;
            runtime.update().await;

            let mut temp = Vec::new();
            for context in queue.drain(..) {
                match runtime.queue(context).await {
                    Ok(SlotResult::Failure(context)) => temp.push(*context),
                    Ok(SlotResult::Success(batch)) => log::info!("[enqueue][ok][slot: {batch}]"),
                    Ok(SlotResult::Fault(batch)) => log::info!("[enqueue][fault][slot: {batch}]"),
                    Ok(SlotResult::Error(err)) => log::error!("[enqueue][error] {err}"),
                    Err(err) => log::error!("[enqueue][error] {err}"),
                }
            }
            std::mem::swap(&mut queue, &mut temp);

            if queue.is_empty() {
                break 'inner;
            }

            match receiver.try_recv() {
                Ok(context) => queue.push(context),
                Err(TryRecvError::Empty) => tokio::time::sleep(timer).await,
                Err(TryRecvError::Disconnected) => break 'outer,
            }
        }
    }
}

async fn finalize(runtime: CoreRuntime, receiver: Receiver<GenerateContext>, timer: Duration) {
    while !receiver.is_disconnected() {
        runtime.maintain_cache().await;
        runtime.update().await;
        tokio::time::sleep(timer).await;
    }
}

async fn infer(
    reload: Arc<ReloadRequest>,
    runtime: Weak<dyn Runtime + Send + Sync>,
    state: Arc<dyn State + Send + Sync>,
    receiver: Receiver<InferBatch>,
) -> Result<()> {
    let mut senders = vec![None; reload.max_batch];
    let batches = vec![
        InferInputBatch {
            tokens: vec![],
            option: Default::default(),
        };
        reload.max_batch
    ];
    let inference = InferInput::new(batches, reload.token_chunk_size);
    let mut inference = Some(inference);

    async fn schedule(
        inference: &mut Option<InferInput>,
        senders: &mut [Option<Sender<TensorCpu<f32>>>],
        state: Arc<dyn State + Send + Sync>,
        batch: InferBatch,
    ) -> Result<()> {
        match batch {
            InferBatch::Run {
                batch,
                tokens,
                option,
                sender,
            } => {
                let mut input = inference.take().expect("inference must not be `None`");
                input.batches[batch] = InferInputBatch { tokens, option };
                senders[batch] = Some(sender);
                inference.replace(input);
            }
            InferBatch::Load { batch, tensor } => state.load(tensor, batch)?,
            InferBatch::Back { batch, sender } => {
                let tensor = state.back(batch).await?;
                let _ = sender.send_async(tensor).await;
            }
            InferBatch::Write { batch, tensor } => state.write(tensor, batch)?,
            InferBatch::Read { batch, sender } => {
                let tensor = state.read(batch)?;
                let _ = sender.send_async(tensor).await;
            }
        }
        Ok(())
    }

    'outer: while let Ok(batch) = receiver.recv_async().await {
        schedule(&mut inference, &mut senders, state.clone(), batch).await?;

        while inference
            .as_ref()
            .map(|input| input.num_token() > 0)
            .expect("inference must not be `None`")
        {
            'inner: loop {
                let state = state.clone();
                match receiver.try_recv() {
                    Ok(batch) => schedule(&mut inference, &mut senders, state, batch).await?,
                    Err(TryRecvError::Empty) => break 'inner,
                    Err(TryRecvError::Disconnected) => break 'outer,
                }
            }

            let Some(runtime) = runtime.upgrade() else {
                break 'outer;
            };
            let input = inference.take().expect("inference must not be `None`");
            let (input, output) = runtime.infer(input).await?;
            inference.replace(input);

            for (InferOutputBatch(output), sender) in output
                .iter()
                .zip_eq(senders.clone().into_iter())
                .filter(|(output, _)| !output.is_empty())
                .filter_map(|(output, sender)| sender.map(|sender| (output, sender)))
            {
                let _ = sender.send_async(output.clone()).await;
            }
        }
    }

    log::info!("[infer] exit");
    Ok(())
}

async fn softmax(
    reload: Arc<ReloadRequest>,
    context: Context,
    receiver: Receiver<SoftmaxBatch>,
) -> Result<()> {
    let mut batches = Vec::with_capacity(reload.max_batch);

    while let Ok(batch) = receiver.recv_async().await {
        batches.push(batch);

        for batch in receiver.drain() {
            batches.push(batch);
        }

        let input = batches.iter().map(|batch| batch.input.clone()).collect();
        let output = web_rwkv::runtime::softmax::softmax(&context, input).await?;

        for (batch, tensor) in batches.iter().zip_eq(output.into_iter()) {
            let _ = batch.sender.send_async(tensor).await;
        }

        batches.clear();
    }

    log::info!("[softmax] exit");
    Ok(())
}

pub async fn run(
    context: Context,
    runtime: Weak<dyn Runtime + Send + Sync>,
    state: Arc<dyn State + Send + Sync>,
    receiver: Receiver<GenerateContext>,
    RuntimeInfo {
        reload,
        info,
        states,
        tokenizer,
        ..
    }: RuntimeInfo,
) {
    let slots = std::iter::repeat_with(Default::default)
        .take(reload.max_batch)
        .collect();
    let slots = Arc::new(Mutex::new(slots));

    let caches = {
        let mut caches = CacheHub::default();
        // set up default initial state
        if let Some(state) = states.iter().find(|state| state.default) {
            caches.default.state = Some(state.clone());
        }
        // set up other initial states with ids
        for state in states {
            let id = state.id;
            let item = Cache {
                state: Some(state),
                cache: Trie::new(),
            };
            caches.backed.insert(id, item);
        }
        Arc::new(Mutex::new(caches))
    };

    let max_batch = reload.max_batch;
    let runtime = {
        let infer = {
            let (sender, receiver) = flume::unbounded();
            tokio::spawn(infer(reload.clone(), runtime, state.clone(), receiver));
            sender
        };
        let softmax = {
            let (sender, receiver) = flume::unbounded();
            tokio::spawn(softmax(reload.clone(), context.clone(), receiver));
            sender
        };
        let sender = RuntimeSender { infer, softmax };
        CoreRuntime {
            context,
            info,
            reload,
            state,
            sender,
            tokenizer,
            slots,
            caches,
        }
    };
    let timer = Duration::from_secs_f32(1.0);
    for _ in 0..max_batch {
        tokio::spawn(enqueue(runtime.clone(), receiver.clone(), timer));
    }
    tokio::spawn(finalize(runtime, receiver, timer));
}
