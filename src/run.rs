use std::{
    borrow::Borrow,
    cmp::Ordering,
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use bnf_sampler::{grammar::Grammar, sampler::AcceptTokenResult, vocabulary::Vocabulary};
use flume::{Receiver, Sender};
use half::f16;
use itertools::Itertools;
use qp_trie::Trie;
use serde::Serialize;
use tokio::sync::{Mutex, RwLock};
use web_rwkv::{
    context::Context,
    runtime::{
        infer::{InferChunk, InferInfo, InferInput, InferInputBatch, InferOption, InferOutput},
        model::{ModelInfo, ModelRuntime, State},
        softmax::softmax,
        Job, JobBuilder, JobRuntime, Submission,
    },
    tensor::{TensorCpu, TensorInit},
    tokenizer::Tokenizer,
};

use crate::{
    middleware::{Environment, FinishReason, GenerateRequest, ReloadRequest, Token, TokenCounter},
    sampler::{bnf::BnfSampler, Transformer},
};

const _PENALTY_FREE_LIST: [&str; 5] = ["\n", ",", ".", "\u{002c}", "\u{002f}"];
const PROMPT_CACHE_TOKENS: usize = 32;
const MAX_CACHE_ITEMS: usize = 256;
const SAMPLER_ARENA_CAPACITY: usize = 1048576;
const GRAMMAR_ARENA_CAPACITY: usize = 1024;

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
    /// Whether the prompt has already been processed and cached.
    pub prompt_cached: bool,
    /// Tokens that have been computed and cached.
    pub prefix: Tokens,
    /// Tokens to be computed.
    pub suffix: Tokens,
    /// Texts that are output by the model.
    pub model_text: Vec<u8>,
    /// Model may output partial utf-8. This makes sure the output is always valid.
    pub buffer: Vec<u8>,
    /// Tokens that are output by the model.
    pub model_tokens: Vec<u16>,
    /// Compiled BNF schema, if any.
    pub bnf_sampler: Option<Arc<RwLock<BnfSampler>>>,
    /// Generate request provided by the caller.
    pub request: GenerateRequest,
    /// To send back generated tokens.
    pub sender: Sender<Token>,
}

#[derive(Debug)]
struct CachedItem<T> {
    item: Arc<T>,
    instant: Instant,
}

impl<T> CachedItem<T> {
    pub fn new(backed: T) -> Self {
        Self {
            item: Arc::new(backed),
            instant: Instant::now(),
        }
    }

    pub fn renew(cached: CachedItem<T>) -> Self {
        Self {
            item: cached.item,
            instant: Instant::now(),
        }
    }
}

impl<T> Clone for CachedItem<T> {
    fn clone(&self) -> Self {
        Self {
            item: self.item.clone(),
            instant: self.instant,
        }
    }
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

pub struct Runtime {
    context: Context,
    reload: ReloadRequest,
    info: ModelInfo,
    state: Arc<dyn State + Send + Sync>,
    model: Arc<dyn ModelSerialize + Send + Sync>,
    runtime: JobRuntime<InferInput, InferOutput<f16>>,
    tokenizer: Arc<Tokenizer>,
    vocab: Arc<Vocabulary>,
    slots: Mutex<Vec<SlotState>>,
    backed: Mutex<Trie<Tokens, CachedItem<TensorCpu<f32>>>>,
}

impl Runtime {
    pub async fn new<J, B>(
        context: Context,
        builder: B,
        reload: ReloadRequest,
        tokenizer: Tokenizer,
        vocab: Vocabulary,
    ) -> Self
    where
        J: Job<Info = InferInfo, Input = InferChunk, Output = InferOutput<f16>>,
        B: JobBuilder<J, Info = InferInfo> + ModelRuntime,
    {
        let slots = (0..reload.max_batch)
            .map(|_| SlotState::default())
            .collect();

        let info = builder.info();
        let state = Arc::new(builder.state());
        let model = Arc::new(Model(builder.model()));
        let runtime = JobRuntime::new(builder).await;

        Self {
            context,
            reload,
            info,
            state,
            model,
            runtime,
            tokenizer: Arc::new(tokenizer),
            vocab: Arc::new(vocab),
            slots: Mutex::new(slots),
            backed: Mutex::new(Trie::new()),
        }
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
    async fn checkout(&self, tokens: &[u16], batch: usize) -> (Vec<u16>, Arc<TensorCpu<f32>>) {
        let mut cache = self.backed.lock().await;
        let prefix = cache.longest_common_prefix(tokens.as_token_slice());
        let len = (1..=prefix.len())
            .rev()
            .find(|len| cache.contains_key(prefix[0..*len].as_token_slice()))
            .unwrap_or_default();
        log::info!("slot {} checks out backed cache of length {}", batch, len);

        let prefix = prefix[0..len].to_vec();
        let reload = match cache.remove(prefix[..].as_token_slice()) {
            Some(reload) => CachedItem::renew(reload),
            None => CachedItem::new(self.state.init()),
        };
        if len > 0 {
            let key = Tokens(prefix.clone());
            cache.insert(key, reload.clone());
        }
        (prefix, reload.item)
    }

    /// Compile and cache the given schema into a BNF sampler.
    async fn compile_bnf_schema(&self, schema: String) -> Result<BnfSampler> {
        let grammar = Grammar::new(&schema, self.vocab.clone(), GRAMMAR_ARENA_CAPACITY)?;
        let start_nonterminal = self.reload.bnf.start_nonterminal.clone();
        let sampler = bnf_sampler::sampler::Sampler::new(
            grammar,
            start_nonterminal,
            self.vocab.clone(),
            SAMPLER_ARENA_CAPACITY,
            self.reload.bnf.enable_bytes_cache,
        )?;
        Ok(BnfSampler::new(sampler))
    }

    /// Queue an inference task.
    pub async fn queue(&self, context: GenerateContext) -> Result<SlotResult> {
        let mut slots = self.slots.lock().await;

        // we must ensure that there is at least one token as the suffix, otherwise the whole slot will loop forever as there is no input
        let (last, tokens) = match [context.prefix, context.suffix].concat().split_last() {
            Some((last, tokens)) => (*last, tokens.to_vec()),
            None => return Ok(SlotResult::Error("empty task is not queued".into())),
        };

        // compile the BNF schema.
        let bnf_sampler = if let Some(schema) = context.request.bnf_schema.clone() {
            match self.compile_bnf_schema(schema).await {
                Ok(bnf_sampler) => Some(Arc::new(RwLock::new(bnf_sampler))),
                Err(err) => return Ok(SlotResult::Error(err.to_string())),
            }
        } else {
            None
        };

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
                    suffix: Tokens([tokens, vec![last]].concat()),
                    bnf_sampler,
                    ..context
                }
                .into(),
            )),
            // back a non-relative and non-empty slot and use it for our new context
            Some(SlotChoice::Back(batch)) => {
                log::info!("start at non-empty slot {}", batch);
                let (prefix, reload) = self.checkout(&tokens, batch).await;
                self.state.load(batch, reload.as_ref().clone())?;

                let tokens = [tokens, vec![last]].concat();
                let len = prefix.len();
                let mut state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        bnf_sampler,
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
                let (prefix, reload) = self.checkout(&tokens, batch).await;
                self.state.load(batch, reload.as_ref().clone())?;

                let tokens = [tokens, vec![last]].concat();
                let len = prefix.len();
                let state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        bnf_sampler,
                        ..context
                    }
                    .into(),
                );
                slots[batch] = state;
                Ok(SlotResult::Fault(batch))
            }
            // continue from an existing slot; no need backing as well
            Some(SlotChoice::Continue(batch, len)) => {
                log::info!("continue at slot {}", batch);
                let tokens = [tokens, vec![last]].concat();
                let state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        bnf_sampler,
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
    async fn prepare(&self, payloads: &mut [Payload]) -> Result<()> {
        let mut slots = self.slots.lock().await;

        // sync payloads and slots: kill dead payloads
        for (slot, payload) in slots.iter().zip_eq(payloads.iter_mut()) {
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
            if context.request.embed {
                let layer = context
                    .request
                    .embed_layer
                    .clamp(0, self.info.num_layer - 1);
                let backed = backed.clone();
                let embed = self.state.embed(layer, backed)?.to_vec();
                let _ = context.sender.send(Token::Embed(embed));
            }

            let mut cache = self.backed.lock().await;
            cache.insert(context.prefix.clone(), CachedItem::new(backed));
            log::info!(
                "backed completed slot {} of length {}",
                batch,
                context.prefix.len()
            );

            assert!(matches!(slots[batch], SlotState::Busy));
            slots[batch] = SlotState::Idle(context.prefix, Instant::now());
        }

        // take data from some waiting slots
        let occupancy = payloads
            .iter()
            .filter(|x| matches!(x, Payload::Busy(_)))
            .count();
        let remain = self.reload.max_runtime_batch - self.reload.max_runtime_batch.min(occupancy);
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
                    payloads[batch] = Payload::Busy(*context);
                }
                _ => unreachable!(),
            };
        }

        Ok(())
    }

    async fn process(&self, payloads: &mut [Payload]) -> Result<()> {
        self.prepare(payloads).await?;

        let batches = payloads
            .iter()
            .map(|payload| match payload {
                Payload::Busy(context) => context.suffix.0.clone(),
                _ => vec![],
            })
            .map(|tokens| InferInputBatch {
                tokens,
                option: InferOption::Last,
            })
            .collect();
        let inference = InferInput::new(batches, self.reload.token_chunk_size);
        if inference.num_token() == 0 {
            return Ok(());
        }
        let mut inference = Some(inference);

        // run the model until there is at least one slot finished
        let outputs = loop {
            let (sender, receiver) = tokio::sync::oneshot::channel();
            let input = inference.take().unwrap();
            let submission = Submission { input, sender };

            let _ = self.runtime.send(submission).await;
            let (input, output) = receiver.await?;
            inference = Some(input);

            if output.iter().any(|batch| batch.size() > 0) {
                break output;
            }
        };

        // update raw outputs
        let mut set = tokio::task::JoinSet::new();
        for (batch, (payload, output)) in payloads.iter().zip_eq(outputs.iter()).enumerate() {
            match (payload, output) {
                (Payload::Busy(context), output) if output.size() > 0 => {
                    let num_vocab = self.info.num_vocab;
                    let output = output.0.clone();
                    let bnf = context.bnf_sampler.clone();
                    let sampler = context.request.sampler.clone();
                    let bias = context.request.bias.clone();
                    set.spawn(async move {
                        let mut data = output.map(|x| x.to_f32()).to_vec();
                        assert_eq!(data.len(), num_vocab);

                        sampler.read().await.transform(&mut data);
                        for (token, bias) in bias.iter() {
                            data[*token as usize] += *bias;
                        }
                        if let Some(bnf) = bnf {
                            bnf.read().await.transform(&mut data);
                        }

                        let data = data.into_iter().map(f16::from_f32).collect_vec();
                        (batch, data)
                    });
                }
                _ => {}
            }
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
        for (batch, (payload, output)) in
            payloads.iter_mut().zip_eq(outputs.into_iter()).enumerate()
        {
            match (payload, output) {
                (Payload::Busy(context), output) if output.size() > 0 => {
                    let num_vocab = self.info.num_vocab;
                    let sampler = context.request.sampler.clone();
                    set.spawn(async move {
                        let data = output.map(|x| x.to_f32()).to_vec();
                        assert_eq!(data.len(), num_vocab);
                        let token = sampler.write().await.sample(&data);
                        (batch, token)
                    });
                }
                _ => {}
            }
        }
        let mut tokens = HashMap::new();
        while let Some(Ok((batch, token))) = set.join_next().await {
            tokens.insert(batch, token);
        }

        let inference = inference.unwrap();
        for (batch, (payload, input)) in
            itertools::multizip((payloads.iter_mut(), inference.batches.into_iter())).enumerate()
        {
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

            let Some(&token) = tokens.get(&batch) else {
                continue;
            };

            // cache the prompt if it is too long.
            if !context.prompt_cached && context.prompt_tokens.len() > PROMPT_CACHE_TOKENS {
                let mut cache = self.backed.lock().await;
                let backed = self.state.back(batch).await?;

                cache.insert(context.prefix.clone(), CachedItem::new(backed));
                context.prompt_cached = true;

                log::info!(
                    "backed prompt of slot {} of length {}",
                    batch,
                    context.prefix.len()
                );
            }

            assert_eq!(context.suffix.len(), 0);
            context.suffix.0.push(token);

            let mut word = self.tokenizer.decode(&[token])?;
            context.model_text.append(&mut word.clone());
            context.buffer.append(&mut word);
            context.model_tokens.push(token);

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

            let mut done = false;
            let mut finish = |reason| {
                let _ = context.sender.send(Token::Stop(reason, count_tokens()));
                let _ = context.sender.send(Token::Done);
                done = true;
            };

            // update the BNF state
            let mut exhausted = false;
            if let Some(bnf) = context.bnf_sampler.clone() {
                let mut bnf = bnf.write().await;
                match bnf.update(token) {
                    AcceptTokenResult::Continue => {}
                    AcceptTokenResult::End => exhausted = true,
                    AcceptTokenResult::Failed => {
                        log::warn!("slot {batch} bnf failure");
                        exhausted = true;
                    }
                }
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
            } else if exhausted || stop_matched {
                let output = String::from_utf8_lossy(head);
                let _ = context.sender.send(Token::Content(output.into()));
                finish(FinishReason::Stop);
            } else if context.model_tokens.len() >= context.request.max_tokens {
                finish(FinishReason::Length);
            } else if let Ok(word) = String::from_utf8(head.to_vec()) {
                let _ = context.sender.send(Token::Content(word));
                context.buffer = tail.to_vec();
            }

            done.then(|| payload.finalize());
        }

        Ok(())
    }

    /// Keep the items in the cache less then [`MAX_CACHE_ITEMS`].
    async fn maintain_cache(&self) {
        let mut cache = self.backed.lock().await;
        if cache.count() <= MAX_CACHE_ITEMS {
            return;
        }

        let mut removing = vec![];
        for (tokens, _) in cache
            .iter()
            .sorted_unstable_by_key(|(_, item)| item.instant.elapsed())
            .skip(MAX_CACHE_ITEMS)
        {
            removing.push(tokens.to_owned());
        }

        for tokens in removing.into_iter() {
            cache.remove(&tokens);
        }
    }
}

#[tokio::main]
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
