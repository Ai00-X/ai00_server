use std::{
    borrow::Borrow,
    cmp::Ordering,
    collections::HashSet,
    convert::Infallible,
    future::Future,
    io::Write,
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use bnf_sampler::{grammar::Grammar, sampler::AcceptTokenResult, vocabulary::Vocabulary};
use flume::{Receiver, Sender};
use itertools::Itertools;
use qp_trie::Trie;
use serde::Serialize;
use tokio::sync::{Mutex, RwLock};
use web_rwkv::{
    model::{
        BackedState, Build, Model, ModelInfo, ModelInput, ModelOutput, ModelState, StateBuilder,
    },
    tokenizer::Tokenizer,
};

use crate::{
    middleware::{Environment, FinishReason, GenerateRequest, ReloadRequest, Token, TokenCounter},
    sampler::{bnf::BnfSampler, Transformer},
};

const PENALTY_FREE_LIST: [&str; 5] = ["\n", ",", ".", "\u{002c}", "\u{002f}"];
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

    /// Returns `true` if the payload is [`Busy`].
    ///
    /// [`Busy`]: Payload::Busy
    #[must_use]
    pub fn is_busy(&self) -> bool {
        matches!(self, Self::Busy(..))
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

pub trait Runner {
    fn info(&self) -> &ModelInfo;
    fn num_batch(&self) -> usize;
    fn tokenizer(&self) -> Arc<Tokenizer>;
    fn reload(&self) -> &ReloadRequest;

    /// Serialize the model into the given path.
    fn serialize_model(&self, path: PathBuf) -> Result<()>;

    /// Queue an inference task.
    fn queue(
        &self,
        context: GenerateContext,
    ) -> Pin<Box<dyn Future<Output = SlotResult> + Send + '_>>;

    /// Note: only called on the process thread.
    fn process<'a>(
        &'a self,
        payloads: &'a mut [Payload],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>>;

    /// Keep the items in the cache less then [`MAX_CACHE_ITEMS`].
    fn maintain_cache(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

#[derive(Debug)]
pub struct Runtime<M, S, B>
where
    B: BackedState,
    S: ModelState<BackedState = B>,
    M: Model<State = S> + Serialize,
    StateBuilder: Build<B, Error = Infallible>,
{
    model: M,
    state: S,
    tokenizer: Arc<Tokenizer>,
    vocab: Arc<Vocabulary>,
    slots: Mutex<Vec<SlotState>>,
    backed: Mutex<Trie<Tokens, CachedItem<B>>>,
    reload: ReloadRequest,
    _penalty_free_tokens: HashSet<u16>,
}

impl<M, S, B> Runtime<M, S, B>
where
    B: BackedState,
    S: ModelState<BackedState = B>,
    M: Model<State = S> + Serialize,
    StateBuilder: Build<B, Error = Infallible>,
{
    pub fn new(
        tokenizer: Tokenizer,
        vocab: Vocabulary,
        model: M,
        state: S,
        reload: ReloadRequest,
    ) -> Self {
        let slots = (0..state.num_batch())
            .map(|_| SlotState::default())
            .collect();
        let _penalty_free_tokens = (0..u16::MAX)
            .filter(|&token| {
                let word = tokenizer.decode(&[token]).unwrap_or_default();
                let word = String::from_utf8_lossy(&word).into_owned();
                PENALTY_FREE_LIST.iter().any(|x| word.contains(x))
            })
            .collect();

        Self {
            model,
            state,
            tokenizer: Arc::new(tokenizer),
            vocab: Arc::new(vocab),
            slots: Mutex::new(slots),
            backed: Mutex::new(Trie::new()),
            reload,
            _penalty_free_tokens,
        }
    }

    #[inline]
    pub fn reload(&self) -> &ReloadRequest {
        &self.reload
    }

    fn serialize_model(&self, path: PathBuf) -> Result<()> {
        use cbor4ii::{core::enc::Write, serde::Serializer};
        use std::fs::File;

        struct FileWriter(File);
        impl Write for FileWriter {
            type Error = std::io::Error;
            fn push(&mut self, input: &[u8]) -> Result<(), Self::Error> {
                self.0.write_all(input)
            }
        }

        let file = FileWriter(File::create(path)?);
        let mut serializer = Serializer::new(file);
        self.model.serialize(&mut serializer)?;

        Ok(())
    }

    /// Search for the longest common prefix in the memory cache and checkout the state from that point.
    /// Should there be a cache miss, an initial state is returned.
    async fn checkout(&self, tokens: &[u16], batch: usize) -> (Vec<u16>, Arc<B>) {
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
            None => {
                let context = self.model.context();
                let info = self.model.info();
                let backed = StateBuilder::new(context, info)
                    .with_chunk_size(self.reload.state_chunk_size)
                    .build()
                    .unwrap();
                CachedItem::new(backed)
            }
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
    async fn queue(&self, context: GenerateContext) -> SlotResult {
        let mut slots = self.slots.lock().await;

        // we must ensure that there is at least one token as the suffix, otherwise the whole slot will loop forever as there is no input
        let (last, tokens) = match [context.prefix, context.suffix].concat().split_last() {
            Some((last, tokens)) => (*last, tokens.to_vec()),
            None => return SlotResult::Error("empty task is not queued".into()),
        };

        // compile the BNF schema.
        let bnf_sampler = if let Some(schema) = context.request.bnf_schema.clone() {
            match self.compile_bnf_schema(schema).await {
                Ok(bnf_sampler) => Some(Arc::new(RwLock::new(bnf_sampler))),
                Err(err) => return SlotResult::Error(err.to_string()),
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
            .max_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)));

        match choice {
            // we cannot find a slot because all slots are occupied
            // in this case, we hand the request back to the caller
            None => SlotResult::Failure(
                GenerateContext {
                    prefix: Default::default(),
                    suffix: Tokens([tokens, vec![last]].concat()),
                    bnf_sampler,
                    ..context
                }
                .into(),
            ),
            // back a non-relative and non-empty slot and use it for our new context
            Some((SlotChoice::Back(batch), _)) => {
                log::info!("start at non-empty slot {}", batch);
                let (prefix, reload) = self.checkout(&tokens, batch).await;

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
                match state {
                    SlotState::Idle(_, _) => {
                        // let backed = self.state.back_batch(batch).await.unwarp();
                        // cache.insert(content, backed.into());
                        self.state.load_batch(&reload, batch).unwrap();
                        SlotResult::Fault(batch)
                    }
                    _ => unreachable!(),
                }
            }
            // directly occupy an empty slot so no need backing
            Some((SlotChoice::Empty(batch), _)) => {
                log::info!("start at empty slot {}", batch);
                let (prefix, reload) = self.checkout(&tokens, batch).await;

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

                self.state.load_batch(&reload, batch).expect("load state");
                SlotResult::Fault(batch)
            }
            // continue from an existing slot. No need backing as well
            Some((SlotChoice::Continue(batch, len), _)) => {
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
                SlotResult::Success(batch)
            }
        }
    }

    /// This critical section synchronizes `slots` and fills `payloads`.
    async fn prepare(&self, payloads: &mut [Payload]) {
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

            let backed = self.state.back_batch(batch).await.unwrap();

            if context.request.embed {
                let embed_layer = context
                    .request
                    .embed_layer
                    .clamp(0, self.model.info().num_layer - 1);
                let embed = backed.embed(0, embed_layer);
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
    }

    async fn process(&self, payloads: &mut [Payload]) -> Result<()> {
        self.prepare(payloads).await;

        let mut inputs = payloads
            .iter()
            .map(|payload| match payload {
                Payload::Busy(context) => context.suffix.0.clone(),
                _ => vec![],
            })
            .map(|tokens| ModelInput {
                tokens,
                ..Default::default()
            })
            .collect_vec();

        // run the model until there is at least one slot finished
        let occupancy = payloads.iter().filter(|x| x.is_busy()).count();
        let outputs = match occupancy {
            0 => vec![ModelOutput::None; payloads.len()],
            _ => loop {
                let output = self.model.run(&mut inputs, &self.state).await?;
                if output.iter().any(ModelOutput::is_some) {
                    break output;
                }
            },
        };

        // update raw outputs
        let handles = payloads
            .iter()
            .zip_eq(outputs.into_iter())
            .map(|(payload, output)| match payload {
                Payload::Busy(context) => match output {
                    ModelOutput::None => None,
                    ModelOutput::Last(data) => Some((
                        context.bnf_sampler.clone(),
                        context.request.sampler.clone(),
                        context.request.bias.clone(),
                        data,
                    )),
                    ModelOutput::Full(_) => unreachable!(),
                },
                _ => None,
            })
            .map(|bundle| async move {
                match bundle {
                    Some((bnf, sampler, bias, mut data)) => {
                        sampler.read().await.transform(&mut data);
                        for (token, bias) in bias.iter() {
                            data[*token as usize] += *bias
                        }
                        if let Some(bnf) = bnf {
                            let bnf = bnf.read().await;
                            bnf.transform(&mut data);
                        }
                        Some(data)
                    }
                    None => None,
                }
            })
            .map(tokio::spawn)
            .collect_vec();
        let outputs = {
            let mut outputs = vec![];
            for handle in handles {
                let output = match handle.await? {
                    Some(data) => ModelOutput::Last(data),
                    None => ModelOutput::None,
                };
                outputs.push(output);
            }
            outputs
        };

        // compute probabilities
        let outputs = match occupancy {
            0 => vec![ModelOutput::None; payloads.len()],
            _ => self.model.softmax(outputs).await?,
        };

        // sample tokens
        let handles = payloads
            .iter()
            .zip_eq(outputs.into_iter())
            .map(|(payload, output)| match payload {
                Payload::Busy(context) => match output {
                    ModelOutput::None => None,
                    ModelOutput::Last(data) => Some((context.request.sampler.clone(), data)),
                    ModelOutput::Full(_) => unreachable!(),
                },
                _ => None,
            })
            .map(|bundle| async move {
                match bundle {
                    Some((sampler, data)) => Some(sampler.write().await.sample(&data)),
                    None => None,
                }
            })
            .map(tokio::spawn)
            .collect_vec();
        let outputs = {
            let mut outputs = vec![];
            for handle in handles {
                outputs.push(handle.await?);
            }
            outputs
        };

        for (batch, payload, token, input) in payloads
            .iter_mut()
            .zip_eq(outputs.into_iter().zip_eq(inputs.into_iter()))
            .enumerate()
            .map(|(i, (x, (y, z)))| (i, x, y, z))
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

            let Some(token) = token else {
                continue;
            };

            // cache the prompt if it is too long.
            if !context.prompt_cached && context.prompt_tokens.len() > PROMPT_CACHE_TOKENS {
                let mut cache = self.backed.lock().await;
                let backed = self.state.back_batch(batch).await.unwrap();

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

impl<M, S, B> Runner for Runtime<M, S, B>
where
    B: BackedState + Send + Sync,
    S: ModelState<BackedState = B> + Send + Sync,
    M: Model<State = S> + Serialize + Send + Sync,
    StateBuilder: Build<B, Error = Infallible>,
{
    #[inline]
    fn info(&self) -> &ModelInfo {
        self.model.info()
    }

    #[inline]
    fn num_batch(&self) -> usize {
        self.state.num_batch()
    }

    #[inline]
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    #[inline]
    fn reload(&self) -> &ReloadRequest {
        self.reload()
    }

    #[inline]
    fn serialize_model(&self, path: PathBuf) -> Result<()> {
        Runtime::serialize_model(self, path)
    }

    #[inline]
    fn queue(
        &self,
        context: GenerateContext,
    ) -> Pin<Box<dyn Future<Output = SlotResult> + Send + '_>> {
        Box::pin(self.queue(context))
    }

    #[inline]
    fn process<'a>(
        &'a self,
        payloads: &'a mut [Payload],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
        Box::pin(self.process(payloads))
    }

    #[inline]
    fn maintain_cache(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(self.maintain_cache())
    }
}

#[tokio::main]
pub async fn run(receiver: Receiver<()>, env: Arc<RwLock<Environment>>) {
    {
        // this task constantly runs, cleaning up state cache
        let env = env.clone();
        tokio::spawn(async move {
            loop {
                if let Environment::Loaded { runtime, .. } = &*env.read().await {
                    runtime.maintain_cache().await;
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
    }

    while let Ok(()) = receiver.recv_async().await {
        if let Environment::Loaded { runtime, .. } = &*env.read().await {
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
