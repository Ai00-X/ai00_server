use std::{
    collections::HashMap,
    convert::Infallible,
    fs::File,
    io::{BufReader, Read},
    mem,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{bail, Result};
use bnf_sampler::{utils::U8ArrayWrapper, vocabulary::Vocabulary};
use derivative::Derivative;
use flume::{Receiver, Sender};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use qp_trie::Trie;
use rustc_hash::FxHashMap;
use safetensors::SafeTensors;

use salvo::oapi::{ToResponse, ToSchema};
use serde::{de::DeserializeSeed, Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{
        loader::{Loader, Lora, LoraBlend},
        v4, v5, v6, Build, BuildFuture, EmbedDevice, Model, ModelBuilder, ModelInfo, ModelState,
        ModelVersion, Quant, StateBuilder,
    },
    tensor::serialization::Seed,
    tokenizer::Tokenizer,
    wgpu::{Backends, PowerPreference},
};

use crate::{
    config::{AdapterOption, BnfOption},
    run::{GenerateContext, Runner, Runtime, SlotResult, Tokens},
    sampler::{nucleus::NucleusSampler, Sampler},
};

pub const MAX_TOKENS: usize = 4096;

#[derive(Debug)]
pub enum Token {
    Start,
    Content(String),
    Stop(FinishReason, TokenCounter),
    Embed(Vec<f32>),
    Done,
}

#[derive(Debug, Default, Clone, Copy, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub enum FinishReason {
    /// API returned complete model output.
    Stop,
    /// Incomplete model output due to max_tokens parameter or token limit.
    Length,
    /// Omitted content due to a flag from our content filters.
    ContentFilter,
    /// API response still in progress or incomplete.
    #[default]
    #[serde(untagged)]
    Null,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum Array<T: ToSchema + 'static> {
    #[default]
    None,
    Item(T),
    Vec(Vec<T>),
}

impl<T> From<Array<T>> for Vec<T>
where
    T: std::fmt::Debug + Clone + Serialize + ToSchema,
{
    fn from(value: Array<T>) -> Self {
        match value {
            Array::None => vec![],
            Array::Item(item) => vec![item],
            Array::Vec(vec) => vec,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ThreadRequest {
    /// Acquire a list of current available adapters.
    Adapter(Sender<AdapterList>),
    /// Get the current runtime info.
    Info(Sender<RuntimeInfo>),
    /// Request the server to complement a prompt.
    Generate {
        request: Box<GenerateRequest>,
        tokenizer: Arc<Tokenizer>,
        sender: Sender<Token>,
    },
    /// Reload the runtime with custom config.
    Reload {
        request: Box<ReloadRequest>,
        sender: Option<Sender<bool>>,
    },
    /// Unload the runtime.
    Unload,
    /// Save the current model with config.
    Save {
        request: SaveRequest,
        sender: Sender<bool>,
    },
}

#[derive(Default)]
pub enum Environment {
    Loaded {
        runtime: Box<dyn Runner + Send + Sync>,
    },
    #[default]
    None,
}

impl Environment {
    pub async fn enqueue(&self, context: GenerateContext) -> Vec<GenerateContext> {
        let mut queue = vec![];
        match self {
            Environment::Loaded { runtime, .. } => match runtime.queue(context).await {
                SlotResult::Success(batch) => log::info!("queued task at slot {batch}"),
                SlotResult::Fault(batch) => log::info!("swapped task at slot {batch}"),
                SlotResult::Failure(context) => {
                    log::info!("failed to queue task");
                    queue.push(*context);
                }
                SlotResult::Error(reason) => log::warn!("queue task failed: {}", reason),
            },
            Environment::None => queue.push(context),
        };
        queue
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    pub reload: ReloadRequest,
    pub model: ModelInfo,
    pub tokenizer: Arc<Tokenizer>,
}

#[derive(Debug, Default, Clone)]
pub struct AdapterList(pub Vec<String>);

#[derive(Clone, Derivative)]
#[derivative(Debug, Default)]
pub struct GenerateRequest {
    /// The prompt for the model.
    pub prompt: String,
    /// All text the model output earlier.
    pub model_text: String,
    /// Output token limit.
    pub max_tokens: usize,
    /// Stop indicators.
    pub stop: Vec<String>,
    /// Bias added to tokens before sampling.
    pub bias: Arc<HashMap<u16, f32>>,
    /// Optional BNF schema for formatted generation.
    pub bnf_schema: Option<String>,
    /// Sampler parameters.
    #[derivative(
        Debug = "ignore",
        Default(value = "Arc::new(RwLock::new(NucleusSampler::default()))")
    )]
    pub sampler: Arc<RwLock<dyn Sampler + Send + Sync>>,
    /// Whether this is an embedding request.
    pub embed: bool,
    /// The (reversed) number of layer at which the output is as embedding.
    pub embed_layer: usize,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct ReloadRequest {
    /// Path to the model.
    pub model_path: PathBuf,
    /// List of LoRA blended on the model.
    pub lora: Vec<crate::config::Lora>,
    /// Specify layers that needs to be quantized.
    pub quant: usize,
    /// Quantization type (Int8 or NF4).
    pub quant_type: Quant,
    /// Whether to use alternative GEMM kernel to speed-up long prompts.
    #[derivative(Default(value = "true"))]
    pub turbo: bool,
    /// Maximum tokens to be processed in parallel at once.
    #[derivative(Default(value = "128"))]
    pub token_chunk_size: usize,
    /// The chunk size of layers in model state.
    #[derivative(Default(value = "4"))]
    pub state_chunk_size: usize,
    #[derivative(Default(value = "8"))]
    /// Maximum number of batches that are active at once.
    pub max_runtime_batch: usize,
    /// Number of states that are cached on GPU.
    #[derivative(Default(value = "16"))]
    pub max_batch: usize,
    /// Device to put the embed tensor.
    pub embed_device: EmbedDevice,
    /// Path to the tokenizer.
    pub tokenizer_path: PathBuf,
    /// BNF options.
    pub bnf: BnfOption,
    /// Adapter selection.
    pub adapter: AdapterOption,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SaveRequest {
    /// Path to save model.
    pub model_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Prefab {
    info: ModelInfo,
}

#[derive(Debug, Clone, Copy)]
enum LoadType {
    SafeTensors,
    Prefab,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, ToSchema, ToResponse)]
pub struct TokenCounter {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone)]
pub struct ThreadState {
    pub sender: Sender<ThreadRequest>,
    pub model_path: PathBuf,
}

fn list_adapters() -> AdapterList {
    let backends = Backends::all();
    let instance = Instance::new();
    let list = instance
        .enumerate_adapters(backends)
        .map(|adapter| adapter.get_info())
        .map(|info| format!("{} ({:?})", info.name, info.backend))
        .collect();
    AdapterList(list)
}

async fn create_context(adapter: AdapterOption, info: &ModelInfo) -> Result<Context> {
    let backends = Backends::all();
    let instance = Instance::new();
    let adapter = match adapter {
        AdapterOption::Auto => instance.adapter(PowerPreference::HighPerformance).await,
        AdapterOption::Economical => instance.adapter(PowerPreference::LowPower).await,
        AdapterOption::Manual(selection) => instance.select_adapter(backends, selection),
    }?;
    let context = ContextBuilder::new(adapter)
        .with_auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(Tokenizer::new(&contents)?)
}

fn load_vocab(tokenizer: &Tokenizer) -> Vocabulary {
    let vocab = tokenizer.bytes_to_token_index();
    let token_to_id: Trie<_, _> = vocab
        .iter()
        .map(|(k, v)| (U8ArrayWrapper(k.clone().into_boxed_slice()), *v as u32))
        .collect();
    let id_to_token: FxHashMap<_, _> = vocab.iter().map(|(k, v)| (*v as u32, k.clone())).collect();
    let id_to_token_string: FxHashMap<_, _> = vocab
        .iter()
        .map(|(k, v)| (*v as u32, String::from_utf8_lossy(k).to_string()))
        .collect();
    Vocabulary {
        token_to_id,
        id_to_token,
        id_to_token_string,
    }
}

async fn load_model<M, S>(
    context: &Context,
    request: &ReloadRequest,
    load_type: LoadType,
) -> Result<(M, S)>
where
    S: ModelState,
    M: Model<State = S>,
    for<'a> ModelBuilder<SafeTensors<'a>>: BuildFuture<M, Error = anyhow::Error>,
    for<'de> Seed<'de, Context, M>: DeserializeSeed<'de, Value = M>,
    StateBuilder: Build<S, Error = Infallible>,
{
    let ReloadRequest {
        model_path,
        quant,
        quant_type,
        lora,
        token_chunk_size,
        turbo,
        embed_device,
        ..
    } = request.clone();

    let file = File::open(model_path)?;
    let data = unsafe { Mmap::map(&file) }?;

    let model = match load_type {
        LoadType::SafeTensors => {
            let model = SafeTensors::deserialize(&data)?;

            let quant = (0..quant).map(|layer| (layer, quant_type)).collect();
            let model = ModelBuilder::new(context, model)
                .with_quant(quant)
                .with_turbo(turbo)
                .with_embed_device(embed_device)
                .with_token_chunk_size(token_chunk_size);

            let lora: Vec<_> = lora
                .into_iter()
                .map(|lora| -> Result<_> {
                    let file = File::open(lora.path)?;
                    let data = unsafe { Mmap::map(&file) }?;
                    let blend = LoraBlend::full(lora.alpha);
                    Ok((data, blend))
                })
                .try_collect()?;
            let lora: Vec<_> = lora
                .iter()
                .map(|(data, blend)| -> Result<_> {
                    let data = SafeTensors::deserialize(data)?;
                    let blend = blend.clone();
                    Ok(Lora { data, blend })
                })
                .try_collect()?;
            let model: M = lora
                .into_iter()
                .fold(model, |acc, x| acc.add_lora(x))
                .build()
                .await?;
            model
        }
        LoadType::Prefab => {
            use cbor4ii::{core::utils::SliceReader, serde::Deserializer};
            let reader = SliceReader::new(&data);
            let mut deserializer = Deserializer::new(reader);
            let seed = Seed::<Context, M>::new(context);
            let model: M = seed.deserialize(&mut deserializer)?;
            model
        }
    };

    let state: S = StateBuilder::new(context, model.info())
        .with_num_batch(request.max_batch)
        .with_chunk_size(request.state_chunk_size)
        .build()?;
    Ok((model, state))
}

#[tokio::main]
pub async fn model_route(receiver: Receiver<ThreadRequest>) -> Result<()> {
    let env: Arc<RwLock<Environment>> = Default::default();
    let queue: Arc<Mutex<Vec<GenerateContext>>> = Default::default();

    let sender = {
        let (sender, receiver) = flume::unbounded();
        let env = env.clone();
        tokio::task::spawn_blocking(move || crate::run::run(receiver, env));
        sender
    };

    let dequeue = {
        let env = env.clone();
        let queue = queue.clone();
        let sender = sender.clone();

        async move {
            loop {
                let mut queue = queue.lock().await;
                let mut temp = vec![];
                for context in queue.drain(..) {
                    temp.append(&mut env.read().await.enqueue(context).await);
                    let _ = sender.send(());
                }
                std::mem::swap(&mut *queue, &mut temp);
                drop(queue);

                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    };
    tokio::spawn(dequeue);

    loop {
        let listen = async {
            match receiver.recv_async().await.unwrap() {
                ThreadRequest::Adapter(sender) => {
                    tokio::spawn(async move {
                        let _ = sender.send(list_adapters());
                    });
                }
                ThreadRequest::Info(sender) => {
                    let env = env.clone();
                    tokio::spawn(async move {
                        let env = &(*env.read().await);
                        if let Environment::Loaded { runtime } = env {
                            let reload = runtime.reload().clone();
                            let model = runtime.info().clone();
                            let tokenizer = runtime.tokenizer();
                            let _ = sender.send(RuntimeInfo {
                                reload,
                                model,
                                tokenizer,
                            });
                        }
                    });
                }
                ThreadRequest::Reload {
                    request,
                    sender: reload_sender,
                } => {
                    let request = *request;
                    let sender = sender.clone();
                    let env = env.clone();
                    let reload = async move {
                        let sender = sender.clone();

                        let file = File::open(&request.model_path)?;
                        let data = unsafe { Mmap::map(&file)? };

                        let (info, load_type) = {
                            let st = SafeTensors::deserialize(&data);
                            let prefab = cbor4ii::serde::from_slice::<Prefab>(&data);
                            match (st, prefab) {
                                (Ok(model), _) => (Loader::info(&model)?, LoadType::SafeTensors),
                                (_, Ok(prefab)) => (prefab.info, LoadType::Prefab),
                                _ => bail!("failed to read model info"),
                            }
                        };
                        log::info!("{:#?}", info);
                        log::info!("type: {:?}", load_type);

                        let context = create_context(request.adapter, &info).await?;
                        let tokenizer = load_tokenizer(&request.tokenizer_path)?;
                        let vocab = load_vocab(&tokenizer);
                        log::info!("{:#?}", context.adapter.get_info());

                        let mut env = env.write().await;
                        drop(mem::take(&mut *env));

                        let runtime: Box<dyn Runner + Send + Sync> = match info.version {
                            ModelVersion::V4 => {
                                type M<'a> = v4::Model<'a, f16>;
                                let (model, state) =
                                    load_model::<M, _>(&context, &request, load_type).await?;
                                Box::new(Runtime::new(tokenizer, vocab, model, state, request))
                            }
                            ModelVersion::V5 => {
                                type M<'a> = v5::Model<'a, f16>;
                                let (model, state) =
                                    load_model::<M, _>(&context, &request, load_type).await?;
                                Box::new(Runtime::new(tokenizer, vocab, model, state, request))
                            }
                            ModelVersion::V6 => {
                                type M<'a> = v6::Model<'a, f16>;
                                let (model, state) =
                                    load_model::<M, _>(&context, &request, load_type).await?;
                                Box::new(Runtime::new(tokenizer, vocab, model, state, request))
                            }
                        };
                        *env = Environment::Loaded { runtime };

                        let _ = sender.send(());
                        anyhow::Ok(())
                    };
                    let callback = move |result: bool| {
                        if let Some(sender) = reload_sender {
                            let _ = sender.send(result);
                        }
                    };
                    tokio::spawn(async move {
                        match reload.await {
                            Ok(_) => {
                                callback(true);
                                log::info!("model reloaded")
                            }
                            Err(err) => {
                                callback(false);
                                log::error!("reload model failed: {}", err);
                            }
                        };
                    });
                }
                ThreadRequest::Unload => {
                    let env = env.clone();
                    tokio::spawn(async move {
                        let mut env = env.write().await;
                        *env = Environment::None;
                        log::info!("model unloaded");
                    });
                }
                ThreadRequest::Generate {
                    request,
                    tokenizer,
                    sender: token_sender,
                } => {
                    let request = *request;
                    let tokens = Tokens(tokenizer.encode(request.prompt.as_bytes())?);
                    let model_tokens = Tokens(tokenizer.encode(request.model_text.as_bytes())?);
                    // init sampler state here
                    request.sampler.write().await.init(&model_tokens);

                    let context = GenerateContext {
                        prompt_tokens: tokens.to_vec(),
                        prompt_cached: false,
                        prefix: Default::default(),
                        suffix: tokens,
                        model_text: Default::default(),
                        buffer: Default::default(),
                        model_tokens: Default::default(),
                        bnf_sampler: None,
                        request,
                        sender: token_sender,
                    };

                    let env = env.clone();
                    let queue = queue.clone();
                    let sender = sender.clone();
                    tokio::spawn(async move {
                        let mut queue = queue.lock().await;
                        queue.append(&mut env.read().await.enqueue(context).await);
                        let _ = sender.send(());
                    });
                }
                ThreadRequest::Save { request, sender } => {
                    let env = env.clone();
                    tokio::spawn(async move {
                        let env = &(*env.read().await);
                        if let Environment::Loaded { runtime, .. } = env {
                            log::info!("serializing model into {:?}", &request.model_path);
                            let _ = match runtime.serialize_model(request.model_path) {
                                Ok(()) => sender.send(true),
                                Err(err) => {
                                    log::error!("{}", err);
                                    sender.send(false)
                                }
                            };
                        }
                    });
                }
            };
            anyhow::Ok(())
        };

        if let Err(err) = listen.await {
            log::error!("{err}");
        }
    }
}
