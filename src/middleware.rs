use std::{
    collections::HashMap,
    mem,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{anyhow, bail, Result};
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
use tokio::{fs::File, io::BufReader};
use tokio::{
    io::AsyncReadExt,
    sync::{Mutex, RwLock},
};
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        loader::{Loader, Lora, LoraBlend},
        model::{
            Build, ContextAutoLimits, EmbedDevice, ModelBuilder, ModelInfo, ModelVersion, Quant,
        },
        v4, v5, v6,
    },
    tensor::{serialization::Seed, TensorCpu},
    tokenizer::Tokenizer,
    wgpu::{Backends, Maintain, PowerPreference},
};

use crate::{
    config::{AdapterOption, BnfOption},
    run::{GenerateContext, Runtime, SlotResult, Tokens},
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
    Loaded(Runtime),
    #[default]
    None,
}

impl Environment {
    pub async fn enqueue(&self, context: GenerateContext) -> Vec<GenerateContext> {
        let mut queue = vec![];
        match self {
            Environment::Loaded(runtime) => {
                match runtime.queue(context).await.expect("queue task error") {
                    SlotResult::Success(batch) => log::info!("queued task at slot {batch}"),
                    SlotResult::Fault(batch) => log::info!("swapped task at slot {batch}"),
                    SlotResult::Failure(context) => {
                        log::info!("failed to queue task");
                        queue.push(*context);
                    }
                    SlotResult::Error(reason) => log::warn!("queue task failed: {}", reason),
                }
            }
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
    /// Path to the initial state.
    pub state: Option<crate::config::State>,
    /// Specify layers that needs to be quantized.
    pub quant: usize,
    /// Quantization type (Int8 or NF4).
    pub quant_type: Quant,
    /// Maximum tokens to be processed in parallel at once.
    #[derivative(Default(value = "128"))]
    pub token_chunk_size: usize,
    /// Number of states that are cached on GPU.
    #[derivative(Default(value = "8"))]
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
    pub path: PathBuf,
}

fn list_adapters() -> AdapterList {
    let backends = Backends::all();
    let instance = web_rwkv::wgpu::Instance::default();
    let list = instance
        .enumerate_adapters(backends)
        .into_iter()
        .map(|adapter| adapter.get_info())
        .map(|info| format!("{} ({:?})", info.name, info.backend))
        .collect();
    AdapterList(list)
}

async fn create_context(adapter: AdapterOption, info: &ModelInfo) -> Result<Context> {
    let backends = Backends::all();
    let instance = web_rwkv::wgpu::Instance::default();
    let adapter = match adapter {
        AdapterOption::Auto => instance.adapter(PowerPreference::HighPerformance).await,
        AdapterOption::Economical => instance.adapter(PowerPreference::LowPower).await,
        AdapterOption::Manual(selection) => Ok(instance
            .enumerate_adapters(backends)
            .into_iter()
            .nth(selection)
            .ok_or(web_rwkv::context::CreateEnvironmentError::RequestAdapterFailed)?),
    }?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

async fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    let file = File::open(path).await?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
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

async fn load_init_state(
    context: &Context,
    info: &ModelInfo,
    model: SafeTensors<'_>,
) -> Option<TensorCpu<f32>> {
    let state = match info.version {
        ModelVersion::V4 => Err(anyhow!("v4 does not support init state yet")),
        ModelVersion::V5 => v5::read_state(context, info, model).await,
        ModelVersion::V6 => v6::read_state(context, info, model).await,
    };
    match state {
        Ok(state) => {
            log::info!("initial state loaded");
            Some(state)
        }
        Err(err) => {
            log::warn!("initial state not loaded: {}", err);
            None
        }
    }
}

async fn load_runtime(
    context: &Context,
    reload: &ReloadRequest,
    info: ModelInfo,
    load: LoadType,
) -> Result<Runtime> {
    let ReloadRequest {
        model_path,
        lora,
        state,
        quant,
        quant_type,
        max_batch,
        embed_device,
        tokenizer_path,
        ..
    } = reload.clone();

    let tokenizer = load_tokenizer(tokenizer_path).await?;
    let vocab = load_vocab(&tokenizer);

    let file = File::open(model_path).await?;
    let data = unsafe { Mmap::map(&file) }?;

    let runtime = match load {
        LoadType::SafeTensors => {
            let model = SafeTensors::deserialize(&data)?;

            let init_state = match state {
                Some(state) => {
                    let file = File::open(state.path).await?;
                    let data = unsafe { Mmap::map(&file) }?;
                    let model = SafeTensors::deserialize(&data)?;
                    load_init_state(context, &info, model).await
                }
                None => load_init_state(context, &info, model).await,
            };

            let model = SafeTensors::deserialize(&data)?;
            let quant = (0..quant).map(|layer| (layer, quant_type)).collect();
            let lora = {
                let mut x = Vec::with_capacity(lora.len());
                for lora in lora.into_iter() {
                    let file = File::open(lora.path).await?;
                    let data = unsafe { Mmap::map(&file) }?;
                    let blend = LoraBlend::full(lora.alpha);
                    x.push((data, blend))
                }
                x
            };
            let lora: Vec<_> = lora
                .iter()
                .map(|(data, blend)| -> Result<_> {
                    let data = SafeTensors::deserialize(data)?;
                    let blend = blend.clone();
                    Ok(Lora { data, blend })
                })
                .try_collect()?;

            let builder = ModelBuilder::new(context, model)
                .quant(quant)
                .embed_device(embed_device);
            let builder = lora.into_iter().fold(builder, |b, x| b.lora(x));

            let context = context.clone();
            let reload = reload.clone();
            match info.version {
                ModelVersion::V4 => {
                    let model = Build::<v4::Model>::build(builder).await?;
                    let builder = v4::ModelRuntime::<f16>::new(model, max_batch);
                    Runtime::new(context, builder, reload, init_state, tokenizer, vocab).await
                }
                ModelVersion::V5 => {
                    let model = Build::<v5::Model>::build(builder).await?;
                    let builder = v5::ModelRuntime::<f16>::new(model, max_batch);
                    Runtime::new(context, builder, reload, init_state, tokenizer, vocab).await
                }
                ModelVersion::V6 => {
                    let model = Build::<v6::Model>::build(builder).await?;
                    let builder = v6::ModelRuntime::<f16>::new(model, max_batch);
                    Runtime::new(context, builder, reload, init_state, tokenizer, vocab).await
                }
            }
        }
        LoadType::Prefab => {
            use cbor4ii::{core::utils::SliceReader, serde::Deserializer};

            let reader = SliceReader::new(&data);
            let mut deserializer = Deserializer::new(reader);

            let context = context.clone();
            let reload = reload.clone();
            match info.version {
                ModelVersion::V4 => {
                    let seed: Seed<_, v4::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let builder = v4::ModelRuntime::<f16>::new(model, reload.max_batch);
                    Runtime::new(context, builder, reload, None, tokenizer, vocab).await
                }
                ModelVersion::V5 => {
                    let seed: Seed<_, v5::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let builder = v5::ModelRuntime::<f16>::new(model, reload.max_batch);
                    Runtime::new(context, builder, reload, None, tokenizer, vocab).await
                }
                ModelVersion::V6 => {
                    let seed: Seed<_, v6::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let builder = v6::ModelRuntime::<f16>::new(model, reload.max_batch);
                    Runtime::new(context, builder, reload, None, tokenizer, vocab).await
                }
            }
        }
    };

    Ok(runtime)
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
                        if let Environment::Loaded(runtime) = env {
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

                        let file = File::open(&request.model_path).await?;
                        let data = unsafe { Mmap::map(&file)? };

                        let (info, load) = {
                            let st = SafeTensors::deserialize(&data);
                            let prefab = cbor4ii::serde::from_slice::<Prefab>(&data);
                            match (st, prefab) {
                                (Ok(model), _) => (Loader::info(&model)?, LoadType::SafeTensors),
                                (_, Ok(prefab)) => (prefab.info, LoadType::Prefab),
                                _ => bail!("failed to read model info"),
                            }
                        };
                        log::info!("{:#?}", info);
                        log::info!("type: {:?}", load);

                        let context = create_context(request.adapter, &info).await?;
                        log::info!("{:#?}", context.adapter.get_info());

                        let mut env = env.write().await;
                        // drop(mem::take(&mut *env));
                        'unload: {
                            let env = mem::take(&mut *env);
                            let context = match env {
                                Environment::Loaded(runtime) => runtime.context().clone(),
                                Environment::None => break 'unload,
                            };
                            context.queue.submit(None);
                            context.device.poll(Maintain::Wait);
                        }

                        let runtime = load_runtime(&context, &request, info, load).await?;
                        *env = Environment::Loaded(runtime);

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
                        let env = mem::take(&mut *env);
                        log::info!("model unloaded");

                        let context = match env {
                            Environment::Loaded(runtime) => runtime.context().clone(),
                            Environment::None => return,
                        };
                        context.queue.submit(None);
                        context.device.poll(Maintain::Wait);
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
                        if let Environment::Loaded(runtime) = env {
                            log::info!("serializing model into {:?}", &request.model_path);
                            let _ = match runtime.serialize_model(request.model_path).await {
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
