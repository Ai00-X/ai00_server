use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, Result};
use derivative::Derivative;
use flume::{Receiver, Sender};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use reload::{AdapterOption, BnfOption, Precision};
use safetensors::SafeTensors;
use salvo::oapi::ToSchema;
use serde::{de::DeserializeSeed, Deserialize, Serialize};
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
    sync::{Mutex, RwLock},
    time::Duration,
};
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        loader::{Loader, Lora, LoraBlend, Reader},
        model::{ContextAutoLimits, EmbedDevice, ModelBuilder, ModelInfo, ModelVersion, Quant},
        v4, v5, v6,
    },
    tensor::{serialization::Seed, TensorCpu},
    tokenizer::Tokenizer,
    wgpu::{Backends, PowerPreference},
};

use crate::{
    run::{GenerateContext, InitState, Runtime, StateId, Tokens},
    sampler::Sampler,
};

pub mod reload;
pub mod run;
pub mod sampler;

pub const MAX_TOKENS: usize = usize::MAX;

#[derive(Debug)]
pub enum Token {
    Start,
    Content(String),
    Stop(FinishReason, TokenCounter),
    Embed(Vec<f32>),
    Choose(Vec<f32>),
    Done,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, ToSchema)]
pub struct TokenCounter {
    #[serde(alias = "prompt_tokens")]
    pub prompt: usize,
    #[serde(alias = "completion_tokens")]
    pub completion: usize,
    #[serde(alias = "total_tokens")]
    pub total: usize,
    pub duration: Duration,
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

#[derive(Debug, Clone)]
pub enum ThreadRequest {
    /// Acquire a list of current available adapters.
    Adapter(Sender<AdapterList>),
    /// Get the current runtime info.
    Info(Sender<RuntimeInfo>),
    /// Request the runtime to complement a prompt.
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
    /// Additionally load an initial state.
    StateLoad {
        request: reload::State,
        sender: Option<Sender<bool>>,
    },
    /// Unload an initial state given its id.
    StateUnload(StateId),
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

#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    pub reload: ReloadRequest,
    pub model: ModelInfo,
    pub states: Vec<(StateId, InitState)>,
    pub tokenizer: Arc<Tokenizer>,
}

#[derive(Debug, Default, Clone)]
pub struct AdapterList(pub Vec<String>);

#[derive(Debug, Default, Clone)]
pub enum GenerateKind {
    /// Normal text completion.
    #[default]
    None,
    /// The (reversed) number of layer at which the output is as embedding.
    Embed { layer: usize },
    /// Choose options by perplexity.
    Choose { choices: Vec<String> },
}

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
        Default(value = "Arc::new(RwLock::new(sampler::nucleus::NucleusSampler::default()))")
    )]
    pub sampler: Arc<RwLock<dyn Sampler + Send + Sync>>,
    /// Generation output kind.
    pub kind: GenerateKind,
    /// Initial state ID.
    pub state: StateId,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize, ToSchema)]
#[derivative(Default)]
#[serde(default)]
pub struct ReloadRequest {
    /// Path to the model.
    #[salvo(schema(value_type = String))]
    pub model_path: PathBuf,
    /// List of LoRA blended on the model.
    pub lora: Vec<reload::Lora>,
    /// Path to the initial state.
    pub state: Vec<reload::State>,
    /// Specify layers that needs to be quantized.
    pub quant: usize,
    /// Quantization type (`Int8` or `NF4`).
    #[salvo(schema(value_type = sealed::Quant))]
    pub quant_type: Quant,
    /// Precision for intermediate tensors (`Fp16` or `Fp32`).
    pub precision: Precision,
    /// Maximum tokens to be processed in parallel at once.
    #[derivative(Default(value = "128"))]
    pub token_chunk_size: usize,
    /// Number of states that are cached on GPU.
    #[derivative(Default(value = "8"))]
    pub max_batch: usize,
    /// Device to put the embed tensor.
    #[salvo(schema(value_type = sealed::EmbedDevice))]
    pub embed_device: EmbedDevice,
    /// Path to the tokenizer.
    #[salvo(schema(value_type = String))]
    pub tokenizer_path: PathBuf,
    /// BNF options.
    pub bnf: BnfOption,
    /// Adapter selection.
    pub adapter: AdapterOption,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct SaveRequest {
    /// Path to save the model.
    #[serde(alias = "model_path")]
    #[salvo(schema(value_type = String))]
    pub path: PathBuf,
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

async fn load_init_state<R: Reader>(
    context: &Context,
    info: &ModelInfo,
    model: R,
) -> Result<TensorCpu<f32>> {
    let state = match info.version {
        ModelVersion::V4 => bail!("v4 does not support init state yet"),
        ModelVersion::V5 => v5::read_state(context, info, model).await,
        ModelVersion::V6 => v6::read_state(context, info, model).await,
        ModelVersion::V7 => bail!("v7 is not supported yet"),
    };
    state.map_err(Into::into)
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

    let file = File::open(model_path).await?;
    let data = unsafe { Mmap::map(&file) }?;

    let mut states = vec![];
    for reload::State {
        path,
        name,
        id,
        default,
    } in state
    {
        let name = match name {
            Some(name) => name,
            None => match path.file_name() {
                Some(name) => name.to_string_lossy().to_string(),
                None => continue,
            },
        };
        let file = File::open(path).await?;
        let data = unsafe { Mmap::map(&file) }?;
        let model = SafeTensors::deserialize(&data)?;
        match load_init_state(context, &info, model).await {
            Ok(data) => {
                let state = InitState {
                    name,
                    id,
                    data,
                    default,
                };
                log::info!("{:#?}", state);
                states.push(state);
            }
            Err(err) => log::warn!("initial state not loaded: {}", err),
        }
    }

    let runtime = match load {
        LoadType::SafeTensors => {
            let model = SafeTensors::deserialize(&data)?;
            if let Ok(data) = load_init_state(context, &info, model).await {
                let name = "internal".into();
                let id = StateId::new();
                let state = InitState {
                    name,
                    id,
                    data,
                    default: true,
                };
                states.push(state);
            }

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

            macro_rules! match_safe_tensors {
                (($v:expr, $p:expr), { $(($version:path, $precision:path, $model:ty, $build:ident, $bundle:ty)),+ }) => {
                    match ($v, $p) {
                        $(
                            ($version, $precision) => {
                                let model = builder.$build().await?;
                                let bundle = <$bundle>::new(model, max_batch);
                                Runtime::new(context, bundle, reload, states, tokenizer).await
                            }
                        )+
                        (version, _) => bail!("unsupported version: {:?}", version)
                    }
                }
            }
            match_safe_tensors!(
                (info.version, reload.precision),
                {
                    (ModelVersion::V4, Precision::Fp16, v4::Model, build_v4, v4::Bundle::<f16>),
                    (ModelVersion::V5, Precision::Fp16, v5::Model, build_v5, v5::Bundle::<f16>),
                    (ModelVersion::V6, Precision::Fp16, v6::Model, build_v6, v6::Bundle::<f16>),
                    (ModelVersion::V4, Precision::Fp32, v4::Model, build_v4, v4::Bundle::<f32>),
                    (ModelVersion::V5, Precision::Fp32, v5::Model, build_v5, v5::Bundle::<f32>),
                    (ModelVersion::V6, Precision::Fp32, v6::Model, build_v6, v6::Bundle::<f32>)
                }
            )
        }
        LoadType::Prefab => {
            use cbor4ii::{core::utils::SliceReader, serde::Deserializer};

            let reader = SliceReader::new(&data);
            let mut deserializer = Deserializer::new(reader);

            let context = context.clone();
            let reload = reload.clone();

            macro_rules! match_prefab {
                (($v:expr, $p:expr), { $(($version:path, $precision:path, $model:ty, $bundle:ty)),+ }) => {
                    match ($v, $p) {
                        $(
                            ($version, $precision) => {
                                let seed: Seed<_, $model> = Seed::new(&context);
                                let model = seed.deserialize(&mut deserializer)?;
                                let bundle = <$bundle>::new(model, reload.max_batch);
                                Runtime::new(context, bundle, reload, states, tokenizer).await
                            }
                        )+
                        (version, _) => bail!("unsupported version: {:?}", version)
                    }
                }
            }
            match_prefab!(
                (info.version, reload.precision),
                {
                    (ModelVersion::V4, Precision::Fp16, v4::Model, v4::Bundle::<f16>),
                    (ModelVersion::V5, Precision::Fp16, v5::Model, v5::Bundle::<f16>),
                    (ModelVersion::V6, Precision::Fp16, v6::Model, v6::Bundle::<f16>),
                    (ModelVersion::V4, Precision::Fp32, v4::Model, v4::Bundle::<f32>),
                    (ModelVersion::V5, Precision::Fp32, v5::Model, v5::Bundle::<f32>),
                    (ModelVersion::V6, Precision::Fp32, v6::Model, v6::Bundle::<f32>)
                }
            )
        }
    };

    Ok(runtime)
}

pub async fn model_route(receiver: Receiver<ThreadRequest>) -> Result<()> {
    let env: Arc<RwLock<Environment>> = Default::default();
    let queue: Arc<Mutex<Vec<GenerateContext>>> = Default::default();

    let sender = {
        let (sender, receiver) = flume::unbounded();
        let env = env.clone();
        tokio::spawn(crate::run::run(receiver, env));
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
        let Ok(request) = receiver.recv_async().await else {
            log::info!("core exit");
            break Ok(());
        };

        let listen = async {
            match request {
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
                            let states = runtime.states().await;
                            let tokenizer = runtime.tokenizer();
                            let _ = sender.send(RuntimeInfo {
                                reload,
                                model,
                                states,
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
                        log::info!("loading model {:?}", request.model_path);
                        log::info!("{:#?}", info);
                        log::info!("type: {:?}", load);

                        let context = create_context(request.adapter, &info).await?;
                        log::info!("{:#?}", context.adapter.get_info());

                        let mut env = env.write().await;
                        // drop(mem::take(&mut *env));
                        'unload: {
                            let env = std::mem::take(&mut *env);
                            let _context = match env {
                                Environment::Loaded(runtime) => runtime.context().clone(),
                                Environment::None => break 'unload,
                            };
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
                                log::info!("model loaded")
                            }
                            Err(err) => {
                                callback(false);
                                log::error!("load runtime failed: {}", err);
                            }
                        };
                    });
                }
                ThreadRequest::Unload => {
                    let env = env.clone();
                    tokio::spawn(async move {
                        let mut env = env.write().await;
                        let env = std::mem::take(&mut *env);
                        log::info!("runtime unloaded");

                        let _context = match env {
                            Environment::Loaded(runtime) => runtime.context().clone(),
                            Environment::None => return,
                        };
                    });
                }
                ThreadRequest::StateLoad { request, sender } => {
                    let env = env.clone();
                    let load = async move {
                        let env = env.read().await;
                        let Environment::Loaded(runtime) = &*env else {
                            bail!("runtime not loaded")
                        };

                        let reload::State {
                            path,
                            name,
                            id,
                            default,
                        } = request;
                        let name = match name {
                            Some(name) => name,
                            None => match path.file_name() {
                                Some(name) => name.to_string_lossy().to_string(),
                                None => bail!("failed to parse state name"),
                            },
                        };
                        let file = File::open(&path).await?;
                        let data = unsafe { Mmap::map(&file)? };

                        let context = runtime.context();
                        let info = runtime.info();
                        let model = SafeTensors::deserialize(&data)?;
                        match load_init_state(context, info, model).await {
                            Ok(data) => {
                                let state = InitState {
                                    name,
                                    id,
                                    data,
                                    default,
                                };
                                log::info!("{:#?}", state);
                                runtime.load_init_state(state).await;
                            }
                            Err(err) => log::warn!("initial state not loaded: {}", err),
                        };
                        Ok(())
                    };
                    let callback = move |result: bool| {
                        if let Some(sender) = sender {
                            let _ = sender.send(result);
                        }
                    };
                    tokio::spawn(async move {
                        match load.await {
                            Ok(_) => {
                                callback(true);
                                log::info!("state loaded")
                            }
                            Err(err) => {
                                callback(false);
                                log::error!("load state failed: {}", err);
                            }
                        };
                    });
                }
                ThreadRequest::StateUnload(id) => {
                    let env = env.clone();
                    tokio::spawn(async move {
                        let env = env.read().await;
                        let Environment::Loaded(runtime) = &*env else {
                            return;
                        };
                        runtime.unload_init_state(id).await;
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

                    let choices = match &request.kind {
                        GenerateKind::Choose { choices } => {
                            let choices: Vec<_> = choices
                                .iter()
                                .map(|prompt| tokenizer.encode(prompt.as_bytes()))
                                .try_collect()?;
                            choices.into_iter().map(Tokens).collect()
                        }
                        _ => vec![],
                    };

                    let context = GenerateContext {
                        prompt_tokens: tokens.to_vec(),
                        prompt_cached: Default::default(),
                        prefix: Default::default(),
                        suffix: tokens,
                        output: None,
                        choices,
                        model_text: vec![],
                        buffer: vec![],
                        model_tokens: vec![],
                        formatters: vec![],
                        instant: None,
                        request,
                        sender: token_sender,
                    };

                    let env = env.clone();
                    let queue = queue.clone();
                    let sender = sender.clone();
                    tokio::spawn(async move {
                        let context = &mut env.read().await.enqueue(context).await;
                        let mut queue = queue.lock().await;
                        queue.append(context);
                        let _ = sender.send(());
                    });
                }
                ThreadRequest::Save { request, sender } => {
                    let env = env.clone();
                    tokio::spawn(async move {
                        let env = &(*env.read().await);
                        if let Environment::Loaded(runtime) = env {
                            log::info!("serializing model into {:?}", &request.path);
                            let _ = match runtime.serialize_model(request.path).await {
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

#[allow(dead_code)]
mod sealed {
    use salvo::oapi::ToSchema;

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, ToSchema)]
    pub enum Quant {
        /// No quantization.
        #[default]
        None,
        /// Use `Int8` quantization.
        Int8,
        /// Use `NF4` quantization.
        NF4,
    }

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ToSchema)]
    pub enum EmbedDevice {
        #[default]
        Cpu,
        Gpu,
    }
}
