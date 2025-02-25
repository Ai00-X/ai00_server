use std::{
    collections::HashMap,
    error::Error,
    ops::Deref,
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
        model::{
            ContextAutoLimits, EmbedDevice, ModelBuilder, ModelInfo, ModelVersion, Quant, State,
        },
        v4, v5, v6, v7, Runtime,
    },
    tensor::{serialization::Seed, TensorCpu},
    tokenizer::Tokenizer,
    wgpu::{Backends, PowerPreference},
};

use crate::{
    run::{GenerateContext, Tokens},
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
    Loaded(RuntimeInfo),
    #[default]
    None,
}

#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct RuntimeInfo {
    pub reload: Arc<ReloadRequest>,
    pub info: ModelInfo,
    #[derivative(Debug = "ignore")]
    pub model: Arc<dyn ModelSerialize + Send + Sync>,
    pub states: Vec<InitState>,
    pub tokenizer: Arc<Tokenizer>,
}

struct Model<M>(M);

pub trait ModelSerialize {
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
    Choose {
        choices: Vec<String>,
        calibrate: bool,
    },
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
        ModelVersion::V7 => v7::read_state(context, info, model).await,
    };
    state.map_err(Into::into)
}

async fn load_init_states() {}

async fn load_model<R: Reader>(
    context: &Context,
    info: &ModelInfo,
    reload: &ReloadRequest,
    model: R,
) -> Result<(
    Arc<dyn Runtime + Send + Sync>,
    Arc<dyn State + Send + Sync>,
    Arc<dyn ModelSerialize + Send + Sync>,
)> {
    let builder = ModelBuilder::new(context, model);

    todo!()
}

pub async fn serve(receiver: Receiver<ThreadRequest>) -> Result<()> {
    let env: Arc<RwLock<Environment>> = Default::default();

    loop {
        let Ok(request) = receiver.recv_async().await else {
            break Ok(());
        };

        let env = env.clone();
        let listen = async move {
            match request {
                ThreadRequest::Adapter(sender) => {
                    let _ = sender.send(list_adapters());
                }
                ThreadRequest::Info(sender) => {
                    let env = env.read().await;
                    if let Environment::Loaded(runtime) = &*env {
                        let _ = sender.send(runtime.clone());
                    }
                }
                ThreadRequest::Generate {
                    request,
                    tokenizer,
                    sender,
                } => todo!(),
                ThreadRequest::Reload { request, sender } => {
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
                    log::info!("{:#?}", request);
                    log::info!("{:#?}", info);
                    log::info!("model type: {:?}", load);

                    let context = create_context(request.adapter, &info).await?;
                    log::info!("{:#?}", context.adapter.get_info());

                    let mut env = env.write().await;
                    let _ = std::mem::take(&mut *env);

                    let tokenizer = Arc::new(load_tokenizer(&request.tokenizer_path).await?);

                    let mut states = Vec::with_capacity(request.state.len());
                    for reload::State {
                        path,
                        name,
                        id,
                        default,
                    } in request.state.iter().cloned()
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
                        match load_init_state(&context, &info, model).await {
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

                    let reload = Arc::new(*request);
                    let info = RuntimeInfo {
                        reload,
                        info,
                        model: todo!(),
                        states,
                        tokenizer,
                    };
                }
                ThreadRequest::Unload => todo!(),
                ThreadRequest::StateLoad { request, sender } => todo!(),
                ThreadRequest::StateUnload(state_id) => todo!(),
                ThreadRequest::Save { request, sender } => todo!(),
            };
            Ok(())
        };
        tokio::spawn(listen);
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
        /// Use `SF4` quantization.
        SF4,
    }

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ToSchema)]
    pub enum EmbedDevice {
        #[default]
        Cpu,
        Gpu,
    }
}
