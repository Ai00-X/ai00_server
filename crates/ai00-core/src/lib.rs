use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, Result};
use derivative::Derivative;
use flume::{Receiver, Sender};
use futures::future::join_all;
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
    sync::RwLock,
    time::Duration,
};
use web_rwkv::{
    context::{Context, ContextBuilder, ContextError, InstanceExt},
    runtime::{
        infer::{InferInput, InferOutput},
        loader::{Loader, Lora, LoraBlend, Reader},
        model::{
            Bundle, ContextAutoLimits, EmbedDevice, ModelBuilder, ModelInfo, ModelVersion, Quant,
            State,
        },
        v4, v5, v6, v7, Runtime, TokioRuntime,
    },
    tensor::{serialization::Seed, TensorCpu, TensorError, TensorInit},
    tokenizer::Tokenizer,
    wgpu::{Backends, PowerPreference},
};

use crate::{run::GenerateContext, sampler::Sampler};

pub mod reload;
pub mod run;
pub mod sampler;

pub const MAX_TOKENS: usize = usize::MAX;

#[derive(Debug)]
pub enum Token {
    Start,
    Content(String),
    Stop(FinishReason, TokenCounter),
    Embed(Vec<f32>, [usize; 4]),
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
    /// Save the current model with config.
    Save {
        request: SaveRequest,
        sender: Sender<bool>,
    },
}

#[derive(Default)]
pub enum Environment {
    Loaded {
        info: RuntimeInfo,
        runtime: Arc<dyn Runtime + Send + Sync>,
        model: Arc<dyn ModelSerialize + Send + Sync>,
        sender: Sender<GenerateContext>,
    },
    #[default]
    None,
}

#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct RuntimeInfo {
    pub reload: Arc<ReloadRequest>,
    pub info: ModelInfo,
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
    /// The state of input.
    State,
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
    /// Initial state.
    pub state: Arc<InputState>,
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

#[derive(Debug, Default, Clone, Serialize, Deserialize, ToSchema)]
pub struct StateValue {
    pub name: String,
    pub id: StateId,
    pub data: Vec<f32>,
    pub shape: [usize; 4],
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, ToSchema)]
pub struct StateFile {
    pub name: String,
    pub id: StateId,
    #[salvo(schema(value_type = String))]
    pub path: PathBuf,
}

/// State input from the user. Can be a single ID or full state data.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum InputState {
    Key(StateId),
    Value(StateValue),
    File(StateFile),
}

impl Default for InputState {
    fn default() -> Self {
        Self::Key(Default::default())
    }
}

impl InputState {
    pub fn id(&self) -> StateId {
        match self {
            InputState::Key(id) => *id,
            InputState::Value(value) => value.id,
            InputState::File(file) => file.id,
        }
    }
}

#[derive(Derivative, Clone, Serialize, Deserialize)]
#[derivative(Debug)]
pub struct InitState {
    pub name: String,
    pub id: StateId,
    pub default: bool,
    #[derivative(Debug = "ignore")]
    pub data: TensorCpu<f32>,
}

impl TryFrom<StateValue> for InitState {
    type Error = TensorError;

    fn try_from(
        StateValue {
            name,
            id,
            data,
            shape,
        }: StateValue,
    ) -> Result<Self, Self::Error> {
        let default = false;
        let data = TensorCpu::from_data(shape, data)?;
        Ok(Self {
            name,
            id,
            default,
            data,
        })
    }
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
            .ok_or(ContextError::RequestAdapterFailed)?),
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

async fn load_model_state<R: Reader>(
    context: &Context,
    info: &ModelInfo,
    model: R,
) -> Result<TensorCpu<f32>> {
    match info.version {
        ModelVersion::V4 => bail!("v4 does not support init state yet"),
        ModelVersion::V5 => Ok(v5::read_state(context, info, model).await?),
        ModelVersion::V6 => Ok(v6::read_state(context, info, model).await?),
        ModelVersion::V7 => Ok(v7::read_state(context, info, model).await?),
    }
}

async fn load_runtime(
    context: &Context,
    info: &ModelInfo,
    request: &ReloadRequest,
    load: LoadType,
) -> Result<(
    Vec<InitState>,
    Arc<dyn Runtime + Send + Sync>,
    Arc<dyn State + Send + Sync>,
    Arc<dyn ModelSerialize + Send + Sync>,
)> {
    let ReloadRequest {
        model_path,
        lora,
        state,
        quant,
        quant_type,
        precision,
        max_batch,
        embed_device,
        ..
    } = request.clone();

    let mut states = Vec::with_capacity(state.len());
    for state in state.into_iter() {
        let reload::State {
            path,
            name,
            id,
            default,
        } = state;
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
        match load_model_state(context, info, model).await {
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

    let file = File::open(model_path).await?;
    let data = unsafe { Mmap::map(&file) }?;

    match load {
        LoadType::SafeTensors => {
            let model = SafeTensors::deserialize(&data)?;
            if let Ok(data) = load_model_state(context, info, model).await {
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
            let lora: Vec<Result<_>> = join_all(lora.iter().map(|lora| async move {
                let reload::Lora { path, alpha } = lora;
                let file = File::open(path).await?;
                let data = unsafe { Mmap::map(&file)? };
                let blend = LoraBlend::full(*alpha);
                Ok((data, blend))
            }))
            .await;
            let lora: Vec<_> = lora.into_iter().try_collect()?;
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
            let builder = lora.into_iter().fold(builder, |builder, x| builder.lora(x));

            macro_rules! match_safe_tensors {
                (($v:expr, $p:expr), { $(($version:path, $precision:path, $model:ty, $build:ident, $bundle:ty)),+ }) => {
                    match ($v, $p) {
                        $(
                            ($version, $precision) => {
                                let model = builder.$build().await?;
                                let bundle = <$bundle>::new(model, max_batch);
                                let state = Arc::new(bundle.state());
                                let model = Arc::new(Model(bundle.model()));
                                let runtime = Arc::new(TokioRuntime::<InferInput, InferOutput>::new(bundle).await);
                                Ok((states, runtime, state, model))
                            }
                        )+
                    }
                }
            }
            match_safe_tensors!(
                (info.version, precision),
                {
                    (ModelVersion::V4, Precision::Fp16, v4::Model, build_v4, v4::Bundle::<f16>),
                    (ModelVersion::V5, Precision::Fp16, v5::Model, build_v5, v5::Bundle::<f16>),
                    (ModelVersion::V6, Precision::Fp16, v6::Model, build_v6, v6::Bundle::<f16>),
                    (ModelVersion::V7, Precision::Fp16, v7::Model, build_v7, v7::Bundle::<f16>),
                    (ModelVersion::V4, Precision::Fp32, v4::Model, build_v4, v4::Bundle::<f32>),
                    (ModelVersion::V5, Precision::Fp32, v5::Model, build_v5, v5::Bundle::<f32>),
                    (ModelVersion::V6, Precision::Fp32, v6::Model, build_v6, v6::Bundle::<f32>),
                    (ModelVersion::V7, Precision::Fp32, v7::Model, build_v7, v7::Bundle::<f32>)
                }
            )
        }
        LoadType::Prefab => {
            use cbor4ii::{core::utils::SliceReader, serde::Deserializer};

            let reader = SliceReader::new(&data);
            let mut deserializer = Deserializer::new(reader);

            macro_rules! match_prefab {
                (($v:expr, $p:expr), { $(($version:path, $precision:path, $model:ty, $bundle:ty)),+ }) => {
                    match ($v, $p) {
                        $(
                            ($version, $precision) => {
                                let seed: Seed<_, $model> = Seed::new(context);
                                let model = seed.deserialize(&mut deserializer)?;
                                let bundle = <$bundle>::new(model, max_batch);
                                let state = Arc::new(bundle.state());
                                let model = Arc::new(Model(bundle.model()));
                                let runtime = Arc::new(TokioRuntime::<InferInput, InferOutput>::new(bundle).await);
                                Ok((states, runtime, state, model))
                            }
                        )+
                    }
                }
            }
            match_prefab!(
                (info.version, precision),
                {
                    (ModelVersion::V4, Precision::Fp16, v4::Model, v4::Bundle::<f16>),
                    (ModelVersion::V5, Precision::Fp16, v5::Model, v5::Bundle::<f16>),
                    (ModelVersion::V6, Precision::Fp16, v6::Model, v6::Bundle::<f16>),
                    (ModelVersion::V7, Precision::Fp16, v7::Model, v7::Bundle::<f16>),
                    (ModelVersion::V4, Precision::Fp32, v4::Model, v4::Bundle::<f32>),
                    (ModelVersion::V5, Precision::Fp32, v5::Model, v5::Bundle::<f32>),
                    (ModelVersion::V6, Precision::Fp32, v6::Model, v6::Bundle::<f32>),
                    (ModelVersion::V7, Precision::Fp32, v7::Model, v7::Bundle::<f32>)
                }
            )
        }
    }
}

async fn process(env: Arc<RwLock<Environment>>, request: ThreadRequest) -> Result<()> {
    match request {
        ThreadRequest::Adapter(sender) => {
            let _ = sender.send(list_adapters());
        }
        ThreadRequest::Info(sender) => {
            let env = env.read().await;
            if let Environment::Loaded { info, .. } = &*env {
                let _ = sender.send(info.clone());
            }
        }
        ThreadRequest::Generate {
            request,
            tokenizer,
            sender,
        } => {
            let context = GenerateContext::new(*request, sender, &tokenizer).await?;
            let env = env.read().await;
            if let Environment::Loaded { sender, .. } = &*env {
                let _ = sender.send(context);
            }
        }
        ThreadRequest::Reload { request, sender } => {
            let handle = tokio::spawn(async move {
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

                let (states, runtime, state, model) =
                    load_runtime(&context, &info, &request, load).await?;

                let reload = Arc::new(*request);
                let info = RuntimeInfo {
                    reload,
                    info,
                    states,
                    tokenizer,
                };

                let sender = {
                    let runtime = Arc::downgrade(&runtime);
                    let (sender, receiver) = flume::unbounded();
                    tokio::spawn(crate::run::run(
                        context,
                        runtime,
                        state,
                        receiver,
                        info.clone(),
                    ));
                    sender
                };

                log::info!("model loaded");

                let _ = std::mem::replace(
                    &mut *env,
                    Environment::Loaded {
                        info,
                        runtime,
                        model,
                        sender,
                    },
                );
                Ok(())
            });

            if let Some(sender) = sender {
                let _ = match handle.await? {
                    Ok(_) => sender.send(true),
                    Err(err) => {
                        log::error!("[reload] error: {err:#?}");
                        sender.send(false)
                    }
                };
            }
        }
        ThreadRequest::Unload => {
            let mut env = env.write().await;
            let _ = std::mem::take(&mut *env);
            log::info!("model unloaded");
        }
        ThreadRequest::Save { request, sender } => {
            let env = env.read().await;
            if let Environment::Loaded { model, .. } = &*env {
                log::info!("serializing model into {:?}", &request.path);
                let model = model.clone();
                let handle = tokio::task::spawn_blocking(move || {
                    let file = std::fs::File::create(request.path)?;
                    model.serialize(file)
                });
                drop(env);

                let _ = match handle.await? {
                    Ok(_) => sender.send(true),
                    Err(err) => {
                        log::error!("[save] error: {err:#?}");
                        sender.send(false)
                    }
                };
            }
        }
    };
    Ok(())
}

pub async fn serve(receiver: Receiver<ThreadRequest>) {
    let env: Arc<RwLock<Environment>> = Default::default();
    while let Ok(request) = receiver.recv_async().await {
        let future = process(env.clone(), request);
        tokio::spawn(future);
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
