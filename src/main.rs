use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, Cursor, Read},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use config::{AdapterOption, Config};
use flume::{Receiver, Sender};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tower_http::{cors::CorsLayer, services::ServeDir};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{
        loader::{Loader, Lora, LoraBlend},
        v4, v5, v6, EmbedDevice, Model, ModelBuilder, ModelInfo, ModelState, ModelVersion, Quant,
        StateBuilder,
    },
    tokenizer::Tokenizer,
    wgpu::{Backends, PowerPreference},
};

use crate::{
    run::{GenerateContext, Runner, Runtime, SlotResult, Tokens},
    sampler::Sampler,
};

mod api;
mod config;
mod oai;
mod run;
mod sampler;
mod utils;

pub const MAX_TOKENS: usize = 4096;

#[derive(Debug)]
pub enum Token {
    Start,
    Token(String),
    Stop(FinishReason, TokenCounter),
    Embed(Vec<f32>),
    Done,
}

#[derive(Debug, Default, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// API returned complete model output.
    Stop,
    /// Incomplete model output due to max_tokens parameter or token limit.
    Length,
    /// Omitted content due to a flag from our content filters.
    ContentFilter,
    /// API response still in progress or incomplete.
    #[default]
    Null,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Array<T> {
    #[default]
    None,
    Item(T),
    Vec(Vec<T>),
}

impl<T> From<Array<T>> for Vec<T>
where
    T: std::fmt::Debug + Clone + Serialize,
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
    Adapter(Sender<AdapterList>),
    Info(Sender<RuntimeInfo>),
    Generate {
        request: GenerateRequest,
        tokenizer: Arc<Tokenizer>,
        sender: Sender<Token>,
    },
    Reload {
        request: ReloadRequest,
        sender: Option<Sender<bool>>,
    },
    Unload,
}

#[derive(Default)]
pub enum Environment {
    Loaded {
        runtime: Box<dyn Runner + Send + Sync>,
        reload: ReloadRequest,
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
                SlotResult::Error => log::warn!("empty task is not queued"),
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

#[derive(Debug, Default, Clone)]
pub struct GenerateRequest {
    /// The prompt for the model.
    pub prompt: String,
    /// All text the model output earlier.
    pub model_text: String,
    /// Output token limit.
    pub max_tokens: usize,
    /// Stop indicators.
    pub stop: Vec<String>,
    /// Sampler parameters.
    pub sampler: Sampler,
    /// Bias added to tokens before sampling.
    pub logit_bias: HashMap<u16, f32>,
    /// Whether this is an embedding request.
    pub embed: bool,
    /// The (reversed) number of layer at which the output is as embedding.
    pub embed_layer: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReloadRequest {
    /// Path to the model.
    pub model_path: PathBuf,
    /// List of LoRA blended on the model.
    pub lora: Vec<config::Lora>,
    /// Specify layers that needs to be quantized.
    pub quant: usize,
    /// Quantization type (Int8 or NF4).
    pub quant_type: Quant,
    /// Whether to use alternative GEMM kernel to speed-up long prompts.
    pub turbo: bool,
    /// Maximum tokens to be processed in parallel at once.
    pub token_chunk_size: usize,
    /// The chunk size for each split of the head matrix.
    pub head_chunk_size: usize,
    /// The chunk size of layers in model state.
    pub state_chunk_size: usize,
    /// Maximum number of batches that are active at once.
    pub max_runtime_batch: usize,
    /// Number of states that are cached on GPU.
    pub max_batch: usize,
    /// Device to put the embed tensor.
    pub embed_device: EmbedDevice,
    /// Path to the tokenizer.
    pub tokenizer_path: PathBuf,
    /// Adapter selection.
    pub adapter: AdapterOption,
}

impl Default for ReloadRequest {
    fn default() -> Self {
        config::Config::default().into()
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct TokenCounter {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone)]
pub struct ThreadState(pub Sender<ThreadRequest>);

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

async fn load_model<M, S>(context: &Context, request: ReloadRequest, data: &[u8]) -> Result<(M, S)>
where
    S: ModelState,
    M: Model<State = S>,
{
    let ReloadRequest {
        quant,
        quant_type,
        lora,
        token_chunk_size,
        turbo,
        embed_device,
        ..
    } = request;
    let quant = (0..quant).map(|layer| (layer, quant_type)).collect();

    let lora: Vec<_> = lora
        .into_iter()
        .map(|lora| -> Result<_> {
            let file = File::open(&lora.path)?;
            let data = unsafe { Mmap::map(&file) }?;
            Ok((data, lora.alpha))
        })
        .try_collect()?;
    let lora: Vec<_> = lora
        .iter()
        .map(|(data, alpha)| -> Result<_> {
            let data = SafeTensors::deserialize(data)?;
            Ok((data, *alpha))
        })
        .try_collect()?;
    let lora = lora
        .iter()
        .map(|(data, alpha)| {
            let blend = LoraBlend::full(*alpha);
            Lora { data, blend }
        })
        .collect_vec();

    let model = SafeTensors::deserialize(data)?;
    let model = ModelBuilder::new(context, &model)
        .with_quant(quant)
        .with_turbo(turbo)
        .with_embed_device(embed_device)
        .with_token_chunk_size(token_chunk_size);
    let model: M = lora
        .into_iter()
        .fold(model, |acc, x| acc.add_lora(x))
        .build()
        .await?;

    let state: S = StateBuilder::new(context, model.info())
        .with_num_batch(request.max_batch)
        .with_chunk_size(request.state_chunk_size)
        .build();
    Ok((model, state))
}

fn load_web(path: impl AsRef<Path>, target: &Path) -> Result<()> {
    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };
    zip_extract::extract(Cursor::new(&map), target, false)?;
    Ok(())
}

fn load_plugin(path: impl AsRef<Path>, target: &Path, name: &String) -> Result<()> {
    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };
    let root = target.join("plugins");
    if !root.exists() {
        fs::create_dir(&root)?;
    }
    let dir = root.join(name);
    fs::create_dir(&dir)?;
    zip_extract::extract(Cursor::new(&map), &dir, false)?;
    Ok(())
}

fn load_config(path: impl AsRef<Path>) -> Result<Config> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(toml::from_str(&contents)?)
}

#[tokio::main]
async fn model_route(receiver: Receiver<ThreadRequest>) -> Result<()> {
    let env: Arc<RwLock<Environment>> = Default::default();
    let queue: Arc<Mutex<Vec<GenerateContext>>> = Default::default();

    let sender = {
        let (sender, receiver) = flume::unbounded();
        let env = env.clone();
        tokio::task::spawn_blocking(move || run::run(receiver, env));
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
                        if let Environment::Loaded { runtime, reload } = env {
                            let reload = reload.clone();
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
                    let callback = move |result: bool| {
                        if let Some(sender) = reload_sender {
                            let _ = sender.send(result);
                        }
                    };
                    let sender = sender.clone();
                    let env = env.clone();
                    let reload = async move {
                        let sender = sender.clone();
                        let max_runtime_batch = request.max_runtime_batch;
                        let state_chunk_size = request.state_chunk_size;

                        let file = File::open(&request.model_path)?;
                        let data = unsafe { Mmap::map(&file)? };
                        let model = SafeTensors::deserialize(&data)?;
                        let info = Loader::info(&model).await?;
                        log::info!("{:#?}", info);

                        let context = create_context(request.adapter, &info).await?;
                        let tokenizer = load_tokenizer(&request.tokenizer_path)?;
                        log::info!("{:#?}", context.adapter.get_info());

                        let mut env = env.write().await;
                        *env = Environment::None;

                        let runtime: Box<dyn Runner + Send + Sync> = match info.version {
                            ModelVersion::V4 => {
                                let (model, state) =
                                    load_model(&context, request.clone(), &data).await?;
                                Box::new(Runtime::<v4::Model<f16>, _, _>::new(
                                    tokenizer,
                                    model,
                                    state,
                                    max_runtime_batch,
                                    state_chunk_size,
                                ))
                            }
                            ModelVersion::V5 => {
                                let (model, state) =
                                    load_model(&context, request.clone(), &data).await?;
                                Box::new(Runtime::<v5::Model<f16>, _, _>::new(
                                    tokenizer,
                                    model,
                                    state,
                                    max_runtime_batch,
                                    state_chunk_size,
                                ))
                            }
                            ModelVersion::V6 => {
                                let (model, state) =
                                    load_model(&context, request.clone(), &data).await?;
                                Box::new(Runtime::<v6::Model<f16>, _, _>::new(
                                    tokenizer,
                                    model,
                                    state,
                                    max_runtime_batch,
                                    state_chunk_size,
                                ))
                            }
                        };
                        let reload = request;
                        *env = Environment::Loaded { runtime, reload };

                        let _ = sender.send(());
                        anyhow::Ok(())
                    };
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
                }
                ThreadRequest::Unload => {
                    let mut env = env.write().await;
                    *env = Environment::None;
                    log::info!("model unloaded");
                }
                ThreadRequest::Generate {
                    request,
                    tokenizer,
                    sender: token_sender,
                } => {
                    let tokens = Tokens(tokenizer.encode(request.prompt.as_bytes())?);
                    let model_tokens = Tokens(tokenizer.encode(request.model_text.as_bytes())?);
                    let mut penalties = HashMap::new();
                    for (index, token) in model_tokens.iter().rev().enumerate() {
                        let ap = request.sampler.presence_penalty;
                        let af = request.sampler.frequency_penalty;
                        let ad = request.sampler.penalty_decay;
                        let mut penalty = penalties.remove(token).unwrap_or(ap);
                        penalty += af * ad.powf(index as f32);
                        penalties.insert(*token, penalty);
                    }

                    let context = GenerateContext {
                        prompt_tokens: tokens.to_vec(),
                        prompt_cached: false,
                        prefix: Default::default(),
                        suffix: tokens,
                        penalties,
                        model_text: Default::default(),
                        buffer: Default::default(),
                        model_tokens: Default::default(),
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
            };
            anyhow::Ok(())
        };

        if let Err(err) = listen.await {
            log::error!("{err}");
        }
    }
}

pub async fn try_request_info(sender: Sender<ThreadRequest>) -> Result<RuntimeInfo> {
    let (info_sender, info_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Info(info_sender));
    let info = info_receiver.recv_async().await?;
    Ok(info)
}

pub async fn request_info(sender: Sender<ThreadRequest>, sleep: Duration) -> RuntimeInfo {
    loop {
        if let Ok(info) = try_request_info(sender.clone()).await {
            break info;
        }
        tokio::time::sleep(sleep).await;
    }
}

pub async fn request_info_stream(
    sender: Sender<ThreadRequest>,
    info_sender: Sender<RuntimeInfo>,
    sleep: Duration,
) {
    loop {
        if let Ok(info) = try_request_info(sender.clone()).await {
            if info_sender.send(info).is_err() {
                break;
            }
        }
        tokio::time::sleep(sleep).await;
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short, value_name = "FILE")]
    config: Option<PathBuf>,
    #[arg(long, short)]
    ip: Option<IpAddr>,
    #[arg(long, short, default_value_t = 65530)]
    port: u16,
}

#[tokio::main]
async fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ai00_server", log::LevelFilter::Trace)
        .init()
        .unwrap();

    let args = Args::parse();
    let (sender, receiver) = flume::unbounded::<ThreadRequest>();

    let request = {
        let path = args
            .config
            .clone()
            .unwrap_or("assets/configs/Config.toml".into());
        log::info!("reading config {}...", path.to_string_lossy());
        load_config(path).expect("load config failed").into()
    };

    tokio::task::spawn_blocking(move || model_route(receiver));
    let _ = sender.send(ThreadRequest::Reload {
        request,
        sender: None,
    });

    let serve_path = {
        let path = tempfile::tempdir()
            .expect("create temp dir failed")
            .into_path();
        load_web("assets/www/index.zip", &path).expect("load frontend failed");
        path
    };

    // create `assets/www/plugins` if it doesn't exist
    if !Path::new("assets/www/plugins").exists() {
        fs::create_dir("assets/www/plugins").expect("create plugins dir failed");
    }

    // extract and load all plugins under `assets/www/plugins`
    match std::fs::read_dir("assets/www/plugins") {
        Ok(dir) => dir
            .filter_map(|x| x.ok())
            .filter(|x| x.path().is_file())
            .filter(|x| x.path().extension().is_some_and(|ext| ext == "zip"))
            .filter(|x| x.path().file_stem().is_some_and(|stem| stem != "api"))
            .for_each(|x| {
                let name = x
                    .path()
                    .file_stem()
                    .expect("this cannot happen")
                    .to_string_lossy()
                    .into();
                match load_plugin(x.path(), &serve_path, &name) {
                    Ok(_) => log::info!("loaded plugin {}", name),
                    Err(err) => log::error!("failed to load plugin {}, {}", name, err),
                }
            }),
        Err(err) => {
            log::error!("failed to read plugin directory: {}", err);
        }
    };

    let app = Router::new()
        .route("/api/adapters", get(api::adapters))
        .route("/api/files/unzip", post(api::unzip))
        .route("/api/files/dir", post(api::dir))
        .route("/api/files/ls", post(api::dir))
        .route("/api/files/config/load", post(api::load_config))
        .route("/api/files/config/save", post(api::save_config))
        .route("/api/models/list", get(api::models))
        .route("/api/models/info", get(api::info))
        .route("/api/models/state", get(api::state))
        .route("/api/models/load", post(api::load))
        .route("/api/models/unload", get(api::unload))
        .route("/api/oai/models", get(oai::models))
        .route("/api/oai/v1/models", get(oai::models))
        .route("/api/oai/completions", post(oai::completions))
        .route("/api/oai/v1/completions", post(oai::completions))
        .route("/api/oai/chat/completions", post(oai::chat_completions))
        .route("/api/oai/v1/chat/completions", post(oai::chat_completions))
        .route("/api/oai/embeddings", post(oai::embeddings))
        .route("/api/oai/v1/embeddings", post(oai::embeddings))
        .fallback_service(ServeDir::new(serve_path))
        .layer(CorsLayer::permissive())
        .with_state(ThreadState(sender));
    let addr = SocketAddr::new(
        args.ip.unwrap_or(IpAddr::from(Ipv4Addr::UNSPECIFIED)),
        args.port,
    );
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    log::info!("server started at {addr}");
    axum::serve(listener, app).await.unwrap();
}
