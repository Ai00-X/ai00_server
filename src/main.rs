use std::{
    collections::HashMap,
    convert::Infallible,
    fs::{self, File},
    io::{BufReader, Cursor, Read},
    net::{Ipv4Addr, SocketAddr},
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
use memmap::Mmap;
use run::RuntimeUntyped;
use serde::{Deserialize, Serialize};
use tower_http::{cors::CorsLayer, services::ServeDir};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{
        loader::Loader, FromBuilder, LayerFlags, Model, ModelBuilder, ModelInfo, ModelState,
        ModelVersion, Quantization, StateBuilder,
    },
    tokenizer::Tokenizer,
    wgpu::PowerPreference,
};

use crate::{
    run::{GenerateContext, Runtime, SlotResult, Tokens},
    sampler::Sampler,
};

mod api;
mod chat;
mod completion;
mod config;
mod embedding;
mod run;
mod sampler;

pub const MAX_TOKENS: usize = 4096;
pub const STATE_CHUNK_SIZE: usize = 4;

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
    Info(Sender<RuntimeInfo>),
    Generate {
        request: GenerateRequest,
        tokenizer: Arc<Tokenizer>,
        sender: Sender<Token>,
    },
    Reload(ReloadRequest),
}

#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    pub reload: ReloadRequest,
    pub model: ModelInfo,
    pub tokenizer: Arc<Tokenizer>,
}

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReloadRequest {
    /// Path to the model.
    pub model_path: PathBuf,
    /// Specify layers that needs to be quantized.
    pub quant: Vec<usize>,
    /// Maximum tokens to be processed in parallel at once.
    pub token_chunk_size: usize,
    /// The chunk size for each split of the head matrix.
    pub head_chunk_size: usize,
    /// Maximum number of batches that are active at once.
    pub max_runtime_batch: usize,
    /// Number of states that are cached on GPU.
    pub max_batch: usize,
    /// the (reversed) number of layer at which the output is as embedding.
    pub embed_layer: usize,
    /// Path to the tokenizer.
    pub tokenizer_path: PathBuf,
    /// Adapter selection.
    pub adapter: AdapterOption,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct TokenCounter {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone)]
pub struct ThreadState(pub Sender<ThreadRequest>);

async fn create_context(adapter: AdapterOption) -> Result<Context> {
    let instance = Instance::new();
    let adapter = match adapter {
        AdapterOption::Auto => instance.adapter(PowerPreference::HighPerformance).await,
        AdapterOption::Economical => instance.adapter(PowerPreference::LowPower).await,
        AdapterOption::Manual(selection) => instance.select_adapter(selection),
    }?;

    let context = ContextBuilder::new(adapter)
        .with_default_pipelines()
        .with_quant_pipelines()
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

fn load_model<'a, M, S>(context: &Context, request: ReloadRequest, data: &'a [u8]) -> Result<(M, S)>
where
    S: ModelState + FromBuilder<Builder<'a> = StateBuilder, Error = Infallible>,
    M: Model<ModelState = S> + FromBuilder<Builder<'a> = ModelBuilder<'a>, Error = anyhow::Error>,
{
    let ReloadRequest {
        quant,
        token_chunk_size,
        head_chunk_size,
        ..
    } = request;
    let quant = if quant.is_empty() {
        Quantization::None
    } else {
        let mut layers = LayerFlags::empty();
        quant
            .into_iter()
            .for_each(|x| layers.insert(LayerFlags::from_layer(x as u64)));
        Quantization::Int8(layers)
    };

    let model: M = ModelBuilder::new(context, data)
        .with_quant(quant)
        .with_token_chunk_size(token_chunk_size)
        .with_head_chunk_size(head_chunk_size)
        .build()?;
    let state: S = StateBuilder::new(context, model.info())
        .with_max_batch(request.max_batch)
        .with_chunk_size(STATE_CHUNK_SIZE)
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

fn model_route(receiver: Receiver<ThreadRequest>) -> Result<()> {
    let mut env: Option<(Arc<RuntimeUntyped<'_>>, ReloadRequest)> = None;
    let mut queue = Vec::new();

    let sender = {
        let (sender, receiver) = flume::unbounded();
        std::thread::spawn(move || run::run(receiver));
        sender
    };

    fn enqueue(
        queue: &mut Vec<GenerateContext>,
        environment: &Option<(Arc<RuntimeUntyped<'_>>, ReloadRequest)>,
        context: GenerateContext,
    ) {
        match environment {
            Some((runtime, _)) => match runtime.queue(context) {
                SlotResult::Success(batch) => log::info!("queued task at {batch}"),
                SlotResult::Fault(batch) => log::info!("swapped task at {batch}"),
                SlotResult::Failure(context) => queue.push(*context),
            },
            None => queue.push(context),
        }
    }

    loop {
        let mut listen = || -> Result<()> {
            match receiver.recv()? {
                ThreadRequest::Info(sender) => {
                    if let Some((runtime, reload)) = &env {
                        let reload = reload.clone();
                        let model = runtime.model_info().clone();
                        let tokenizer = runtime.tokenizer();
                        let _ = sender.send(RuntimeInfo {
                            reload,
                            model,
                            tokenizer,
                        });
                    }
                }
                ThreadRequest::Reload(request) => {
                    let max_runtime_batch = request.max_runtime_batch;
                    let embed_layer = request.embed_layer;

                    let context = pollster::block_on(create_context(request.adapter))?;
                    let tokenizer = load_tokenizer(&request.tokenizer_path)?;
                    log::info!("{:#?}", context.adapter.get_info());

                    let file = File::open(&request.model_path)?;
                    let data = unsafe { Mmap::map(&file)? };
                    let info = Loader::info(&data)?;
                    log::info!("{:#?}", info);

                    let runtime = Arc::new(match info.version {
                        ModelVersion::V4 => {
                            let (model, state) = load_model(&context, request.clone(), &data)?;
                            RuntimeUntyped::V4(Runtime::new(
                                tokenizer,
                                model,
                                state,
                                max_runtime_batch,
                                embed_layer,
                            ))
                        }
                        ModelVersion::V5 => {
                            let (model, state) = load_model(&context, request.clone(), &data)?;
                            RuntimeUntyped::V5(Runtime::new(
                                tokenizer,
                                model,
                                state,
                                max_runtime_batch,
                                embed_layer,
                            ))
                        }
                    });
                    let _ = sender.send(runtime.clone());
                    env.replace((runtime, request));
                }
                ThreadRequest::Generate {
                    request,
                    tokenizer,
                    sender: token_sender,
                } => {
                    let GenerateRequest {
                        prompt,
                        model_text,
                        max_tokens,
                        stop,
                        sampler,
                        logit_bias,
                        embed,
                    } = request;

                    let tokens = Tokens(tokenizer.encode(prompt.as_bytes())?);
                    let model_tokens = Tokens(tokenizer.encode(model_text.as_bytes())?);
                    let mut penalties = HashMap::new();
                    for (index, token) in model_tokens.iter().rev().enumerate() {
                        let ap = sampler.presence_penalty;
                        let af = sampler.frequency_penalty;
                        let ad = sampler.penalty_decay;
                        let mut penalty = penalties.remove(token).unwrap_or(ap);
                        penalty += af * ad.powf(index as f32);
                        penalties.insert(*token, penalty);
                    }

                    let context = GenerateContext {
                        prompt_tokens: tokens.to_vec(),
                        prefix: Default::default(),
                        suffix: tokens,
                        penalties,
                        model_text: Default::default(),
                        output_buffer: Default::default(),
                        model_tokens: Default::default(),
                        max_tokens,
                        stop,
                        sampler,
                        logit_bias,
                        embed,
                        sender: token_sender,
                    };
                    enqueue(&mut queue, &env, context);
                    if let Some((runtime, _)) = &env {
                        let _ = sender.send(runtime.clone());
                    }
                }
            };
            Ok(())
        };

        if let Err(err) = listen() {
            log::error!("{err}");
        }

        while !queue.is_empty() {
            let mut temp = Vec::new();
            for context in queue.drain(..) {
                enqueue(&mut temp, &env, context);
                let _ = env.iter().map(|(runtime, _)| sender.send(runtime.clone()));
            }
            std::mem::swap(&mut queue, &mut temp);
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

pub fn request_info(sender: Sender<ThreadRequest>) -> RuntimeInfo {
    let (info_sender, info_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Info(info_sender));
    info_receiver.recv().expect("request info failed")
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short, value_name = "FILE")]
    config: Option<PathBuf>,
    #[arg(long, short)]
    ip: Option<Ipv4Addr>,
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

    {
        let path = args
            .config
            .clone()
            .unwrap_or("assets/configs/Config.toml".into());
        log::info!("reading config {}...", path.to_string_lossy());

        let request = load_config(path).expect("load config failed").into();
        let _ = sender.send(ThreadRequest::Reload(request));
    }

    let serve_path = {
        let path = tempfile::tempdir()
            .expect("create temp dir failed")
            .into_path();
        load_web("assets/www/index.zip", &path).expect("load frontend failed");
        path
    };

    // extract and load all plugins under `assets/www/plugins`.
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
        .route("/api/unzip", post(api::unzip))
        .route("/api/load", post(api::load))
        .route("/api/files", post(api::files))
        .route("/api/models/info", get(api::info))
        .route("/api/models", get(api::models))
        .route("/api/v1/models", get(api::models))
        .route("/api/completions", post(completion::completions))
        .route("/api/v1/completions", post(completion::completions))
        .route("/api/chat/completions", post(chat::chat_completions))
        .route("/api/v1/chat/completions", post(chat::chat_completions))
        .route("/api/embeddings", post(embedding::embeddings))
        .route("/api/v1/embeddings", post(embedding::embeddings))
        .fallback_service(ServeDir::new(serve_path))
        .layer(CorsLayer::permissive())
        .with_state(ThreadState(sender));

    let addr = SocketAddr::from((args.ip.unwrap_or(Ipv4Addr::new(0, 0, 0, 0)), args.port));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    log::info!("server started at http://{addr}");

    std::thread::spawn(move || model_route(receiver));
    axum::serve(listener, app).await.unwrap();
}
