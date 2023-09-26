use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Cursor, Read},
    net::{Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use clap::{Parser, ValueEnum};
use dialoguer::{theme::ColorfulTheme, Select};
use flume::Receiver;
use memmap::Mmap;
use serde::{Deserialize, Serialize};
use tower_http::{cors::CorsLayer, services::ServeDir};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{LayerFlags, Model, ModelBuilder, ModelState, Quantization},
    tokenizer::Tokenizer,
    wgpu::PowerPreference,
};

use crate::{
    run::{GenerateContext, Runtime, SlotResult, Tokens},
    sampler::Sampler,
};

mod chat;
mod completion;
mod embedding;
mod models;
mod run;
mod sampler;

pub const MAX_TOKENS: usize = 4096;
pub const MAX_PENALTY_COUNT: usize = 1024;
pub const STATE_CACHE_LRU: usize = 16;

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
pub enum OptionArray<T> {
    #[default]
    None,
    Item(T),
    Array(Vec<T>),
}

impl<T> From<OptionArray<T>> for Vec<T>
where
    T: std::fmt::Debug + Clone + Serialize,
{
    fn from(value: OptionArray<T>) -> Self {
        match value {
            OptionArray::None => vec![],
            OptionArray::Item(item) => vec![item],
            OptionArray::Array(vec) => vec,
        }
    }
}

pub enum ThreadRequest {
    Reload(ReloadRequest),
    Generate {
        request: GenerateRequest,
        token_sender: flume::Sender<Token>,
    },
}

#[derive(Debug, Default)]
pub struct GenerateRequest {
    pub prompt: String,
    pub model_text: String,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub sampler: Sampler,
    pub logit_bias: HashMap<u16, f32>,
    pub embed: bool,
}

#[derive(Debug, Deserialize)]
pub struct ReloadRequest {
    /// Path to the model.
    pub path: PathBuf,
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
}

impl Default for ReloadRequest {
    fn default() -> Self {
        Self {
            path: "assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st".into(),
            quant: vec![],
            token_chunk_size: 32,
            head_chunk_size: 8192,
            max_runtime_batch: 8,
            max_batch: 32,
            embed_layer: 2,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct TokenCounter {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone)]
pub struct ThreadState {
    pub sender: flume::Sender<ThreadRequest>,
    pub model_name: Arc<RwLock<String>>,
}

async fn create_context(args: &Args) -> Result<Context> {
    let instance = Instance::new();
    let adapter = match args.adapter {
        AdapterOption::Auto => instance.adapter(PowerPreference::HighPerformance).await,
        AdapterOption::Manual => {
            let adapters = instance.adapters();
            let selection = args.adapter_id.unwrap_or_else(|| {
                Select::with_theme(&ColorfulTheme::default())
                    .with_prompt("Please select an adapter")
                    .default(0)
                    .items(&adapters)
                    .interact()
                    .expect("adapter selection")
            });
            instance.select_adapter(selection)
        }
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

fn load_model<'a>(context: &Context, request: ReloadRequest) -> Result<Model<'a>> {
    let ReloadRequest {
        path,
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

    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };

    ModelBuilder::new(context, &map)
        .with_quant(quant)
        .with_token_chunk_size(token_chunk_size)
        .with_head_chunk_size(head_chunk_size)
        .build()
}

fn load_web(path: impl AsRef<Path>, target: &Path) -> Result<()> {
    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };
    zip_extract::extract(Cursor::new(&map), target, false)?;
    Ok(())
}

fn reload_request_from_config(path: impl AsRef<Path>) -> Result<ReloadRequest> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(toml::from_str(&contents)?)
}

// fn _monitor(context: &Context, tokenizer: Tokenizer, receiver: Receiver<ThreadRequest>) {
//     loop {
//         let request = match receiver.recv() {
//             Ok(ThreadRequest::Reload(request)) => request,
//             _ => continue,
//         };

//         let quantization = if request.quantized_layers.is_empty() {
//             Quantization::None
//         } else {
//             let mut layers = LayerFlags::empty();
//             request
//                 .quantized_layers
//                 .into_iter()
//                 .for_each(|layer| layers |= LayerFlags::from_layer(layer as u64));
//             Quantization::Int8(layers)
//         };

//         let model = match load_model(context.clone(), request.model_path, quantization) {
//             Ok(model) => model,
//             Err(err) => {
//                 log::error!("{}", err);
//                 continue;
//             }
//         }
//         .into();
//         let tokenizer = tokenizer.clone();
//         let receiver = receiver.clone();
//         std::thread::spawn(move || model_task(model, tokenizer, receiver));
//     }
// }

// fn model_task(
//     args: Args,
//     context: Context,
//     tokenizer: Tokenizer,
//     path: impl AsRef<Path>,
//     receiver: Receiver<ThreadRequest>,
// ) {
//     let quant = match args.quant {
//         None => Quantization::None,
//         Some(layer) => {
//             let mut layers = LayerFlags::empty();
//             (0..layer).for_each(|layer| layers |= LayerFlags::from_layer(layer as u64));
//             Quantization::Int8(layers)
//         }
//     };

//     let model = match load_model(&context, path, quant, args.head_chunk) {
//         Ok(model) => model,
//         Err(err) => {
//             log::error!("{}", err);
//             return;
//         }
//     };

//     log::info!("{:#?}", context.adapter.get_info());
//     log::info!("{:#?}", model.info());
//     log::info!("chosen head chunk size: {}", args.head_chunk);

//     let penalty_free_tokens = {
//         let mut set = HashSet::new();
//         for token in 0..u16::MAX {
//             let word = tokenizer.decode(&[token]).unwrap_or_default();
//             let word = String::from_utf8(word).unwrap_or_default();
//             if word.contains('\n') {
//                 set.insert(token);
//             }
//         }
//         set
//     };

//     let mut state_cache = Trie::<&[u8], (BackedState, usize)>::new();
//     let state = ModelState::new(&context, model.info(), 1);

//     let mut respond = || -> Result<()> {
//         let (request, mut occurrences, token_sender) = match receiver.recv() {
//             Ok(ThreadRequest::Generate {
//                 request,
//                 occurrences,
//                 token_sender,
//             }) => (request, occurrences, token_sender),
//             Ok(ThreadRequest::Reload(_)) => return Ok(()),
//             Err(_) => return Ok(()),
//         };

//         let GenerateRequest {
//             prompt,
//             max_tokens,
//             stop,
//             sampler,
//             logit_bias,
//             embedding,
//         } = request;

//         log::trace!("{:#?}", sampler);

//         let remain = {
//             let prefix = state_cache
//                 .longest_common_prefix(prompt.as_bytes())
//                 .to_vec();
//             let mut remain = prompt.as_bytes().to_vec();
//             if let Some(count) = state_cache
//                 .get_mut(&prefix[..])
//                 .and_then(|(backed, count)| state.load(backed).ok().and(Some(count)))
//             {
//                 log::trace!("state cache hit: {count}");
//                 *count = 0;
//                 remain.split_off(prefix.len())
//             } else {
//                 log::trace!("state cache miss");
//                 remain
//             }
//         };
//         let mut model_text = String::new();
//         let mut token_counter = TokenCounter::default();

//         let mut tokens = tokenizer.encode(&remain).unwrap_or_default();
//         let _ = token_sender.send(Token::Start);

//         'run: {
//             token_counter.prompt_tokens = tokens.len();
//             token_counter.total_tokens = tokens.len();

//             log::trace!("{}", String::from_utf8(remain).unwrap_or_default());

//             for _ in 0..max_tokens {
//                 if token_sender.is_disconnected() {
//                     break 'run;
//                 }

//                 let shape = Shape::new(1, tokens.len(), 1);
//                 let mut logits = model
//                     .run(&context.tensor_from_data(shape, tokens)?, &mask, &state)?
//                     .to_vec();
//                 for (&token, &count) in occurrences
//                     .iter()
//                     .filter(|(token, _)| !penalty_free_tokens.contains(token))
//                 {
//                     let penalty =
//                         sampler.presence_penalty + sampler.frequency_penalty * count as f32;
//                     logits[token as usize] -= penalty;
//                 }
//                 for (&token, &bias) in &logit_bias {
//                     logits[token as usize] += bias;
//                 }

//                 let probs = model
//                     .softmax(&context.tensor_from_data(model.head_shape(1), logits)?)?
//                     .to_vec();
//                 let token = sampler.sample(probs);
//                 let word = tokenizer
//                     .decode(&[token])
//                     .ok()
//                     .and_then(|x| String::from_utf8(x).ok())
//                     .unwrap_or_default();

//                 print!("{word}");

//                 model_text += &word;
//                 token_counter.completion_tokens += 1;
//                 token_counter.total_tokens += 1;

//                 let count = occurrences.get(&token).copied().unwrap_or_default();
//                 occurrences.insert(token, count + 1);

//                 let _ = token_sender.send(Token::Token(word));
//                 tokens = vec![token];

//                 if token == 0 || stop.iter().any(|x| model_text.contains(x)) {
//                     let _ = token_sender.send(Token::Stop(FinishReason::Stop, token_counter));
//                     break 'run;
//                 }
//             }

//             let _ = token_sender.send(Token::Stop(FinishReason::Length, token_counter));
//         }

//         println!("[DONE]");

//         let back = BackedState::from(state.clone());
//         if embedding {
//             let num_layer = model.info().num_layers;
//             // let embedding = back.0[num_layer - 3][4 * num_emb..].to_vec();
//             let embedding = back.as_slice((.., 5 * (num_layer - 3) + 4, ..))?.to_vec();
//             let _ = token_sender.send(Token::Embed(embedding));
//         }

//         let key = (prompt + &model_text).as_bytes().to_vec();
//         state_cache.insert(key.leak(), (back, 0));

//         let mut keys_to_remove = vec![];
//         for (&key, (_, count)) in state_cache.iter_mut() {
//             *count += 1;
//             if *count > STATE_CACHE_LRU {
//                 keys_to_remove.push(key);
//             }
//         }

//         log::trace!("state cache evicted: {}", keys_to_remove.len());
//         for key in keys_to_remove {
//             state_cache.remove(key);
//         }

//         let _ = token_sender.send(Token::Done);
//         Ok(())
//     };

//     loop {
//         match respond() {
//             Ok(()) => {}
//             Err(err) => log::error!("{}", err),
//         }
//     }
// }

fn model_route(
    context: Context,
    tokenizer: Tokenizer,
    receiver: Receiver<ThreadRequest>,
) -> Result<()> {
    let runtime: Arc<Mutex<Option<Runtime>>> = Default::default();
    let mut pending = Vec::new();

    let sender = {
        let (sender, receiver) = flume::unbounded();
        let runtime = runtime.clone();
        let tokenizer = tokenizer.clone();
        std::thread::spawn(move || run::run(runtime, tokenizer, receiver));
        sender
    };

    let queue = |context| -> Vec<GenerateContext> {
        let mut pending = Vec::new();
        let mut runtime = runtime.lock().unwrap();
        match &mut *runtime {
            Some(runtime) => match runtime.queue(context) {
                SlotResult::Success(batch) => log::info!("queued task at {batch}"),
                SlotResult::Fault(batch) => log::info!("swapped task at {batch}"),
                SlotResult::Failure(context) => pending.push(*context),
            },
            None => pending.push(context),
        }
        let _ = sender.send(());
        pending
    };

    let listen = |pending: &mut Vec<GenerateContext>| -> Result<()> {
        match receiver.recv()? {
            ThreadRequest::Reload(request) => {
                let max_runtime_batch = request.max_runtime_batch;
                let max_batch = request.max_batch;
                let embed_layer = request.embed_layer;

                let model = Arc::new(load_model(&context, request)?);
                let info = model.info();
                let state = ModelState::new(&context, info, max_batch);
                let mut runtime = runtime.lock().unwrap();
                let _ = runtime.replace(Runtime::new(model, state, max_runtime_batch, embed_layer));
            }
            ThreadRequest::Generate {
                request,
                token_sender,
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
                pending.append(&mut queue(context));
            }
        };
        Ok(())
    };

    loop {
        if let Err(err) = listen(&mut pending) {
            log::error!("{err}");
        }

        while !pending.is_empty() {
            let mut temp = Vec::new();
            for context in pending.drain(..) {
                temp.append(&mut queue(context));
            }
            std::mem::swap(&mut pending, &mut temp);
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short, default_value_t = AdapterOption::Manual)]
    #[clap(value_enum)]
    adapter: AdapterOption,
    #[arg(long)]
    adapter_id: Option<usize>,
    #[arg(long, short, value_name = "FILE")]
    tokenizer: Option<PathBuf>,
    #[arg(long, short, value_name = "FILE")]
    model: Option<PathBuf>,
    #[arg(long, short)]
    ip: Option<Ipv4Addr>,
    #[arg(long, short, default_value_t = 65530)]
    port: u16,
}

#[derive(Debug, Default, Clone, Copy, ValueEnum)]
enum AdapterOption {
    Auto,
    #[default]
    Manual,
}

#[tokio::main]
async fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ai00_server", log::LevelFilter::Trace)
        .init()
        .unwrap();

    let args = Args::parse();
    // let model_name = model_path
    //     .file_name()
    //     .and_then(OsStr::to_str)
    //     .map(String::from_str)
    //     .and_then(Result::ok)
    //     .map(|name| name.replace(".st", ""))
    //     .unwrap();

    let tokenizer_path = args
        .tokenizer
        .clone()
        .unwrap_or("assets/rwkv_vocab_v20230424.json".into());

    let context = create_context(&args).await.unwrap();
    let tokenizer = load_tokenizer(&tokenizer_path).unwrap();
    log::info!("{:#?}", context.adapter.get_info());

    let (sender, receiver) = flume::unbounded::<ThreadRequest>();

    {
        let path = args.model.clone().unwrap_or("Config.toml".into());
        log::info!("loading model config {}...", path.to_string_lossy());

        let request = reload_request_from_config(path).expect("load model config failed");
        let _ = sender.send(ThreadRequest::Reload(request));
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let temp_path = temp_dir.into_path();
    load_web("assets/www.zip", &temp_path).unwrap();

    let app = Router::new()
        .route("/load", post(models::load))
        .route("/models", get(models::models))
        .route("/v1/models", get(models::models))
        .route("/completions", post(completion::completions))
        .route("/v1/completions", post(completion::completions))
        .route("/chat/completions", post(chat::chat_completions))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/embeddings", post(embedding::embeddings))
        .route("/v1/embeddings", post(embedding::embeddings))
        .fallback_service(ServeDir::new(temp_path.join("www")))
        .layer(CorsLayer::permissive())
        .with_state(ThreadState {
            sender,
            model_name: Default::default(),
        });

    let addr = SocketAddr::from((args.ip.unwrap_or(Ipv4Addr::new(0, 0, 0, 0)), args.port));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    log::info!("server started at http://{addr}");

    std::thread::spawn(move || model_route(context, tokenizer, receiver));
    axum::serve(listener, app).await.unwrap();
}
