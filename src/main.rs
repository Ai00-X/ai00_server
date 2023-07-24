use anyhow::Result;
use axum::{routing::post, Router};
use clap::Parser;
use flume::Receiver;
use itertools::Itertools;
use memmap::Mmap;
use qp_trie::Trie;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    ffi::OsStr,
    fs::File,
    io::{BufReader, Read, Write},
    net::SocketAddr,
    path::PathBuf,
    str::FromStr,
};
use tower_http::{cors::CorsLayer, services::ServeDir};
use web_rwkv::{BackedModelState, Environment, Model, Tokenizer};

mod chat;
mod completion;
mod embedding;
mod sampler;

use crate::{
    chat::ChatRequest, completion::CompletionRequest, embedding::EmbeddingRequest, sampler::Sampler,
};

pub const MAX_TOKENS: usize = 4096;
pub const MAX_PENALTY_COUNT: usize = 1024;

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

pub enum RequestKind {
    Completion(CompletionRequest),
    Chat(ChatRequest),
    Embedding(EmbeddingRequest),
}

pub struct ThreadRequest {
    pub request: RequestKind,
    pub token_sender: flume::Sender<Token>,
}

pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub sampler: Sampler,
    pub occurrences: HashMap<u16, usize>,
    pub embedding: bool,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct TokenCounter {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ThreadState {
    pub sender: flume::Sender<ThreadRequest>,
    pub model_name: String,
}

fn load_tokenizer(path: PathBuf) -> Result<Tokenizer> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(Tokenizer::new(&contents)?)
}

fn load_model(env: &Environment, path: PathBuf) -> Result<Model> {
    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };
    let model = env.create_model_from_bytes(&map)?;
    Ok(model)
}

fn model_task(
    env: Environment,
    model: PathBuf,
    tokenizer: PathBuf,
    receiver: Receiver<ThreadRequest>,
) -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ai00_server", log::LevelFilter::Trace)
        .init()?;

    let tokenizer = load_tokenizer(tokenizer)?;
    let model = load_model(&env, model)?;

    log::info!("{:#?}", env.adapter.get_info());
    log::info!("{:#?}", model.info());

    let penalty_free_tokens = {
        let mut set = HashSet::new();
        for token in 0..u16::MAX {
            let word = tokenizer.decode(&[token]).unwrap_or_default();
            let word = String::from_utf8(word).unwrap_or_default();
            if word.contains('\n') {
                set.insert(token);
            }
        }
        set
    };

    let mut state_cache = Trie::<&[u8], BackedModelState>::new();

    loop {
        let ThreadRequest {
            request,
            token_sender,
        } = match receiver.recv() {
            Ok(request) => request,
            Err(_) => continue,
        };

        let GenerateRequest {
            prompt,
            max_tokens,
            stop,
            sampler,
            mut occurrences,
            embedding,
        } = match request {
            RequestKind::Completion(request) => request.into(),
            RequestKind::Chat(request) => {
                let model_text = Vec::from(request.messages.clone())
                    .into_iter()
                    .filter(|record| record.role == chat::Role::Assistant)
                    .map(|record| record.content)
                    .join(" ");
                let model_tokens = tokenizer.encode(model_text.as_bytes()).unwrap_or_default();
                let occurances = model_tokens
                    .into_iter()
                    .rev()
                    .take(MAX_PENALTY_COUNT)
                    .counts();

                let mut request = GenerateRequest::from(request);
                request.occurrences = occurances;
                request
            }
            RequestKind::Embedding(request) => request.into(),
        };

        log::info!("{:#?}", sampler);

        let state = model.create_state();
        let remain = {
            let prefix = state_cache.longest_common_prefix(prompt.as_bytes());
            let mut remain = prompt.as_bytes().to_vec();
            if state_cache
                .get(prefix)
                .and_then(|backed| state.load(backed).ok())
                .is_some()
            {
                log::info!("state cache hit");
                remain.split_off(prefix.len())
            } else {
                log::info!("state cache miss");
                remain
            }
        };
        let mut model_text = String::new();
        let mut token_counter = TokenCounter::default();

        let mut tokens = tokenizer.encode(&remain).unwrap_or_default();
        let _ = token_sender.send(Token::Start);

        'run: {
            token_counter.prompt_tokens = tokens.len();
            token_counter.total_tokens = tokens.len();

            for _ in 0..max_tokens {
                if token_sender.is_disconnected() {
                    break 'run;
                }

                let mut logits = model.run(&tokens, &state).unwrap_or_default();
                for (&token, &count) in occurrences
                    .iter()
                    .filter(|(token, _)| !penalty_free_tokens.contains(token))
                {
                    let penalty =
                        sampler.presence_penalty + sampler.frequency_penalty * count as f32;
                    logits[token as usize] -= penalty;
                }

                let token = sampler.sample(logits);
                let word = tokenizer
                    .decode(&[token])
                    .ok()
                    .and_then(|x| String::from_utf8(x).ok())
                    .unwrap_or_default();

                print!("{word}");
                std::io::stdout().flush()?;

                model_text += &word;
                token_counter.completion_tokens += 1;
                token_counter.total_tokens += 1;

                let count = occurrences.get(&token).copied().unwrap_or_default();
                occurrences.insert(token, count + 1);

                let _ = token_sender.send(Token::Token(word));
                tokens = vec![token];

                if stop.iter().any(|x| model_text.contains(x)) {
                    let _ = token_sender.send(Token::Stop(FinishReason::Stop, token_counter));
                    break 'run;
                }
            }

            let _ = token_sender.send(Token::Stop(FinishReason::Length, token_counter));
        }

        print!("\n\n");
        std::io::stdout().flush()?;

        if let Ok(back) = state.back() {
            if embedding {
                let num_layer = model.info().num_layers;
                let num_emb = model.info().num_emb;
                let embedding = back.0[num_layer - 3][4 * num_emb..].to_vec();
                let _ = token_sender.send(Token::Embed(embedding));
            }

            let mut prompt = prompt.as_bytes().to_vec();
            let mut model_text = model_text.as_bytes().to_vec();
            prompt.append(&mut model_text);
            state_cache.insert(prompt.leak(), back);
        }

        let _ = token_sender.send(Token::Done);
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short, value_name = "FILE")]
    model: Option<String>,
    #[arg(long, short, value_name = "FILE")]
    tokenizer: Option<String>,
    #[arg(long, short, default_value_t = 3000)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let model_path = PathBuf::from(
        args.model
            .unwrap_or("assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st".into()),
    );
    let model_name = model_path
        .file_name()
        .and_then(OsStr::to_str)
        .map(String::from_str)
        .and_then(Result::ok)
        .map(|name| name.replace(".st", ""))
        .unwrap();

    let tokenizer_path = PathBuf::from(
        args.tokenizer
            .unwrap_or("assets/rwkv_vocab_v20230424.json".into()),
    );

    let (sender, receiver) = flume::unbounded::<ThreadRequest>();
    let env = Environment::create().await?;
    std::thread::spawn(move || model_task(env, model_path, tokenizer_path, receiver));

    let app = Router::new()
        .route("/completions", post(completion::completions))
        .route("/v1/completions", post(completion::completions))
        .route("/chat/completions", post(chat::chat_completions))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/embeddings", post(embedding::embeddings))
        .route("/v1/embeddings", post(embedding::embeddings))
        .fallback_service(ServeDir::new("assets/www"))
        .layer(CorsLayer::permissive())
        .with_state(ThreadState { sender, model_name });

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
