use anyhow::Result;
use axum::{routing::post, Router};
use flume::Receiver;
use itertools::Itertools;
use memmap::Mmap;
use qp_trie::Trie;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read, Write},
};
use web_rwkv::{BackedModelState, Environment, Model, Tokenizer};

mod chat;
mod completion;
mod sampler;

use crate::{chat::ChatRequest, completion::CompletionRequest, sampler::Sampler};

pub const MAX_TOKENS: usize = 4096;
pub const PENALTY_COUNT: usize = 256;

#[derive(Debug)]
pub enum Token {
    Token(String),
    EndOfText,
    CutOff,
}

#[derive(Debug, Default, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// API returned complete model output.
    #[default]
    Stop,
    /// Incomplete model output due to max_tokens parameter or token limit.
    Length,
    /// Omitted content due to a flag from our content filters.
    ContentFilter,
    /// API response still in progress or incomplete.
    Null,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OptionArray<T: std::fmt::Debug + Clone + Serialize> {
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
}

pub struct ThreadRequest {
    pub request: RequestKind,
    pub prompt_tokens_sender: flume::Sender<usize>,
    pub token_sender: flume::Sender<Token>,
}

pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub sampler: Sampler,
    pub occurrences: HashMap<u16, usize>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct TokenCounter {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

fn load_tokenizer() -> Result<Tokenizer> {
    let file = File::open("assets/rwkv_vocab_v20230424.json")?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(Tokenizer::new(&contents)?)
}

fn load_model(env: &Environment) -> Result<Model> {
    let file = File::open("assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st")?;
    let map = unsafe { Mmap::map(&file)? };
    let model = env.create_model_from_bytes(&map)?;
    print!("{:#?}\n\n", model.info());
    Ok(model)
}

async fn model_task(receiver: Receiver<ThreadRequest>) -> Result<()> {
    let env = Environment::create().await?;
    let tokenizer = load_tokenizer()?;
    let model = load_model(&env)?;

    let mut state_cache = Trie::<&[u8], BackedModelState>::new();

    loop {
        let ThreadRequest {
            request,
            prompt_tokens_sender,
            token_sender,
        } = match receiver.recv_async().await {
            Ok(request) => request,
            Err(_) => continue,
        };

        let GenerateRequest {
            prompt,
            max_tokens,
            stop,
            sampler,
            mut occurrences,
        } = match request {
            RequestKind::Completion(request) => request.into(),
            RequestKind::Chat(request) => {
                let model_text = Vec::from(request.messages.clone())
                    .into_iter()
                    .filter(|record| record.role == chat::Role::Assistant)
                    .map(|record| record.content)
                    .join(" ");
                let model_tokens = tokenizer.encode(model_text.as_bytes()).unwrap_or_default();
                let occurances = model_tokens.into_iter().counts();

                let mut request = GenerateRequest::from(request);
                request.occurrences = occurances;
                request
            }
        };

        println!("{:#?}", sampler);
        std::io::stdout().flush()?;

        let state = model.create_state();
        let remain = {
            let prefix = state_cache.longest_common_prefix(prompt.as_bytes());
            let mut remain = prompt.as_bytes().to_vec();
            if state_cache
                .get(prefix)
                .and_then(|backed| state.load(backed).ok())
                .is_some()
            {
                remain.split_off(prefix.len())
            } else {
                remain
            }
        };
        let mut model_text = String::new();

        let mut tokens = tokenizer.encode(&remain).unwrap_or_default();
        let _ = prompt_tokens_sender.send_async(tokens.len()).await;

        'run: {
            for _ in 0..max_tokens {
                let mut logits = model.run(&tokens, &state).unwrap_or_default();
                for (&token, &count) in &occurrences {
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
                model_text += &word;
                tokens = vec![token];

                print!("{}", word);
                std::io::stdout().flush()?;

                let count = occurrences.get(&token).copied().unwrap_or_default();
                occurrences.insert(token, count + 1);

                let _ = token_sender.send_async(Token::Token(word)).await;

                if stop.iter().any(|x| model_text.contains(x)) {
                    let _ = token_sender.send_async(Token::EndOfText).await;
                    break 'run;
                }
            }

            let _ = token_sender.send_async(Token::CutOff).await;
        }

        print!("\n\n");
        std::io::stdout().flush()?;

        if let Ok(back) = state.back() {
            let mut prompt = prompt.as_bytes().to_vec();
            let mut model_text = model_text.as_bytes().to_vec();
            prompt.append(&mut model_text);
            state_cache.insert(prompt.leak(), back);
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let (sender, receiver) = flume::unbounded::<ThreadRequest>();
    let handle = tokio::task::spawn(model_task(receiver));

    let app = Router::new()
        .route("/completions", post(completion::completions))
        .route("/chat/completions", post(chat::chat_completions))
        .with_state(sender);
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await?;

    handle.await??;
    Ok(())
}
