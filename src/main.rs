use anyhow::Result;
use axum::{routing::post, Router};
use flume::Receiver;
use itertools::Itertools;
use memmap::Mmap;
use qp_trie::Trie;
use serde::Serialize;
use std::{
    fs::File,
    io::{BufReader, Read},
};
use web_rwkv::{BackedModelState, Environment, Model, Tokenizer};

mod chat;
mod completion;
mod sampler;

use chat::{chat, ChatRecord, ChatRequest};
use completion::{completions, CompletionRequest};
use sampler::Sampler;

pub const MAX_TOKENS: usize = 4096;

#[derive(Debug)]
pub enum Token {
    Token(String),
    EndOfText,
    CutOff,
}

#[derive(Debug, Default, Clone, Copy, Serialize)]
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

pub enum RequestKind {
    Completion(CompletionRequest),
    Chat(ChatRequest),
}

pub struct ThreadRequest {
    pub request: RequestKind,
    pub prompt_tokens_sender: flume::Sender<usize>,
    pub token_sender: flume::Sender<Token>,
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
    println!("{:#?}", model.info());
    Ok(model)
}

async fn model_task(receiver: Receiver<ThreadRequest>) -> Result<()> {
    let env = Environment::create().await?;
    let tokenizer = load_tokenizer()?;
    let model = load_model(&env)?;
    print!("{:#?}", model.info());

    let state_cache = Trie::<Vec<u8>, BackedModelState>::new();

    loop {
        let ThreadRequest {
            request,
            prompt_tokens_sender,
            token_sender,
        } = match receiver.recv() {
            Ok(request) => request,
            Err(_) => continue,
        };

        let (text, max_tokens, stop, sampler) = match request {
            RequestKind::Completion(CompletionRequest {
                prompt,
                max_tokens,
                stop,
                temperature,
                top_p,
                presence_penalty,
                frequency_penalty,
            }) => {
                let text = prompt.join("");
                let max_tokens = max_tokens.min(MAX_TOKENS);
                (
                    text,
                    max_tokens,
                    stop,
                    Sampler {
                        top_p,
                        temperature,
                        presence_penalty,
                        frequency_penalty,
                    },
                )
            }
            RequestKind::Chat(ChatRequest {
                messages,
                max_tokens,
                stop,
                temperature,
                top_p,
                presence_penalty,
                frequency_penalty,
            }) => {
                let text = messages
                    .into_iter()
                    .map(|ChatRecord { role, content }| {
                        let role = role.to_string();
                        let content = content.trim();
                        format!("{role}: {content}")
                    })
                    .join("\n\n");

                let assistant = chat::Role::Assistant.to_string();
                let text = text + &format!("\n\n{assistant}: ");

                let max_tokens = max_tokens.min(MAX_TOKENS);
                (
                    text,
                    max_tokens,
                    stop,
                    Sampler {
                        top_p,
                        temperature,
                        presence_penalty,
                        frequency_penalty,
                    },
                )
            }
        };

        let state = model.create_state();
        let remain_text = {
            let mut key = text.as_bytes().to_vec();
            if let Some((prefix, backed)) = state_cache.iter_prefix(&key).last() {
                match state.load(backed) {
                    Ok(_) => String::from_utf8(key.split_off(prefix.len())).unwrap_or(text),
                    Err(_) => text,
                }
            } else {
                text
            }
        };

        let tokens = tokenizer.encode(remain_text.as_bytes()).unwrap_or_default();
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let (sender, receiver) = flume::unbounded::<ThreadRequest>();
    let handle = tokio::task::spawn(model_task(receiver));

    let app = Router::new()
        .route("/completions", post(completions))
        .route("/chat/completions", post(chat))
        .with_state(sender);
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await?;

    handle.await??;
    Ok(())
}
