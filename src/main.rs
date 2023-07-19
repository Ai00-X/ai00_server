use anyhow::Result;
use axum::{routing::post, Router};
use flume::Receiver;
use itertools::Itertools;
use memmap::Mmap;
use qp_trie::Trie;
use serde::Serialize;
use std::{
    collections::HashMap,
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
pub const PENALTY_COUNT: usize = 256;

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

    let mut state_cache = Trie::<&[u8], BackedModelState>::new();

    loop {
        let ThreadRequest {
            request,
            prompt_tokens_sender,
            token_sender,
        } = match receiver.recv() {
            Ok(request) => request,
            Err(_) => continue,
        };

        let (prompt, max_tokens, stop, sampler, mut occurrences) = match request {
            RequestKind::Completion(CompletionRequest {
                prompt,
                max_tokens,
                stop,
                temperature,
                top_p,
                presence_penalty,
                frequency_penalty,
            }) => {
                let prompt = prompt.join("");
                let max_tokens = max_tokens.min(MAX_TOKENS);
                (
                    prompt,
                    max_tokens,
                    stop,
                    Sampler {
                        top_p,
                        temperature,
                        presence_penalty,
                        frequency_penalty,
                    },
                    HashMap::new(),
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
                let model_text = messages
                    .iter()
                    .filter(|&record| record.role == chat::Role::Assistant)
                    .map(|record| record.content.clone())
                    .join(" ");
                let model_tokens = tokenizer.encode(model_text.as_bytes()).unwrap_or_default();
                let occurances = model_tokens.into_iter().counts();

                let prompt = messages
                    .into_iter()
                    .map(|ChatRecord { role, content }| {
                        let role = role.to_string();
                        let content = content.trim();
                        format!("{role}: {content}")
                    })
                    .join("\n\n");

                let assistant = chat::Role::Assistant.to_string();
                let text = prompt + &format!("\n\n{assistant}:");

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
                    occurances,
                )
            }
        };

        let state = model.create_state();
        let remain = {
            let prefix = state_cache.longest_common_prefix(prompt.as_bytes());
            let mut prompt = prompt.as_bytes().to_vec();
            if let Some((_, backed)) = state_cache.iter_prefix(prefix).last() {
                match state.load(backed) {
                    Ok(_) => prompt.split_off(prefix.len()),
                    Err(_) => prompt,
                }
            } else {
                prompt
            }
        };
        let mut model_text = String::new();

        let mut tokens = tokenizer.encode(&remain).unwrap_or_default();
        let _ = prompt_tokens_sender.send_async(tokens.len()).await;

        'generate: {
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

                let count = occurrences.get(&token).copied().unwrap_or_default();
                occurrences.insert(token, count + 1);

                let _ = token_sender.send_async(Token::Token(word)).await;

                if stop.iter().any(|x| model_text.contains(x)) {
                    let _ = token_sender.send_async(Token::EndOfText).await;
                    break 'generate;
                }
            }

            let _ = token_sender.send_async(Token::CutOff).await;
        }

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
        .route("/completions", post(completions))
        .route("/chat/completions", post(chat))
        .with_state(sender);
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await?;

    handle.await??;
    Ok(())
}
