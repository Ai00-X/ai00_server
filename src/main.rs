use anyhow::Result;
use axum::{routing::post, Router};
use flume::Receiver;
use memmap::Mmap;
use serde::Serialize;
use std::{
    fs::File,
    io::{BufReader, Read},
};
use web_rwkv::{Environment, Model, Tokenizer};

mod chat;
mod completion;

use chat::{chat, ChatRequest};
use completion::{completions, CompletionRequest};

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

    loop {
        let thread_request = receiver.recv()?;
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
