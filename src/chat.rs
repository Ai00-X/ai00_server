use axum::{extract::State, Json};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{FinishReason, ThreadRequest, TokenCounter};

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum Role {
    System,
    #[default]
    User,
    Assistant,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct ChatRecord {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ChatRequest {
    pub messages: Vec<ChatRecord>,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub stream: bool,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            max_tokens: 256,
            stop: Vec::new(),
            temperature: 1.0,
            top_p: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stream: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub message: ChatRecord,
    pub index: usize,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatResponse {
    pub object: String,
    pub choices: Vec<ChatChoice>,
    pub usage: TokenCounter,
}

pub async fn chat(
    State(sender): State<flume::Sender<ThreadRequest>>,
    Json(request): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let (prompt_tokens_sender, prompt_tokens_receiver) = flume::unbounded();
    let (token_sender, token_receiver) = flume::unbounded();

    sender
        .send(ThreadRequest {
            request: crate::RequestKind::Chat(request),
            prompt_tokens_sender,
            token_sender,
        })
        .unwrap();

    let prompt_tokens = prompt_tokens_receiver.recv_async().await.unwrap();
    let mut usage = TokenCounter {
        prompt_tokens,
        completion_tokens: 0,
        total_tokens: prompt_tokens,
    };

    let mut finish_reason = FinishReason::Null;
    let mut text = String::new();
    let mut stream = token_receiver.into_stream();

    while let Some(token) = stream.next().await {
        match token {
            crate::Token::Token(token) => {
                text += &token;
                usage.completion_tokens += 1;
                usage.total_tokens += 1;
            }
            crate::Token::EndOfText => {
                finish_reason = FinishReason::Stop;
            }
        }
    }

    Json(ChatResponse {
        object: "text.completion".into(),
        choices: vec![ChatChoice {
            message: ChatRecord {
                role: Role::Assistant,
                content: text,
            },
            index: 0,
            finish_reason,
        }],
        usage,
    })
}
