use axum::{extract::State, Json};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{FinishReason, ThreadRequest, TokenCounter};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    System,
    #[default]
    User,
    Assistant,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "System"),
            Role::User => write!(f, "User"),
            Role::Assistant => write!(f, "Assistant"),
        }
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct ChatRecord {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ChatRequest {
    pub messages: Vec<ChatRecord>,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
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
    #[serde(rename = "usage")]
    pub counter: TokenCounter,
}

pub async fn chat(
    State(sender): State<flume::Sender<ThreadRequest>>,
    Json(request): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let (prompt_tokens_sender, prompt_tokens_receiver) = flume::unbounded();
    let (token_sender, token_receiver) = flume::unbounded();

    let _ = sender
        .send_async(ThreadRequest {
            request: crate::RequestKind::Chat(request.clone()),
            prompt_tokens_sender,
            token_sender,
        })
        .await;

    let prompt_tokens = prompt_tokens_receiver
        .recv_async()
        .await
        .unwrap_or_default();
    let mut counter = TokenCounter {
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
                counter.completion_tokens += 1;
                counter.total_tokens += 1;
            }
            crate::Token::EndOfText => {
                finish_reason = FinishReason::Stop;
            }
            crate::Token::CutOff => {
                finish_reason = FinishReason::Length;
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
        counter,
    })
}
