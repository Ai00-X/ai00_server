use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::Result;
use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures_util::{Stream, StreamExt};
use itertools::Itertools;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::SamplerParams;
use crate::{
    api::request_info,
    middleware::{
        Array, FinishReason, GenerateRequest, ThreadRequest, ThreadState, Token, TokenCounter,
        MAX_TOKENS,
    },
    sampler::Sampler,
};

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    #[default]
    #[serde(alias = "system")]
    System,
    #[serde(alias = "user")]
    User,
    #[serde(alias = "assistant")]
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
    role: Role,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ChatRequest {
    messages: Array<ChatRecord>,
    names: HashMap<Role, String>,
    max_tokens: usize,
    stop: Array<String>,
    stream: bool,
    #[serde(alias = "logit_bias")]
    bias: HashMap<u16, f32>,
    #[serde(flatten)]
    sampler: SamplerParams,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Array::default(),
            names: HashMap::new(),
            max_tokens: 256,
            stop: Array::Item("\n\n".into()),
            stream: false,
            bias: HashMap::new(),
            sampler: Default::default(),
        }
    }
}

impl From<ChatRequest> for GenerateRequest {
    fn from(value: ChatRequest) -> Self {
        let ChatRequest {
            messages,
            names,
            max_tokens,
            stop,
            sampler,
            bias,
            ..
        } = value;

        let re = Regex::new(r"\n(\s*\n)+").unwrap();
        let prompt = Vec::from(messages.clone())
            .into_iter()
            .map(|ChatRecord { role, content }| {
                let role = names.get(&role).cloned().unwrap_or(role.to_string());
                let content = re.replace_all(&content, "\n");
                let content = content.trim();
                format!("{role}: {content}")
            })
            .join("\n\n");
        let model_text = Vec::from(messages)
            .into_iter()
            .filter(|record| record.role == Role::Assistant)
            .map(|record| record.content)
            .join("\n\n");

        let assistant = Role::Assistant;
        let assistant = names
            .get(&assistant)
            .cloned()
            .unwrap_or(assistant.to_string());
        let prompt = prompt + &format!("\n\n{assistant}:");

        let max_tokens = max_tokens.min(MAX_TOKENS);
        let stop = stop.into();
        let bias = Arc::new(bias);
        let sampler: Arc<RwLock<dyn Sampler + Send + Sync>> = sampler.into();

        Self {
            prompt,
            model_text,
            max_tokens,
            stop,
            sampler,
            bias,
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    message: ChatRecord,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    object: String,
    model: String,
    choices: Vec<ChatChoice>,
    #[serde(rename = "usage")]
    counter: TokenCounter,
}

async fn respond_one(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let info = request_info(sender.clone(), Duration::from_secs(1)).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let request = request.into();
    let _ = sender.send(ThreadRequest::Generate {
        request,
        tokenizer: info.tokenizer,
        sender: token_sender,
    });

    let mut token_counter = TokenCounter::default();
    let mut finish_reason = FinishReason::Null;
    let mut text = String::new();
    let mut stream = token_receiver.into_stream();

    while let Some(token) = stream.next().await {
        match token {
            Token::Start => {}
            Token::Content(token) => {
                text += &token;
            }
            Token::Stop(reason, counter) => {
                finish_reason = reason;
                token_counter = counter;
                break;
            }
            _ => unreachable!(),
        }
    }

    Json(ChatResponse {
        object: "chat.completion".into(),
        model: model_name,
        choices: vec![ChatChoice {
            message: ChatRecord {
                role: Role::Assistant,
                content: text.trim().into(),
            },
            index: 0,
            finish_reason,
        }],
        counter: token_counter,
    })
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
enum PartialChatRecord {
    #[default]
    #[serde(rename = "")]
    None,
    Role(Role),
    Content(String),
}

#[derive(Debug, Default, Serialize)]
struct PartialChatChoice {
    delta: PartialChatRecord,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
struct PartialChatResponse {
    object: String,
    model: String,
    choices: Vec<PartialChatChoice>,
}

async fn respond_stream(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event>>> {
    let info = request_info(sender.clone(), Duration::from_secs(1)).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let request = request.into();
    let _ = sender.send(ThreadRequest::Generate {
        request,
        tokenizer: info.tokenizer,
        sender: token_sender,
    });

    let mut start_token = true;
    let stream = token_receiver.into_stream().map(move |token| {
        let choice = match token {
            Token::Start => PartialChatChoice {
                delta: PartialChatRecord::Role(Role::Assistant),
                ..Default::default()
            },
            Token::Content(token) => {
                let token = match start_token {
                    true => token.trim_start().into(),
                    false => token,
                };
                start_token = false;
                PartialChatChoice {
                    delta: PartialChatRecord::Content(token),
                    ..Default::default()
                }
            }
            Token::Stop(finish_reason, _) => PartialChatChoice {
                finish_reason,
                ..Default::default()
            },
            Token::Done => return Ok(Event::default().data("[DONE]")),
            _ => unreachable!(),
        };

        let json = serde_json::to_string(&PartialChatResponse {
            object: "chat.completion.chunk".into(),
            model: model_name.clone(),
            choices: vec![choice],
        })?;
        Ok(Event::default().data(json))
    });

    Sse::new(stream)
}

/// `/api/oai/chat/completions`, `/api/oai/v1/chat/completions`.
pub async fn chat_completions(
    state: State<ThreadState>,
    Json(request): Json<ChatRequest>,
) -> Response {
    match request.stream {
        true => respond_stream(state, Json(request)).await.into_response(),
        false => respond_one(state, Json(request)).await.into_response(),
    }
}
