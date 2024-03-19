use anyhow::Result;
use futures_util::{Stream, StreamExt};
use itertools::Itertools;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};
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
use salvo::sse::{self, SseEvent};
use salvo::{
    macros::{handler, Extractible},
    oapi::extract::JsonBody,
    prelude::*,
    Depot, Writer,
};

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
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

#[derive(Default, Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatRecord {
    role: Role,
    content: String,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
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

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct ChatChoice {
    message: ChatRecord,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct ChatResponse {
    object: String,
    model: String,
    choices: Vec<ChatChoice>,
    #[serde(rename = "usage")]
    counter: TokenCounter,
}

#[derive(Debug, Default, Serialize, ToSchema, ToResponse)]
#[serde(rename_all = "snake_case")]
enum PartialChatRecord {
    #[default]
    #[serde(rename = "")]
    None,
    Role(Role),
    Content(String),
}

#[derive(Debug, Default, Serialize, ToSchema, ToResponse)]
struct PartialChatChoice {
    delta: PartialChatRecord,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct PartialChatResponse {
    object: String,
    model: String,
    choices: Vec<PartialChatChoice>,
}

async fn salvo_oai_chat_respond_one(
    depot: &mut Depot,
    request: ChatRequest,
    res: &mut salvo::prelude::Response,
) {
    let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
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

    let json_ = salvo::prelude::Json(ChatResponse {
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
    });
    res.render(json_);
}

async fn salvo_oai_chat_respond_stream(
    depot: &mut Depot,
    request: ChatRequest,
    res: &mut salvo::prelude::Response,
) {
    let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
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
            Token::Done => return Ok(SseEvent::default().text("[DONE]")),
            _ => unreachable!(),
        };

        match serde_json::to_string(&PartialChatResponse {
            object: "chat.completion.chunk".into(),
            model: model_name.clone(),
            choices: vec![choice],
        }) {
            Ok(json_text) => Ok(SseEvent::default().text(json_text)),
            Err(err) => Err(err),
        }
    });
    salvo::sse::stream(res, stream);
}

/// Generate the chat completions for giving inputs
/// 由RWKV根据输入的上下文作为前提来产生对话
#[endpoint(
    responses(
        (status_code=200, description="Generate one response for stream is false.", body=ChatResponse),
        (status_code=201, description="Generate Server Side Event response for stream is true. StatusCode should be 200.", body=PartialChatResponse)
    )
)]
pub async fn salvo_oai_chat_completions(
    depot: &mut Depot,
    req: JsonBody<ChatRequest>,
    res: &mut salvo::http::Response,
) {
    let request = req.0;
    match request.stream {
        true => {
            salvo_oai_chat_respond_stream(depot, request, res).await;
        }
        false => {
            salvo_oai_chat_respond_one(depot, request, res).await;
        }
    }
}
