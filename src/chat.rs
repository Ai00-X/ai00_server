use std::collections::HashMap;

use anyhow::Result;
use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures_util::{Stream, StreamExt};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    sampler::Sampler, FinishReason, GenerateRequest, OptionArray, ThreadRequest, ThreadState,
    Token, TokenCounter, MAX_PENALTY_COUNT, ALPHA_DECAY
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    #[serde(alias = "system")]
    System,
    #[default]
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
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ChatRequest {
    pub messages: OptionArray<ChatRecord>,
    pub max_tokens: usize,
    pub stop: OptionArray<String>,
    pub stream: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub logit_bias: HashMap<u16, f32>,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: OptionArray::default(),
            max_tokens: 256,
            stop: OptionArray::Item("\n\n".into()),
            stream: false,
            temperature: 1.0,
            top_p: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: HashMap::new(),
        }
    }
}

impl From<ChatRequest> for GenerateRequest {
    fn from(value: ChatRequest) -> Self {
        let ChatRequest {
            messages,
            max_tokens,
            stop,
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            logit_bias,
            ..
        } = value;

        let prompt = Vec::from(messages)
            .into_iter()
            .map(|ChatRecord { role, content }| {
                let role = role.to_string();
                let content = content.trim();
                format!("{role}: {content}")
            })
            .join("\n\n");

        let assistant = Role::Assistant.to_string();
        let prompt = prompt + &format!("\n\n{assistant}:");

        let max_tokens = max_tokens.min(crate::MAX_TOKENS);
        let stop = stop.into();

        Self {
            prompt,
            max_tokens,
            stop,
            sampler: Sampler {
                top_p,
                temperature,
                presence_penalty,
                frequency_penalty,
            },
            logit_bias,
            embedding: false,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub message: ChatRecord,
    pub index: usize,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub object: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(rename = "usage")]
    pub counter: TokenCounter,
}

async fn chat_completions_one(
    State(ThreadState {
        sender,
        model_name,
        tokenizer,
    }): State<ThreadState>,
    Json(request): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let (token_sender, token_receiver) = flume::unbounded();

    let model_text = Vec::from(request.messages.clone())
        .into_iter()
        .filter(|record| record.role == Role::Assistant)
        .map(|record| record.content)
        .join("\n\n");
    let model_tokens = tokenizer.encode(model_text.as_bytes()).unwrap_or_default();
    let mut occurrences:HashMap<u16, f32> = HashMap::new();
    for key in model_tokens.iter().take(MAX_PENALTY_COUNT) {
        if occurrences.contains_key(key) {
            occurrences.insert(*key, occurrences[key] + 1.0);
        }
        else {
            occurrences.insert(*key, 1.0);
        }
        for (_, count) in occurrences.iter_mut() {
            *count = *count * ALPHA_DECAY;
        }
    }
    let request = request.into();

    let _ = sender.send(ThreadRequest::Generate {
        request,
        occurrences,
        token_sender,
    });

    let mut token_counter = TokenCounter::default();
    let mut finish_reason = FinishReason::Null;
    let mut text = String::new();
    let mut stream = token_receiver.into_stream();

    while let Some(token) = stream.next().await {
        match token {
            Token::Start => {}
            Token::Token(token) => {
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
                content: text,
            },
            index: 0,
            finish_reason,
        }],
        counter: token_counter,
    })
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialChatRecord {
    #[default]
    #[serde(rename = "")]
    None,
    Role(Role),
    Content(String),
}

#[derive(Debug, Default, Serialize)]
pub struct PartialChatChoice {
    pub delta: PartialChatRecord,
    pub index: usize,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
pub struct PartialChatResponse {
    pub object: String,
    pub model: String,
    pub choices: Vec<PartialChatChoice>,
}

async fn chat_completions_stream(
    State(ThreadState {
        sender,
        model_name,
        tokenizer,
    }): State<ThreadState>,
    Json(request): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event>>> {
    let (token_sender, token_receiver) = flume::unbounded();

    let model_text = Vec::from(request.messages.clone())
        .into_iter()
        .filter(|record| record.role == Role::Assistant)
        .map(|record| record.content)
        .join("\n\n");
    let model_tokens = tokenizer.encode(model_text.as_bytes()).unwrap_or_default();
    let mut occurrences:HashMap<u16, f32> = HashMap::new();
    for key in model_tokens.iter().take(MAX_PENALTY_COUNT) {
        if occurrences.contains_key(key) {
            occurrences.insert(*key, occurrences[key] + 1.0);
        }
        else {
            occurrences.insert(*key, 1.0);
        }
        for (_, count) in occurrences.iter_mut() {
            *count = *count * ALPHA_DECAY;
        }
    }
    let request = request.into();

    let _ = sender.send(ThreadRequest::Generate {
        request,
        occurrences,
        token_sender,
    });

    let stream = token_receiver.into_stream().map(move |token| {
        let choice = match token {
            Token::Start => PartialChatChoice {
                delta: PartialChatRecord::Role(Role::Assistant),
                ..Default::default()
            },
            Token::Token(token) => PartialChatChoice {
                delta: PartialChatRecord::Content(token),
                ..Default::default()
            },
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

pub async fn chat_completions(
    state: State<ThreadState>,
    Json(request): Json<ChatRequest>,
) -> Response {
    if request.stream {
        chat_completions_stream(state, Json(request))
            .await
            .into_response()
    } else {
        chat_completions_one(state, Json(request))
            .await
            .into_response()
    }
}
