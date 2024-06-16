use std::{collections::HashMap, sync::Arc};

use ai00_core::{
    run::StateId, FinishReason, GenerateRequest, ThreadRequest, Token, TokenCounter, MAX_TOKENS,
};
use derivative::Derivative;
use futures_util::StreamExt;
use itertools::Itertools;
use regex::Regex;
use salvo::{oapi::extract::JsonBody, prelude::*, sse::SseEvent, Depot, Writer};
use serde::{Deserialize, Serialize};

use super::*;
use crate::{
    api::request_info,
    types::{Array, ThreadState},
    SLEEP,
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

#[derive(Debug, Derivative, Deserialize, ToSchema)]
#[derivative(Default)]
#[serde(default)]
#[salvo(schema(
    example = json!({
        "messages": [
            {
                "role": "user",
                "content": "Hi!"
            },
            {
                "role": "assistant",
                "content": "Hello, I am your AI assistant. If you have any questions or instructions, please let me know!"
            },
            {
                "role": "user",
                "content": "Tell me about water."
            }
        ],
        "names": {
            "user": "User",
            "assistant": "Assistant"
        },
        "stop": [
            "\n\nUser:"
        ],
        "stream": false,
        "max_tokens": 1000,
        "sampler_override": {
            "type": "Nucleus",
            "top_p": 0.5,
            "top_k": 128,
            "temperature": 1,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "penalty": 400,
            "penalty_decay": 0.99654026
        },
        "state": "00000000-0000-0000-0000-000000000000"
    })
))]
pub struct ChatRequest {
    messages: Array<ChatRecord>,
    names: HashMap<Role, String>,
    state: StateId,
    #[derivative(Default(value = "256"))]
    max_tokens: usize,
    #[derivative(Default(value = "Array::Item(\"\\n\\n\".into())"))]
    stop: Array<String>,
    stream: bool,
    #[serde(alias = "logit_bias")]
    bias: HashMap<u16, f32>,
    bnf_schema: Option<String>,
    sampler: NucleusParams,
    sampler_override: Option<SamplerParams>,
}

impl From<ChatRequest> for GenerateRequest {
    fn from(value: ChatRequest) -> Self {
        let ChatRequest {
            messages,
            names,
            state,
            max_tokens,
            stop,
            sampler,
            sampler_override,
            bias,
            bnf_schema,
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
        let sampler = match sampler_override {
            Some(sampler) => sampler.into(),
            None => SamplerParams::Nucleus(sampler).into(),
        };

        Self {
            prompt,
            model_text,
            max_tokens,
            stop,
            sampler,
            bias,
            bnf_schema,
            state,
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
#[salvo(schema(
    example = json!({
        "object": "chat.completion",
        "model": "assets/models\\RWKV-x060-World-3B-v2.1-20240417-ctx4096.st",
        "choices": [
            {
                "message": {
                    "role": "Assistant",
                    "content": "Water is a chemical compound made up of two hydrogen atoms and one oxygen atom, bonded together with a shared electron pair. It is the most abundant substance in the Earth's atmosphere and oceans, and is essential for life on Earth. Water can exist in many different forms, including liquid water, ice, and solid ice. It is also an important solvent for many chemical reactions and processes. Water is found in all parts of the world and is essential for life on Earth."
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt": 41,
            "completion": 97,
            "total": 138,
            "duration": {
                "secs": 8,
                "nanos": 381235000
            }
        }
    })
))]
struct ChatResponse {
    object: String,
    model: String,
    choices: Vec<ChatChoice>,
    #[serde(rename = "usage")]
    counter: TokenCounter,
}

#[derive(Debug, Derivative, Serialize, ToSchema, ToResponse)]
#[derivative(Default)]
#[serde(rename_all = "snake_case")]
enum PartialChatRecord {
    Role(Role),
    Content(String),
    #[derivative(Default)]
    #[serde(untagged)]
    None(HashMap<String, String>),
}

#[derive(Debug, Default, Serialize, ToSchema, ToResponse)]
struct PartialChatChoice {
    delta: PartialChatRecord,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
#[salvo(schema(
    example = json!({
        "object": "chat.completion.chunk",
        "model": "assets/models\\RWKV-x060-World-3B-v2.1-20240417-ctx4096.st",
        "choices": [
            {
                "delta": {
                    "role": "Assistant"
                },
                "index": 0,
                "finish_reason": null
            }
        ]
    })
))]
struct PartialChatResponse {
    object: String,
    model: String,
    choices: Vec<PartialChatChoice>,
}

async fn respond_one(depot: &mut Depot, request: ChatRequest, res: &mut Response) {
    let ThreadState { sender, .. } = depot.obtain::<ThreadState>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let request = Box::new(request.into());
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

    let json = Json(ChatResponse {
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
    res.render(json);
}

async fn respond_stream(depot: &mut Depot, request: ChatRequest, res: &mut Response) {
    let ThreadState { sender, .. } = depot.obtain::<ThreadState>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let request = Box::new(request.into());
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

/// Generate chat completions with context.
#[endpoint(
    responses(
        (status_code = 200, description = "Generate one response if `stream` is false.", body = ChatResponse),
        (status_code = 201, description = "Generate SSE response if `stream` is true.", body = PartialChatResponse)
    )
)]
pub async fn chat_completions(depot: &mut Depot, req: JsonBody<ChatRequest>, res: &mut Response) {
    let request = req.0;
    match request.stream {
        true => respond_stream(depot, request, res).await,
        false => respond_one(depot, request, res).await,
    }
}
