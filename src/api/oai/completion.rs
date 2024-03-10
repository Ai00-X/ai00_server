use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::Result;
use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use super::SamplerParams;
use crate::{
    api::request_info,
    middleware::{
        Array, FinishReason, GenerateRequest, ThreadRequest, ThreadState, Token, TokenCounter,
        MAX_TOKENS,
    },
};

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct CompletionRequest {
    prompt: Array<String>,
    max_tokens: usize,
    stop: Array<String>,
    stream: bool,
    #[serde(alias = "logit_bias")]
    bias: HashMap<u16, f32>,
    #[serde(flatten)]
    sampler: SamplerParams,
}

impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            prompt: Array::default(),
            max_tokens: 256,
            stop: Array::default(),
            stream: false,
            bias: HashMap::new(),
            sampler: Default::default(),
        }
    }
}

impl From<CompletionRequest> for GenerateRequest {
    fn from(value: CompletionRequest) -> Self {
        let CompletionRequest {
            prompt,
            max_tokens,
            stop,
            sampler,
            bias,
            ..
        } = value;

        let prompt = Vec::from(prompt).join("");
        let max_tokens = max_tokens.min(MAX_TOKENS);
        let stop = stop.into();
        let bias = Arc::new(bias);
        let sampler = sampler.into();

        Self {
            prompt,
            max_tokens,
            stop,
            sampler,
            bias,
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    object: String,
    model: String,
    choices: Vec<CompletionChoice>,
    #[serde(rename = "usage")]
    counter: TokenCounter,
}

async fn respond_one(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let info = request_info(sender.clone(), Duration::from_secs(1)).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let request = GenerateRequest::from(request);
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

    Json(CompletionResponse {
        object: "text_completion".into(),
        model: model_name,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason,
        }],
        counter: token_counter,
    })
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialCompletionRecord {
    #[default]
    #[serde(rename = "")]
    None,
    Content(String),
}

#[derive(Debug, Default, Serialize)]
pub struct PartialCompletionChoice {
    delta: PartialCompletionRecord,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
pub struct PartialCompletionResponse {
    object: String,
    model: String,
    choices: Vec<PartialCompletionChoice>,
}

async fn respond_stream(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event>>> {
    let info = request_info(sender.clone(), Duration::from_secs(1)).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let request = GenerateRequest::from(request);
    let _ = sender.send(ThreadRequest::Generate {
        request,
        tokenizer: info.tokenizer,
        sender: token_sender,
    });

    let stream = token_receiver.into_stream().skip(1).map(move |token| {
        let choice = match token {
            Token::Content(token) => PartialCompletionChoice {
                delta: PartialCompletionRecord::Content(token),
                ..Default::default()
            },
            Token::Stop(finish_reason, _) => PartialCompletionChoice {
                finish_reason,
                ..Default::default()
            },
            Token::Done => return Ok(Event::default().data("[DONE]")),
            _ => unreachable!(),
        };

        let json = serde_json::to_string(&PartialCompletionResponse {
            object: "text_completion.chunk".into(),
            model: model_name.clone(),
            choices: vec![choice],
        })?;
        Ok(Event::default().data(json))
    });

    Sse::new(stream)
}

/// `/api/oai/completions`, `/api/oai/v1/completions`.
pub async fn completions(
    state: State<ThreadState>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    match request.stream {
        true => respond_stream(state, Json(request)).await.into_response(),
        false => respond_one(state, Json(request)).await.into_response(),
    }
}
