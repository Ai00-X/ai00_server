use std::{collections::HashMap, sync::Arc, time::Duration};

use futures_util::StreamExt;
use salvo::oapi::{ToResponse, ToSchema};
use serde::{Deserialize, Serialize};

use super::SamplerParams;
use crate::middleware::{Array, FinishReason, GenerateRequest, TokenCounter, MAX_TOKENS};

pub use private::completions;

#[derive(Debug, Deserialize, ToSchema, ToResponse)]
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

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct CompletionResponse {
    object: String,
    model: String,
    choices: Vec<CompletionChoice>,
    #[serde(rename = "usage")]
    counter: TokenCounter,
}

#[derive(Debug, Default, Serialize, ToSchema, ToResponse)]
#[serde(rename_all = "snake_case")]
pub enum PartialCompletionRecord {
    #[default]
    #[serde(rename = "")]
    None,
    Content(String),
}

#[derive(Debug, Default, Serialize, ToSchema, ToResponse)]
pub struct PartialCompletionChoice {
    delta: PartialCompletionRecord,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct PartialCompletionResponse {
    object: String,
    model: String,
    choices: Vec<PartialCompletionChoice>,
}

#[cfg(feature = "axum-api")]
mod private {
    use anyhow::Result;
    use axum::{
        extract::State,
        response::{sse::Event, IntoResponse, Response, Sse},
        Json,
    };
    use futures_util::Stream;

    use super::*;
    use crate::{
        api::request_info,
        middleware::{FinishReason, ThreadRequest, ThreadState, Token, TokenCounter},
    };

    async fn respond_one(
        State(ThreadState(sender)): State<ThreadState>,
        Json(request): Json<CompletionRequest>,
    ) -> Json<CompletionResponse> {
        let info = request_info(sender.clone(), Duration::from_secs(1)).await;
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

    async fn respond_stream(
        State(ThreadState(sender)): State<ThreadState>,
        Json(request): Json<CompletionRequest>,
    ) -> Sse<impl Stream<Item = Result<Event>>> {
        let info = request_info(sender.clone(), Duration::from_secs(1)).await;
        let model_name = info.reload.model_path.to_string_lossy().into_owned();

        let (token_sender, token_receiver) = flume::unbounded();
        let request = Box::new(request.into());
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
}

#[cfg(feature = "salvo-api")]
mod private {
    use salvo::{oapi::extract::JsonBody, prelude::*, sse::SseEvent, Depot, Writer};

    use super::*;
    use crate::{
        api::request_info,
        middleware::{FinishReason, ThreadRequest, ThreadState, Token, TokenCounter},
    };

    async fn respond_one(depot: &mut Depot, request: CompletionRequest, res: &mut Response) {
        let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
        let info = request_info(sender.clone(), Duration::from_secs(1)).await;
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

        let json = Json(CompletionResponse {
            object: "text_completion".into(),
            model: model_name,
            choices: vec![CompletionChoice {
                text,
                index: 0,
                finish_reason,
            }],
            counter: token_counter,
        });
        res.render(json);
    }

    async fn respond_stream(depot: &mut Depot, request: CompletionRequest, res: &mut Response) {
        let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
        let info = request_info(sender.clone(), Duration::from_secs(1)).await;
        let model_name = info.reload.model_path.to_string_lossy().into_owned();

        let (token_sender, token_receiver) = flume::unbounded();
        let request = Box::new(request.into());
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
                Token::Done => return Ok(SseEvent::default().text("[DONE]")),
                _ => unreachable!(),
            };

            match serde_json::to_string(&PartialCompletionResponse {
                object: "text_completion.chunk".into(),
                model: model_name.clone(),
                choices: vec![choice],
            }) {
                Ok(json_text) => Ok(SseEvent::default().text(json_text)),
                Err(err) => Err(err),
            }
        });
        salvo::sse::stream(res, stream);
    }

    /// Generate completions for the given text.
    #[endpoint(
        responses(
            (status_code = 200, description = "Generate one response if `stream` is false.", body = CompletionResponse),
            (status_code = 201, description = "Generate SSE response if `stream` is true. `StatusCode` should be 200.", body = PartialCompletionResponse)
        )
    )]
    pub async fn completions(
        depot: &mut Depot,
        req: JsonBody<CompletionRequest>,
        res: &mut Response,
    ) {
        let request = req.0;
        match request.stream {
            true => respond_stream(depot, request, res).await,
            false => respond_one(depot, request, res).await,
        }
    }
}
