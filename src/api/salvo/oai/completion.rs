use anyhow::Result;
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};

use super::SamplerParams;
use crate::{
    api::request_info,
    middleware::{
        Array, FinishReason, GenerateRequest, ThreadRequest, ThreadState, Token, TokenCounter,
        MAX_TOKENS,
    },
};
use salvo::{
    macros::{handler, Extractible},
    oapi::extract::JsonBody,
    prelude::*,
    Depot, Writer,
};

#[derive(Debug, Deserialize, Serialize, ToSchema)]
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

async fn salvo_oai_respond_one(
    depot: &mut Depot,
    request: CompletionRequest,
) -> salvo::prelude::Json<CompletionResponse> {
    let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
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

    salvo::prelude::Json(CompletionResponse {
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

use salvo::sse::{self, SseEvent};

async fn salvo_oai_respond_stream(
    depot: &mut Depot,
    request: CompletionRequest,
    res: &mut salvo::http::Response,
) {
    let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
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


/// Generate the completions for giving text
/// 
/// 由RWKV根据输入的上下文作为前提来产生后续的内容
#[endpoint(
    responses(
        (status_code=200, description="Generate one response for stream is false.", body=CompletionResponse),
        (status_code=201, description="Generate Server Side Event response for stream is true. StatusCode should be 200.", body=PartialCompletionResponse)
    ))]
pub async fn salvo_oai_completions(
    depot: &mut Depot,
    req: JsonBody<CompletionRequest>,
    res: &mut salvo::http::Response,
) {
    let request = req.0;
    match request.stream {
        true => {
            salvo_oai_respond_stream(depot, request, res).await;
        }
        false => {
            let resp = salvo_oai_respond_one(depot, request).await;
            res.render(resp);
        }
    }
}
