use axum::{extract::State, Json};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{FinishReason, ThreadRequest, TokenCounter};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CompletionRequest {
    pub prompt: Vec<String>,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            prompt: Vec::new(),
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
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub object: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(rename = "usage")]
    pub counter: TokenCounter,
}

pub async fn completions(
    State(sender): State<flume::Sender<ThreadRequest>>,
    Json(request): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let (prompt_tokens_sender, prompt_tokens_receiver) = flume::unbounded();
    let (token_sender, token_receiver) = flume::unbounded();

    sender
        .send(ThreadRequest {
            request: crate::RequestKind::Completion(request),
            prompt_tokens_sender,
            token_sender,
        })
        .unwrap();

    let prompt_tokens = prompt_tokens_receiver.recv_async().await.unwrap();
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

    Json(CompletionResponse {
        object: "text_completion".into(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason,
        }],
        counter,
    })
}
