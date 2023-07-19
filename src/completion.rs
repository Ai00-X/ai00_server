use axum::{extract::State, Json};
use serde::Deserialize;

use crate::ThreadRequest;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CompletionRequest {
    prompt: Vec<String>,
    max_tokens: usize,
    stop: Vec<String>,
    temperature: f32,
    top_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
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

pub async fn completions(
    State(state): State<flume::Sender<ThreadRequest>>,
    Json(request): Json<CompletionRequest>,
) {
}
