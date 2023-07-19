use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::ThreadRequest;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
enum Role {
    System,
    #[default]
    User,
    Assistant,
}

#[derive(Default, Debug, Clone, Deserialize)]
struct ChatRecord {
    role: Role,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ChatRequest {
    messages: Vec<ChatRecord>,
    max_tokens: usize,
    stop: Vec<String>,
    temperature: f32,
    top_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    stream: bool,
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
            stream: false,
        }
    }
}

pub async fn chat(
    State(state): State<flume::Sender<ThreadRequest>>,
    Json(request): Json<ChatRequest>,
) {
}
