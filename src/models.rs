use axum::{extract::State, Json};
use serde::Serialize;

use crate::{ReloadRequest, ThreadRequest, ThreadState};

#[derive(Debug, Serialize)]
pub struct ModelChoice {
    pub object: String,
    pub id: String,
}

#[derive(Debug, Serialize)]
pub struct ModelResponse {
    pub data: Vec<ModelChoice>,
}

pub async fn models(
    State(ThreadState { model_name, .. }): State<ThreadState>,
) -> Json<ModelResponse> {
    Json(ModelResponse {
        data: vec![ModelChoice {
            object: "models".into(),
            id: model_name.read().unwrap().clone(),
        }],
    })
}

pub async fn load(
    State(ThreadState { sender, .. }): State<ThreadState>,
    Json(request): Json<ReloadRequest>,
) {
    let _ = sender.send(ThreadRequest::Reload(request));
}
