use axum::{extract::State, Json};
use serde::Serialize;

use crate::{request_info, ReloadRequest, RuntimeInfo, ThreadRequest, ThreadState};

#[derive(Debug, Serialize)]
pub struct ModelChoice {
    pub object: String,
    pub id: String,
}

#[derive(Debug, Serialize)]
pub struct ModelResponse {
    pub data: Vec<ModelChoice>,
}

pub async fn models(State(ThreadState(sender)): State<ThreadState>) -> Json<ModelResponse> {
    let model_name = request_info(sender.clone())
        .map(|info| info.reload.path)
        .and_then(|path| path.file_name().map(|name| name.to_os_string()))
        .and_then(|name| name.into_string().ok())
        .unwrap_or_default();

    Json(ModelResponse {
        data: vec![ModelChoice {
            object: "models".into(),
            id: model_name,
        }],
    })
}

#[derive(Debug, Default, Serialize)]
#[serde(untagged)]
pub enum InfoResponse {
    Some(RuntimeInfo),
    #[default]
    None,
}

pub async fn info(State(ThreadState(sender)): State<ThreadState>) -> Json<InfoResponse> {
    match request_info(sender) {
        Some(info) => Json(InfoResponse::Some(info)),
        None => Json(InfoResponse::None),
    }
}

pub async fn load(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<ReloadRequest>,
) -> Json<InfoResponse> {
    let _ = sender.send(ThreadRequest::Reload(request));
    info(State(ThreadState(sender))).await
}
