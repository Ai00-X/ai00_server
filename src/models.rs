use axum::{extract::State, Json};
use serde::Serialize;

use crate::{ReloadRequest, RuntimeInfo, ThreadRequest, ThreadState};

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
    let model_name = {
        let (info_sender, info_receiver) = flume::bounded(1);
        let _ = sender.send(ThreadRequest::Info(info_sender));
        let info = info_receiver.recv().unwrap();
        info.reload
            .path
            .into_os_string()
            .into_string()
            .unwrap_or_default()
    };

    Json(ModelResponse {
        data: vec![ModelChoice {
            object: "models".into(),
            id: model_name,
        }],
    })
}

pub async fn info(State(ThreadState(sender)): State<ThreadState>) -> Json<RuntimeInfo> {
    let info = {
        let (info_sender, info_receiver) = flume::bounded(1);
        let _ = sender.send(ThreadRequest::Info(info_sender));
        info_receiver.recv().unwrap()
    };
    Json(info)
}

pub async fn load(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<ReloadRequest>,
) -> Json<RuntimeInfo> {
    let _ = sender.send(ThreadRequest::Reload(request));
    info(State(ThreadState(sender))).await
}
