use std::time::Duration;

use axum::{extract::State, Json};
use serde::Serialize;
use web_rwkv::model::ModelInfo;

use crate::{request_info, ReloadRequest, RuntimeInfo, ThreadRequest, ThreadState};

#[derive(Debug, Clone, Serialize)]
pub struct LoadResponse {
    reload: ReloadRequest,
    model: ModelInfo,
}

/// `/api/models/info`.
pub async fn info(State(ThreadState(sender)): State<ThreadState>) -> Json<LoadResponse> {
    let RuntimeInfo { reload, model, .. } = request_info(sender, Duration::from_secs(1)).await;
    Json(LoadResponse { reload, model })
}

/// `/api/models/load`.
pub async fn load(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<ReloadRequest>,
) -> Json<LoadResponse> {
    let _ = sender.send(ThreadRequest::Reload(request));
    info(State(ThreadState(sender))).await
}
