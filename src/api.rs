use std::{ffi::OsString, path::PathBuf};

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use web_rwkv::model::ModelInfo;

use crate::{request_info, ReloadRequest, RuntimeInfo, ThreadRequest, ThreadState};

#[derive(Debug, Serialize)]
struct ModelChoice {
    object: String,
    id: String,
}

#[derive(Debug, Serialize)]
pub struct ModelResponse {
    data: Vec<ModelChoice>,
}

pub async fn models(State(ThreadState(sender)): State<ThreadState>) -> Json<ModelResponse> {
    let info = request_info(sender);
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    Json(ModelResponse {
        data: vec![ModelChoice {
            object: "models".into(),
            id: model_name,
        }],
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct InfoResponse {
    reload: ReloadRequest,
    model: ModelInfo,
}

pub async fn info(State(ThreadState(sender)): State<ThreadState>) -> Json<InfoResponse> {
    let RuntimeInfo { reload, model, .. } = request_info(sender);
    Json(InfoResponse { reload, model })
}

pub async fn load(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<ReloadRequest>,
) -> Json<InfoResponse> {
    let _ = sender.send(ThreadRequest::Reload(request));
    info(State(ThreadState(sender))).await
}

#[derive(Debug, Clone, Deserialize)]
pub struct FileInfoRequest {
    path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileInfo {
    name: OsString,
    size: u64,
}

#[derive(Debug, Default, Clone, Serialize)]
#[serde(untagged)]
pub enum FileInfoResponse {
    Accepted(Vec<FileInfo>),
    #[default]
    Denied,
}

pub async fn files(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<FileInfoRequest>,
) -> Json<FileInfoResponse> {
    if request.path.is_dir() && request.path.starts_with("assets/") {
        let files = match std::fs::read_dir(request.path) {
            Ok(dir) => dir
                .filter_map(|x| x.ok())
                .filter(|x| x.path().is_file())
                .filter_map(|x| Some((x.file_name(), x.metadata().ok()?)))
                .map(|(name, meta)| FileInfo {
                    name,
                    size: meta.len(),
                })
                .collect(),
            Err(_) => Vec::new(),
        };
        Json(FileInfoResponse::Accepted(files))
    } else {
        Json(FileInfoResponse::Denied)
    }
}
