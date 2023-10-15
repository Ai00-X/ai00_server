use std::{
    fs::{self, File},
    io::Cursor,
    path::Path,
    path::PathBuf,
};

use anyhow::Result;
use axum::{extract::State, Json};
use memmap::Mmap;
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
    pub name: String,
    pub size: u64,
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
                    name: name.to_string_lossy().into(),
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

#[derive(Debug, Clone, Deserialize)]
pub struct UnzipRequest {
    pub path: PathBuf,
    pub target: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum UnzipResponse {
    Ok,
    Err,
}

pub async fn unzip(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<UnzipRequest>,
) -> Json<UnzipResponse> {
    let unzip = move || -> Result<()> {
        if Path::new(&request.target).exists() {
            fs::remove_dir_all(&request.target)?;
        }
        fs::create_dir_all(&request.target)?;

        let file = File::open(&request.path)?;
        let map = unsafe { Mmap::map(&file)? };
        zip_extract::extract(Cursor::new(&map), &request.target, false)?;

        Ok(())
    };

    match unzip() {
        Ok(_) => Json(UnzipResponse::Ok),
        Err(err) => {
            log::error!("failed to unzip: {}", err);
            Json(UnzipResponse::Err)
        }
    }
}
