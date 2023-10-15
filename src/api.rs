/*------------------------------------------------------------------
 Import necessary modules and libraries
-------------------------------------------------------------------*/
use crate::{request_info, ReloadRequest, RuntimeInfo, ThreadRequest, ThreadState};
use axum::{extract::State, Json};
use memmap::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufReader, Cursor, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

/*------------------------------------------------------------------
/api/openai/models
/api/openai/v1/models
-------------------------------------------------------------------*/

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
    let model_name = request_info(sender)
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

/*------------------------------------------------------------------
/api/models/info
-------------------------------------------------------------------*/

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

/*------------------------------------------------------------------
/api/models/load
-------------------------------------------------------------------*/

pub async fn load(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<ReloadRequest>,
) -> Json<InfoResponse> {
    let _ = sender.send(ThreadRequest::Reload(request));
    info(State(ThreadState(sender))).await
}

/*------------------------------------------------------------------
/api/models/list
-------------------------------------------------------------------*/

#[derive(Debug, Clone, Serialize)]
pub struct ModelsList {
    pub name: String,
    pub size: u64,
    pub sha: String,
}

#[derive(Debug, Default, Clone, Serialize)]
#[serde(untagged)]
pub enum ModelsListResponse {
    Accepted(Vec<ModelsList>),
    #[default]
    Denied,
}

pub async fn models_list(State(ThreadState(_)): State<ThreadState>) -> Json<ModelsListResponse> {
    let models = match std::fs::read_dir("assets/models") {
        Ok(dir) => dir
            .filter_map(|x| x.ok())
            .filter(|x| x.path().is_file())
            .filter_map(|x| {
                let path = x.path();
                let file = File::open(&path).ok()?;
                let mut reader = BufReader::new(file);
                let meta = x.metadata().ok()?;
                let mut buffer = Vec::new();

                if meta.len() > 10_000_000 {
                    let segment_size = meta.len() / 10;
                    for i in 0..10 {
                        let mut segment_buffer = vec![0; 1_000_000];
                        reader.seek(SeekFrom::Start(i * segment_size)).ok()?;
                        reader.read_exact(&mut segment_buffer).ok()?;
                        buffer.extend(segment_buffer);
                    }
                } else {
                    reader.read_to_end(&mut buffer).ok()?;
                }

                let mut hasher = Sha256::new();
                hasher.update(&buffer);
                let result = hasher.finalize();
                let sha = format!("{:x}", result);

                Some(ModelsList {
                    name: x.file_name().to_string_lossy().into_owned(),
                    size: meta.len(),
                    sha,
                })
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    Json(ModelsListResponse::Accepted(models))
}
/*------------------------------------------------------------------
/api/files/dir
-------------------------------------------------------------------*/

#[derive(Debug, Clone, Deserialize)]
pub struct FileInfoRequest {
    pub path: PathBuf,
    pub is_sha: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileInfo {
    pub name: String,
    pub size: u64,
    pub sha: String,
}

#[derive(Debug, Default, Clone, Serialize)]
#[serde(untagged)]
pub enum FileInfoResponse {
    Accepted(Vec<FileInfo>),
    #[default]
    Denied,
}

pub async fn dir(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<FileInfoRequest>,
) -> Json<FileInfoResponse> {
    if request.path.is_dir() && request.path.starts_with("./") {
        let files = match std::fs::read_dir(request.path) {
            Ok(dir) => dir
                .filter_map(|x| x.ok())
                .filter(|x| x.path().is_file())
                .filter_map(|x| {
                    let mut sha = "".into();
                    if request.is_sha {
                        let path = x.path();
                        let file = File::open(&path).ok()?;
                        let mut reader = BufReader::new(file);
                        let meta = x.metadata().ok()?;
                        let mut buffer = Vec::new();

                        if meta.len() > 10_000_000 {
                            let segment_size = meta.len() / 10;
                            for i in 0..10 {
                                let mut segment_buffer = vec![0; 1_000_000];
                                reader.seek(SeekFrom::Start(i * segment_size)).ok()?;
                                reader.read_exact(&mut segment_buffer).ok()?;
                                buffer.extend(segment_buffer);
                            }
                        } else {
                            reader.read_to_end(&mut buffer).ok()?;
                        }

                        let mut hasher = Sha256::new();
                        hasher.update(&buffer);
                        let result = hasher.finalize();
                        sha = format!("{:x}", result);
                    }

                    let meta = x.metadata().ok()?;
                    Some(FileInfo {
                        name: x.file_name().to_string_lossy().into_owned(),
                        size: meta.len(),
                        sha,
                    })
                })
                .collect(),
            Err(_) => Vec::new(),
        };
        Json(FileInfoResponse::Accepted(files))
    } else {
        Json(FileInfoResponse::Denied)
    }
}

/*------------------------------------------------------------------
/api/files/unzip
-------------------------------------------------------------------*/

#[derive(Debug, Clone, Deserialize)]
pub struct UnzipRequest {
    pub target_dir: PathBuf,
    pub zip_path: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct UnzipResponse {
    pub state: String,
}

pub async fn unzip(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<UnzipRequest>,
) -> Json<UnzipResponse> {
    if Path::new(&request.target_dir).exists() {
        // If exists, remove it
        fs::remove_dir_all(&request.target_dir).unwrap();
    }
    fs::create_dir_all(&request.target_dir).unwrap();

    let file = File::open(&request.zip_path).unwrap();
    let map = unsafe { Mmap::map(&file).unwrap() };
    match zip_extract::extract(Cursor::new(&map), &request.target_dir, false) {
        Ok(_) => Json(UnzipResponse { state: "OK".into() }),
        Err(_) => Json(UnzipResponse {
            state: "ERR".into(),
        }),
    }
}
