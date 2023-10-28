use std::{
    fs::{File, Metadata},
    io::{BufReader, Cursor, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use axum::{extract::State, Json};
use memmap::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{config::Config, ThreadState};

fn compute_sha(path: impl AsRef<Path>, meta: &Metadata) -> Result<String> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();

    if meta.len() > 10_000_000 {
        let segment_size = meta.len() / 10;
        for i in 0..10 {
            let mut segment_buffer = vec![0; 1_000_000];
            reader.seek(SeekFrom::Start(i * segment_size))?;
            reader.read_exact(&mut segment_buffer)?;
            buffer.extend(segment_buffer);
        }
    } else {
        reader.read_to_end(&mut buffer)?;
    }

    let mut sha = Sha256::new();
    sha.update(&buffer);
    let result = sha.finalize();

    Ok(format!("{:x}", result))
}

#[derive(Debug, Clone, Deserialize)]
pub struct FileInfoRequest {
    path: PathBuf,
    #[serde(default)]
    is_sha: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileInfo {
    name: String,
    size: u64,
    sha: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum FileInfoResponse {
    Err,
    #[serde(untagged)]
    Ok(Vec<FileInfo>),
}

/// `/api/files/dir`, `/api/files/ls`.
pub async fn dir(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<FileInfoRequest>,
) -> Json<FileInfoResponse> {
    match std::fs::read_dir(request.path) {
        Ok(dir) => {
            let files = dir
                .filter_map(|x| x.ok())
                .filter(|x| x.path().is_file())
                .filter_map(|x| {
                    let path = x.path();
                    let meta = x.metadata().ok()?;

                    let name = x.file_name().to_string_lossy().into();
                    let sha = request
                        .is_sha
                        .then(|| compute_sha(path, &meta).ok())
                        .flatten()
                        .unwrap_or_default();

                    Some(FileInfo {
                        name,
                        size: meta.len(),
                        sha,
                    })
                })
                .collect();
            Json(FileInfoResponse::Ok(files))
        }
        Err(err) => {
            log::error!("failed to read directory: {}", err);
            Json(FileInfoResponse::Err)
        }
    }
}

/// `/api/models/list`.
pub async fn models(state: State<ThreadState>) -> Json<FileInfoResponse> {
    dir(
        state,
        Json(FileInfoRequest {
            path: "assets/models".into(),
            is_sha: true,
        }),
    )
    .await
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum FileOpResponse {
    Ok,
    Err,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UnzipRequest {
    zip_path: PathBuf,
    target_dir: PathBuf,
}

/// `/api/files/unzip`.
pub async fn unzip(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<UnzipRequest>,
) -> Json<FileOpResponse> {
    let unzip = move || -> Result<()> {
        if Path::new(&request.target_dir).exists() {
            std::fs::remove_dir_all(&request.target_dir)?;
        }
        std::fs::create_dir_all(&request.target_dir)?;

        let file = File::open(&request.zip_path)?;
        let map = unsafe { Mmap::map(&file)? };
        zip_extract::extract(Cursor::new(&map), &request.target_dir, false)?;

        Ok(())
    };

    match unzip() {
        Ok(_) => Json(FileOpResponse::Ok),
        Err(err) => {
            log::error!("failed to unzip: {}", err);
            Json(FileOpResponse::Err)
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoadRequest {
    path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum LoadResponse {
    Err,
    #[serde(untagged)]
    Ok(Config),
}

/// `/api/files/config/load`.
pub async fn load_config(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<LoadRequest>,
) -> Json<LoadResponse> {
    match crate::load_config(request.path) {
        Ok(config) => Json(LoadResponse::Ok(config)),
        Err(err) => {
            log::error!("failed to load config: {}", err);
            Json(LoadResponse::Err)
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SaveRequest {
    path: PathBuf,
    config: Config,
}

/// `/api/files/config/save`.
pub async fn save_config(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<SaveRequest>,
) -> Json<FileOpResponse> {
    let write = || -> Result<()> {
        let mut file = File::create(&request.path)?;
        let buf = toml::to_string(&request.config)?.into_bytes();
        file.write_all(&buf)?;
        Ok(())
    };

    match request.path.extension() {
        Some(ext) if ext == "toml" => match write() {
            Ok(_) => Json(FileOpResponse::Ok),
            Err(err) => {
                log::error!("failed to save config: {err}");
                Json(FileOpResponse::Err)
            }
        },
        _ => {
            log::error!(
                "failed to save config: file path {} is not toml",
                request.path.to_string_lossy()
            );
            Json(FileOpResponse::Err)
        }
    }
}
