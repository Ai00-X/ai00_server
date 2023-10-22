use std::{
    fs::{File, Metadata},
    io::{BufReader, Cursor, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use anyhow::Result;
use axum::{extract::State, Json};
use memmap::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ThreadState;

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
    is_sha: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileInfo {
    name: String,
    size: u64,
    sha: String,
}

#[derive(Debug, Default, Clone, Serialize)]
#[serde(untagged)]
pub enum FileInfoResponse {
    Ok(Vec<FileInfo>),
    #[default]
    Err,
}

/// `/api/dir`, `/api/ls`.
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

#[derive(Debug, Clone, Deserialize)]
pub struct UnzipRequest {
    zip_path: PathBuf,
    target_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum UnzipResponse {
    Ok,
    Err,
}

/// `/api/unzip`.
pub async fn unzip(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<UnzipRequest>,
) -> Json<UnzipResponse> {
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
        Ok(_) => Json(UnzipResponse::Ok),
        Err(err) => {
            log::error!("failed to unzip: {}", err);
            Json(UnzipResponse::Err)
        }
    }
}
