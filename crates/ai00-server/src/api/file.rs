use std::{
    fs::{File, Metadata},
    io::{BufReader, Cursor, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use salvo::{macros::Extractible, prelude::*};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use web_rwkv::runtime::{loader::Loader, model::ModelInfo};

use crate::check_path_permitted;

const PERMITTED_PATHS: [&str; 4] = [
    "assets/models",
    "assets/tokenizer",
    "assets/configs",
    "assets/www",
];
const UNZIP_PATHS: [&str; 2] = ["assets/unzip", "assets/temp"];

fn check_path(path: impl AsRef<Path>) -> Result<()> {
    check_path_permitted(path, &PERMITTED_PATHS)
}

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

#[derive(Debug, Clone, Deserialize, Extractible)]
#[salvo(extract(default_source(from = "body")))]
struct FileInfoRequest {
    path: PathBuf,
    #[serde(default)]
    is_sha: bool,
}

#[derive(Debug, Clone, Serialize)]
struct FileInfo {
    path: PathBuf,
    name: String,
    size: u64,
    sha: String,
    info: Option<ModelInfo>,
}

#[derive(Debug, Clone, Deserialize, Extractible)]
#[salvo(extract(default_source(from = "body")))]
struct UnzipRequest {
    #[serde(alias = "zip_path")]
    path: PathBuf,
    #[serde(alias = "target_dir")]
    output: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Extractible)]
struct LoadRequest {
    path: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Extractible)]
struct SaveRequest {
    path: PathBuf,
    config: crate::config::Config,
}

async fn dir_inner(
    _depot: &mut Depot,
    Json(request): Json<FileInfoRequest>,
) -> Result<(StatusCode, Vec<FileInfo>), StatusCode> {
    match std::fs::read_dir(request.path) {
        Ok(path) => {
            let files = path
                .filter_map(|x| x.ok())
                .filter(|x| x.path().is_file())
                .filter_map(|x| {
                    let path = x.path();
                    let meta = x.metadata().ok()?;

                    let name = x.file_name().to_string_lossy().into();
                    let sha = request
                        .is_sha
                        .then(|| compute_sha(&path, &meta).ok())
                        .flatten()
                        .unwrap_or_default();

                    let file = File::open(&path).ok()?;
                    let data = unsafe { Mmap::map(&file) }.ok()?;
                    let info = SafeTensors::deserialize(&data)
                        .map_err(Into::into)
                        .and_then(|model| Loader::info(&model))
                        .ok();

                    Some(FileInfo {
                        path,
                        name,
                        size: meta.len(),
                        sha,
                        info,
                    })
                })
                .collect_vec();
            Ok((StatusCode::OK, files))
        }
        Err(err) => {
            log::error!("failed to read directory: {}", err);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

#[handler]
pub async fn dir(depot: &mut Depot, req: &mut Request, res: &mut Response) {
    let request = match req.parse_json::<FileInfoRequest>().await {
        Ok(t) => t,
        Err(err) => {
            res.status_code(StatusCode::INTERNAL_SERVER_ERROR);
            res.render(Text::Plain(err.to_string()));
            return;
        }
    };
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        res.status_code(StatusCode::FORBIDDEN);
        res.render("ERROR");
        return;
    }

    match dir_inner(depot, Json(request)).await {
        Ok((status, files)) => {
            res.status_code(status);
            res.render(Json(files));
        }
        Err(status) => {
            res.status_code(status);
            res.render("ERROR");
        }
    }
}

/// `/api/models/list`.
#[handler]
pub async fn models(depot: &mut Depot, res: &mut Response) {
    let config = depot.obtain::<crate::config::Config>().unwrap();
    let request = FileInfoRequest {
        path: config.model.path.clone(),
        is_sha: true,
    };
    match dir_inner(depot, Json(request)).await {
        Ok((status, files)) => {
            res.status_code(status);
            res.render(Json(files));
        }
        Err(status) => {
            res.status_code(status);
            res.render("ERROR");
        }
    }
}

#[handler]
pub async fn unzip(_depot: &mut Depot, request: UnzipRequest) -> StatusCode {
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        return StatusCode::FORBIDDEN;
    }
    if let Err(err) = check_path_permitted(&request.output, &UNZIP_PATHS) {
        log::error!("check path failed: {}", err);
        return StatusCode::FORBIDDEN;
    }

    let unzip_inner = move || -> Result<()> {
        if Path::new(&request.output).exists() {
            std::fs::remove_dir_all(&request.output)?;
        }
        std::fs::create_dir_all(&request.output)?;

        let file = File::open(&request.path)?;
        let map = unsafe { Mmap::map(&file)? };
        zip_extract::extract(Cursor::new(&map), &request.output, false)?;

        Ok(())
    };

    match unzip_inner() {
        Ok(_) => StatusCode::OK,
        Err(err) => {
            log::error!("failed to unzip: {}", err);
            StatusCode::NOT_FOUND
        }
    }
}

/// `/api/files/config/load`.
#[handler]
pub async fn load_config(_depot: &mut Depot, request: LoadRequest, response: &mut Response) {
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        response.status_code(StatusCode::FORBIDDEN);
        response.render("FORBIDDEN");
    }
    match crate::load_config(request.path).await {
        Ok(config) => {
            response.status_code(StatusCode::OK);
            response.render(Json(config));
        }
        Err(err) => {
            log::error!("failed to load config: {}", err);
            // Err(StatusCode::NOT_FOUND)
            response.status_code(StatusCode::NOT_FOUND);
            response.render("NOT_FOUND");
        }
    }
}

/// `/api/files/config/save`.
#[handler]
pub async fn save_config(_depot: &mut Depot, request: SaveRequest) -> StatusCode {
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        return StatusCode::FORBIDDEN;
    }

    let write = || -> Result<()> {
        let mut file = File::create(&request.path)?;
        let buf = toml::to_string(&request.config)?.into_bytes();
        file.write_all(&buf)?;
        Ok(())
    };

    match request.path.extension() {
        Some(ext) if ext == "toml" => match write() {
            Ok(_) => StatusCode::OK,
            Err(err) => {
                log::error!("failed to save config: {err:#?}");
                StatusCode::INTERNAL_SERVER_ERROR
            }
        },
        _ => {
            log::error!(
                "failed to save config: file path {} is not toml",
                request.path.to_string_lossy()
            );
            StatusCode::FORBIDDEN
        }
    }
}
