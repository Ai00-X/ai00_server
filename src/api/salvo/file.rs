use std::{
    fs::{File, Metadata},
    io::{BufReader, Cursor, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use anyhow::{bail, Result};
use itertools::Itertools;
use memmap2::Mmap;
use salvo::{handler, macros::Extractible, Depot};
use salvo::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{config::Config, ThreadState};

const PERMITTED_PATHS: [&str; 4] = [
    "assets/models",
    "assets/tokenizer",
    "assets/configs",
    "assets/www",
];
const UNZIP_PATHS: [&str; 2] = ["assets/unzip", "assets/temp"];

fn check_path_permitted(path: impl AsRef<Path>, permitted: &[&str]) -> Result<()> {
    let current_path = std::env::current_dir()?;
    for sub in permitted {
        let permitted = std::fs::canonicalize(current_path.join(sub))?;
        let path = std::fs::canonicalize(path.as_ref())?;
        if path.starts_with(permitted) {
            return Ok(());
        }
    }
    bail!("path not valid");
}

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

#[derive(Debug, Clone, Deserialize, Serialize, Extractible)]
pub struct FileInfoRequest {
    path: PathBuf,
    #[serde(default)]
    is_sha: bool,
}

#[derive(Debug, Clone, Deserialize, Extractible, Serialize)]
#[salvo(extract(default_source(from = "body")))]
pub struct FileInfo {
    name: String,
    size: u64,
    sha: String,
}

pub async fn salvo_dir_inner(
    depot: &mut Depot,
    salvo::prelude::Json(request): salvo::prelude::Json<FileInfoRequest>,
) -> Result<(salvo::prelude::StatusCode, Vec<FileInfo>), salvo::prelude::StatusCode> {
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        return Err(StatusCode::FORBIDDEN);
    }
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
                .collect_vec();
            Ok((StatusCode::OK, files))
        }
        Err(err) => {
            log::error!("failed to read directory: {}", err);
            Err(StatusCode::NOT_FOUND)
        }
    }
}
/// `/api/models/list`.
#[handler]
pub async fn salvo_models(depot: &mut Depot, res: &mut salvo::http::Response) {
    let state = depot.obtain::<ThreadState>().unwrap();

    let request = FileInfoRequest {
        path: "assets/models".into(),
        is_sha: true,
    };
    match salvo_dir_inner(depot, salvo::prelude::Json(request)).await {
        Ok((status, files)) => {
            res.status_code(status);
            res.render(salvo::prelude::Json(files));
        },
        Err(status) => {
            res.status_code(status);
            res.render("ERROR");
        }
    }
}

#[handler]
pub async fn salvo_dir(depot: &mut Depot, 
            req: &mut salvo::http::Request,  
            res: &mut salvo::http::Response) {
    let request = match req.parse_json::<FileInfoRequest>().await {
        Ok(t) => t,
        Err(err) => {
            res.status_code(salvo::prelude::StatusCode::INTERNAL_SERVER_ERROR);
            res.render(Text::Plain(err.to_string()));
            return;
        }
    };

    match salvo_dir_inner(depot, salvo::prelude::Json(request)).await {
        Ok((status, files)) => {
            res.status_code(status);
            res.render(salvo::prelude::Json(files));
        },
        Err(status) => {
            res.status_code(status);
            res.render("ERROR");
        }
    }
}


#[derive(Debug, Clone, Deserialize, Serialize, Extractible)]
#[salvo(extract(default_source(from = "body")))]
pub struct UnzipRequest {
    #[serde(alias = "zip_path")]
    path: PathBuf,
    #[serde(alias = "target_dir")]
    output: PathBuf,
}



#[handler]
pub async fn salvo_unzip(
    depot: &mut Depot,
    request: UnzipRequest,
) -> salvo::prelude::StatusCode {
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        return salvo::prelude::StatusCode::FORBIDDEN;
    }
    if let Err(err) = check_path_permitted(&request.output, &UNZIP_PATHS) {
        log::error!("check path failed: {}", err);
        return salvo::prelude::StatusCode::FORBIDDEN;
    }

    let unzip = move || -> Result<()> {
        if Path::new(&request.output).exists() {
            std::fs::remove_dir_all(&request.output)?;
        }
        std::fs::create_dir_all(&request.output)?;

        let file = File::open(&request.path)?;
        let map = unsafe { Mmap::map(&file)? };
        zip_extract::extract(Cursor::new(&map), &request.output, false)?;

        Ok(())
    };

    match unzip() {
        Ok(_) => salvo::prelude::StatusCode::OK,
        Err(err) => {
            log::error!("failed to unzip: {}", err);
            salvo::prelude::StatusCode::NOT_FOUND
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Extractible)]
pub struct LoadRequest {
    path: PathBuf,
}


/// `/api/files/config/load`.
#[handler]
pub async fn salvo_load_config(
    depot: &mut Depot,
    request: LoadRequest,
    response: &mut salvo::prelude::Response
) {
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        response.status_code(salvo::prelude::StatusCode::FORBIDDEN);
        response.render("FORBIDDEN");
    }
    match crate::load_config(request.path) {
        Ok(config) => {
            response.status_code(salvo::prelude::StatusCode::OK);
            response.render(salvo::prelude::Json(config));
        },
        Err(err) => {
            log::error!("failed to load config: {}", err);
            // Err(StatusCode::NOT_FOUND)
            response.status_code(salvo::prelude::StatusCode::NOT_FOUND);
            response.render("NOT_FOUND");
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Extractible)]
pub struct SaveRequest {
    path: PathBuf,
    config: Config,
}

/// `/api/files/config/save`.
#[handler]
pub async fn salvo_save_config(
    depot: &mut Depot,
    request: SaveRequest
) -> salvo::prelude::StatusCode {
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        return salvo::prelude::StatusCode::FORBIDDEN;
    }

    let write = || -> Result<()> {
        let mut file = File::create(&request.path)?;
        let buf = toml::to_string(&request.config)?.into_bytes();
        file.write_all(&buf)?;
        Ok(())
    };

    match request.path.extension() {
        Some(ext) if ext == "toml" => match write() {
            Ok(_) => salvo::prelude::StatusCode::OK,
            Err(err) => {
                log::error!("failed to save config: {err}");
                salvo::prelude::StatusCode::INTERNAL_SERVER_ERROR
            }
        },
        _ => {
            log::error!(
                "failed to save config: file path {} is not toml",
                request.path.to_string_lossy()
            );
            salvo::prelude::StatusCode::FORBIDDEN
        }
    }
}
