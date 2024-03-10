use std::{
    fs::File,
    io::{BufReader, Cursor, Read},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
};

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use memmap2::Mmap;
use tower_http::{cors::CorsLayer, services::ServeDir};

use crate::{
    api::oai,
    middleware::{model_route, ThreadRequest, ThreadState},
};

mod api;
mod config;
mod middleware;
mod run;
mod sampler;

fn load_web(path: impl AsRef<Path>, target: &Path) -> Result<()> {
    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };
    zip_extract::extract(Cursor::new(&map), target, false)?;
    Ok(())
}

fn load_plugin(path: impl AsRef<Path>, target: &Path, name: &String) -> Result<()> {
    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };
    let root = target.join("plugins");
    if !root.exists() {
        std::fs::create_dir(&root)?;
    }
    let dir = root.join(name);
    std::fs::create_dir(&dir)?;
    zip_extract::extract(Cursor::new(&map), &dir, false)?;
    Ok(())
}

fn load_config(path: impl AsRef<Path>) -> Result<config::Config> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(toml::from_str(&contents)?)
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short, value_name = "FILE")]
    config: Option<PathBuf>,
    #[arg(long, short)]
    ip: Option<IpAddr>,
    #[arg(long, short, default_value_t = 65530)]
    port: u16,
}

#[tokio::main]
async fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ai00_server", log::LevelFilter::Trace)
        .init()
        .unwrap();

    let args = Args::parse();
    let (sender, receiver) = flume::unbounded::<ThreadRequest>();

    let request = {
        let path = args
            .config
            .clone()
            .unwrap_or("assets/configs/Config.toml".into());
        log::info!("reading config {}...", path.to_string_lossy());
        load_config(path).expect("load config failed").into()
    };

    tokio::task::spawn_blocking(move || model_route(receiver));
    let _ = sender.send(ThreadRequest::Reload {
        request,
        sender: None,
    });

    let serve_path = {
        let path = tempfile::tempdir()
            .expect("create temp dir failed")
            .into_path();
        load_web("assets/www/index.zip", &path).expect("load frontend failed");
        path
    };

    // create `assets/www/plugins` if it doesn't exist
    if !Path::new("assets/www/plugins").exists() {
        std::fs::create_dir("assets/www/plugins").expect("create plugins dir failed");
    }

    // extract and load all plugins under `assets/www/plugins`
    match std::fs::read_dir("assets/www/plugins") {
        Ok(dir) => dir
            .filter_map(|x| x.ok())
            .filter(|x| x.path().is_file())
            .filter(|x| x.path().extension().is_some_and(|ext| ext == "zip"))
            .filter(|x| x.path().file_stem().is_some_and(|stem| stem != "api"))
            .for_each(|x| {
                let name = x
                    .path()
                    .file_stem()
                    .expect("this cannot happen")
                    .to_string_lossy()
                    .into();
                match load_plugin(x.path(), &serve_path, &name) {
                    Ok(_) => log::info!("loaded plugin {}", name),
                    Err(err) => log::error!("failed to load plugin {}, {}", name, err),
                }
            }),
        Err(err) => {
            log::error!("failed to read plugin directory: {}", err);
        }
    };

    let app = Router::new()
        .route("/api/adapters", get(api::adapters))
        .route("/api/files/unzip", post(api::unzip))
        .route("/api/files/dir", post(api::dir))
        .route("/api/files/ls", post(api::dir))
        .route("/api/files/config/load", post(api::load_config))
        .route("/api/files/config/save", post(api::save_config))
        .route("/api/models/list", get(api::models))
        .route("/api/models/info", get(api::info))
        .route("/api/models/state", get(api::state))
        .route("/api/models/load", post(api::load))
        .route("/api/models/unload", get(api::unload))
        .route("/api/oai/models", get(oai::models))
        .route("/api/oai/v1/models", get(oai::models))
        .route("/api/oai/completions", post(oai::completions))
        .route("/api/oai/v1/completions", post(oai::completions))
        .route("/api/oai/chat/completions", post(oai::chat_completions))
        .route("/api/oai/v1/chat/completions", post(oai::chat_completions))
        .route("/api/oai/embeddings", post(oai::embeddings))
        .route("/api/oai/v1/embeddings", post(oai::embeddings))
        .fallback_service(ServeDir::new(serve_path))
        .layer(CorsLayer::permissive())
        .with_state(ThreadState(sender));
    let addr = SocketAddr::new(
        args.ip.unwrap_or(IpAddr::from(Ipv4Addr::UNSPECIFIED)),
        args.port,
    );
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    log::info!("server started at {addr}");
    axum::serve(listener, app).await.unwrap();
}
