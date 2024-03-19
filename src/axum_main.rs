use crate::{
    api::{self, oai},
    load_config, load_plugin, load_web,
    middleware::{model_route, ThreadRequest, ThreadState},
    Args,
};
use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::Path,
};
use tower_http::{cors::CorsLayer, services::ServeDir};

pub async fn axum_main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ai00_server", log::LevelFilter::Info)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
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
