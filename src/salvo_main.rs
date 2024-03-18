use std::{
    fs::File,
    io::{BufReader, Cursor, Read},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
  };
  
  use anyhow::Result;
  
  use clap::Parser;
  use memmap2::Mmap;
  use salvo::Router;
  use salvo::prelude::*;
  use salvo::http::Method;
  use salvo::affix;
  use salvo::cors::Cors;
  use salvo::serve_static::StaticDir;
  use salvo::logging::Logger;
  // use tower_http::{cors::CorsLayer, services::ServeDir};
  use salvo::cors::AllowOrigin;
  use crate::{load_config, load_web, load_plugin, Args};
  use crate::{
    api::{self, oai}, config, middleware::{model_route, ThreadRequest, ThreadState}
  };

  
  #[cfg(feature="salvo-api")]
  pub async fn salvo_main() {
  
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
  
    let cors = Cors::new()
        .allow_origin(AllowOrigin::any())
        .allow_methods(vec![Method::GET, Method::POST, Method::DELETE])
        .allow_headers("authorization")
        .into_handler();
    
  
    let app = Router::new()
        //.hoop(CorsLayer::permissive())
        .hoop(Logger::new())
        .hoop(affix::inject(ThreadState(sender)))
        .hoop(cors)
        .push(
          Router::with_path("/api/adapters").get(api::salvo::salvo_info)
        )
        .push(
          Router::with_path("/api/models/info").get(api::salvo::salvo_info)
        ).push(
          Router::with_path("/api/models/load").post(api::salvo::salvo_load)
        ).push(
          Router::with_path("/api/models/unload").get(api::salvo::salvo_unload)
        ).push(
          Router::with_path("/api/models/state").get(api::salvo::salvo_state)
        ).push(
          Router::with_path("/api/models/list").get(api::salvo::salvo_models)
        ).push(
          Router::with_path("/api/files/unzip").post(api::salvo::salvo_unzip)
        ).push(
          Router::with_path("/api/files/dir").post(api::salvo::salvo_dir)
        ).push(
          Router::with_path("/api/files/ls").post(api::salvo::salvo_dir)
        ).push(
          Router::with_path("/api/files/config/load").post(api::salvo::salvo_load_config)
        ).push(
          Router::with_path("/api/files/config/save").post(api::salvo::salvo_save_config)
        ).push(
          Router::with_path("/api/oai/models").get(api::salvo::oai::salvo_oai_models)
        ).push(
          Router::with_path("/api/oai/v1/models").get(api::salvo::oai::salvo_oai_models)
        ).push(
          Router::with_path("/api/oai/completions").post(api::salvo::oai::salvo_oai_completions)
        ).push(
          Router::with_path("/api/oai/v1/completions").post(api::salvo::oai::salvo_oai_completions)
        ).push(
          Router::with_path("/api/oai/chat/completions").post(api::salvo::oai::salvo_oai_chat_completions)
        ).push(
          Router::with_path("/api/oai/v1/chat/completions").post(api::salvo::oai::salvo_oai_chat_completions)
        ).push(
          Router::with_path("/api/oai/embeddings").post(api::salvo::oai::salvo_oai_embeddings)
        ).push(
          Router::with_path("/api/oai/v1/embeddings").post(api::salvo::oai::salvo_oai_embeddings)
        ).push(
          Router::with_path("<**path>").get(StaticDir::new(serve_path).defaults(["index.html"]))
        );
        //.fallback_service(ServeDir::new(serve_path))
        //.layer(CorsLayer::permissive());
    let addr = SocketAddr::new(
        args.ip.unwrap_or(IpAddr::from(Ipv4Addr::UNSPECIFIED)),
        args.port,
    );
    
    let listener = TcpListener::new(addr).bind().await;
    log::info!("server started at {addr}");
    salvo::server::Server::new(listener).serve(app).await;
  }
  