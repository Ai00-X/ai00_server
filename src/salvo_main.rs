use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::Path,
};
use clap::Parser;
use salvo::affix;
use salvo::cors::Cors;
use salvo::http::Method;
use salvo::logging::Logger;
use salvo::prelude::*;
use salvo::serve_static::StaticDir;
use salvo::Router;
// use tower_http::{cors::CorsLayer, services::ServeDir};
use crate::{
    api::{self},
    middleware::{model_route, ThreadRequest, ThreadState},
};
use crate::{load_config, load_plugin, load_web, Args};
use salvo::cors::AllowOrigin;

pub async fn salvo_main() {
    use clap::CommandFactory;
    use salvo::conn::rustls::{Keycert, RustlsConfig};

    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ai00_server", log::LevelFilter::Info)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .init()
        .unwrap();

    let args = Args::parse();
    let (sender, receiver) = flume::unbounded::<ThreadRequest>();

    let request: crate::middleware::ReloadRequest = {
        let path = args
            .config
            .clone()
            .unwrap_or("assets/configs/Config.toml".into());
        log::info!("reading config {}...", path.to_string_lossy());
        load_config(path).expect("load config failed").into()
    };

    let listen = request.listen.clone();

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
        .push(Router::with_path("/api/adapters").get(api::salvo::salvo_info))
        .push(Router::with_path("/api/models/info").get(api::salvo::salvo_info))
        .push(Router::with_path("/api/models/load").post(api::salvo::salvo_load))
        .push(Router::with_path("/api/models/unload").get(api::salvo::salvo_unload))
        .push(Router::with_path("/api/models/state").get(api::salvo::salvo_state))
        .push(Router::with_path("/api/models/list").get(api::salvo::salvo_models))
        .push(Router::with_path("/api/files/unzip").post(api::salvo::salvo_unzip))
        .push(Router::with_path("/api/files/dir").post(api::salvo::salvo_dir))
        .push(Router::with_path("/api/files/ls").post(api::salvo::salvo_dir))
        .push(Router::with_path("/api/files/config/load").post(api::salvo::salvo_load_config))
        .push(Router::with_path("/api/files/config/save").post(api::salvo::salvo_save_config))
        .push(Router::with_path("/api/oai/models").get(api::salvo::oai::salvo_oai_models))
        .push(Router::with_path("/api/oai/v1/models").get(api::salvo::oai::salvo_oai_models))
        .push(
            Router::with_path("/api/oai/completions").post(api::salvo::oai::salvo_oai_completions),
        )
        .push(
            Router::with_path("/api/oai/v1/completions")
                .post(api::salvo::oai::salvo_oai_completions),
        )
        .push(
            Router::with_path("/api/oai/chat/completions")
                .post(api::salvo::oai::salvo_oai_chat_completions),
        )
        .push(
            Router::with_path("/api/oai/v1/chat/completions")
                .post(api::salvo::oai::salvo_oai_chat_completions),
        )
        .push(Router::with_path("/api/oai/embeddings").post(api::salvo::oai::salvo_oai_embeddings))
        .push(
            Router::with_path("/api/oai/v1/embeddings").post(api::salvo::oai::salvo_oai_embeddings),
        );
    //.push(
    //  Router::with_path("<**path>").get(StaticDir::new(serve_path).defaults(["index.html"]))
    //);
    //.fallback_service(ServeDir::new(serve_path))
    //.layer(CorsLayer::permissive());
    let cmd = Args::command();
    let version = cmd.get_version().unwrap_or("0.0.1");
    let bin_name = cmd.get_bin_name().unwrap_or("ai00_server");

    let doc = OpenApi::new(bin_name, version).merge_router(&app);

    let app = app
        .push(doc.into_router("/api-doc/openapi.json"))
        .push(SwaggerUi::new("/api-doc/openapi.json").into_router("swagger-ui"))
        .push(
            Router::with_path("<**path>").get(StaticDir::new(serve_path).defaults(["index.html"])),
        ); // this static serve should after the swagger.

    let ipaddr = if args.ip.is_some() {
        args.ip.unwrap()
    } else {
        if listen.is_some() {
            let v4_addr = listen
                .clone()
                .unwrap()
                .ip
                .map(|f| f.parse().unwrap_or(Ipv4Addr::UNSPECIFIED))
                .unwrap_or(Ipv4Addr::UNSPECIFIED);
            IpAddr::from(v4_addr)
        } else {
            IpAddr::from(Ipv4Addr::UNSPECIFIED)
        }
    };

    let bind_port = if args.port > 0 && args.port != 65530u16 {
        args.port
    } else if listen.clone().is_some() {
        listen.clone().unwrap().port.unwrap_or(65530u16)
    } else {
        65530u16
    };

    let (bind_domain, use_acme, use_tls) = if listen.clone().is_some() {
        let clone_listen = listen.clone().unwrap();
        let domain = clone_listen.domain.unwrap_or("local".to_string());
        let acme = if domain == "local".to_string() {
            false
        } else {
            clone_listen.acme.unwrap_or_default()
        };
        let tls = if acme {
            true
        } else {
            clone_listen.tls.unwrap_or_default()
        };

        (domain, acme, tls)
    } else {
        ("local".to_string(), false, false)
    };

    let addr = SocketAddr::new(ipaddr, bind_port);

    if use_acme {
        let acceptor = TcpListener::new(addr)
            .acme()
            .cache_path("assets/certs")
            .add_domain(bind_domain)
            .quinn(addr)
            .bind()
            .await;
        log::info!("server started at {addr} with acme and tls.");
        salvo::server::Server::new(acceptor).serve(app).await;
    } else {
        if use_tls {
            let config = RustlsConfig::new(
                Keycert::new()
                    .cert_from_path("assets/certs/cert.pem")
                    .unwrap()
                    .key_from_path("assets/certs/key.pem")
                    .unwrap(),
            );
            let listener = TcpListener::new(addr).rustls(config.clone());
            let acceptor = QuinnListener::new(config, addr).join(listener).bind().await;
            log::info!("server started at {addr} with tls.");
            salvo::server::Server::new(acceptor).serve(app).await;
        } else {
            log::info!("server started at {addr} without tls.");
            let acceptor = TcpListener::new(addr).bind().await;
            salvo::server::Server::new(acceptor).serve(app).await;
        }
    };
}
