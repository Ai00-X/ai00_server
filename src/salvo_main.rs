use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::Path,
};

use clap::Parser;
use salvo::{
    affix,
    cors::{AllowOrigin, Cors},
    http::Method,
    logging::Logger,
    prelude::*,
    serve_static::StaticDir,
    Router,
};

use crate::{
    api, load_config, load_plugin, load_web,
    middleware::{model_route, ReloadRequest, ThreadRequest, ThreadState},
    Args,
};

#[allow(clippy::collapsible_else_if)]
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

    let (listen, request, config) = {
        let path = args
            .config
            .clone()
            .unwrap_or("assets/configs/Config.toml".into());
        log::info!("reading config {}...", path.to_string_lossy());
        let config = load_config(path).expect("load config failed");
        let listen = config.listen.clone();
        (listen, ReloadRequest::from(config.clone()), config)
    };

    tokio::task::spawn_blocking(move || model_route(receiver));
    let _ = sender.send(ThreadRequest::Reload {
        request: Box::new(request),
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
        .hoop(affix::inject(ThreadState(sender, config)))
        .hoop(cors)
        .push(Router::with_path("/api/adapters").get(api::adapters))
        .push(Router::with_path("/api/models/info").get(api::info))
        .push(Router::with_path("/api/models/load").post(api::load))
        .push(Router::with_path("/api/models/unload").get(api::unload))
        .push(Router::with_path("/api/models/state").get(api::state))
        .push(Router::with_path("/api/models/list").get(api::models))
        .push(Router::with_path("/api/files/unzip").post(api::unzip))
        .push(Router::with_path("/api/files/dir").post(api::dir))
        .push(Router::with_path("/api/files/ls").post(api::dir))
        .push(Router::with_path("/api/files/config/load").post(api::load_config))
        .push(Router::with_path("/api/files/config/save").post(api::save_config))
        .push(Router::with_path("/api/oai/models").get(api::oai::models))
        .push(Router::with_path("/api/oai/v1/models").get(api::oai::models))
        .push(Router::with_path("/api/oai/completions").post(api::oai::completions))
        .push(Router::with_path("/api/oai/v1/completions").post(api::oai::completions))
        .push(Router::with_path("/api/oai/chat/completions").post(api::oai::chat_completions))
        .push(Router::with_path("/api/oai/v1/chat/completions").post(api::oai::chat_completions))
        .push(Router::with_path("/api/oai/embeddings").post(api::oai::embeddings))
        .push(Router::with_path("/api/oai/v1/embeddings").post(api::oai::embeddings));
    // .push(
    //     Router::with_path("<**path>").get(StaticDir::new(serve_path).defaults(["index.html"])),
    // )
    // .fallback_service(ServeDir::new(serve_path))
    // .layer(CorsLayer::permissive());

    let cmd = Args::command();
    let version = cmd.get_version().unwrap_or("0.0.1");
    let bin_name = cmd.get_bin_name().unwrap_or("ai00_server");

    let doc = OpenApi::new(bin_name, version).merge_router(&app);

    let app = app
        .push(doc.into_router("/api-doc/openapi.json"))
        .push(SwaggerUi::new("/api-doc/openapi.json").into_router("swagger-ui"))
        .push(
            // this static serve should be after `swagger`
            Router::with_path("<**path>").get(StaticDir::new(serve_path).defaults(["index.html"])),
        );

    let ip_addr = args.ip.unwrap_or(listen.ip);
    let (ipv4_addr, ipv6_addr) = match ip_addr {
        IpAddr::V4(addr) => (addr, None),
        IpAddr::V6(addr) => (Ipv4Addr::new(0, 0, 0, 0), Some(addr)),
    };
    let port = args.port.unwrap_or(listen.port);
    let (acme, tls) = match listen.domain.as_str() {
        "local" => (false, listen.tls),
        _ => (listen.acme, true),
    };
    let addr = SocketAddr::new(IpAddr::V4(ipv4_addr), port);

    if acme {
        let acme_listener = TcpListener::new(addr)
            .acme()
            .cache_path("assets/certs")
            .add_domain(listen.domain)
            .quinn(addr);
        if let Some(ipv6_addr) = ipv6_addr {
            if ipv6_addr.is_unspecified() && ipv4_addr.is_unspecified() {
                panic!("both IpV4 and IpV6 addresses are unspecified");
            }
            let addr_v6 = SocketAddr::new(IpAddr::V6(ipv6_addr), port);
            let acceptor = acme_listener.join(TcpListener::new(addr_v6)).bind().await;
            log::info!("server started at {addr} with acme and tls.");
            log::info!("server started at {addr_v6} with acme and tls.");
            salvo::server::Server::new(acceptor).serve(app).await;
        } else {
            let acceptor = acme_listener.bind().await;
            log::info!("server started at {addr} with acme and tls.");
            salvo::server::Server::new(acceptor).serve(app).await;
        };
    } else if tls {
        let config = RustlsConfig::new(
            Keycert::new()
                .cert_from_path("assets/certs/cert.pem")
                .expect("unable to find cert.pem")
                .key_from_path("assets/certs/key.pem")
                .expect("unable to fine key.pem"),
        );
        let listener = TcpListener::new(addr).rustls(config.clone());
        if let Some(ipv6_addr) = ipv6_addr {
            let addr_v6 = SocketAddr::new(IpAddr::V6(ipv6_addr), port);
            let ipv6_listener = TcpListener::new(addr_v6).rustls(config.clone());
            #[cfg(not(target_os = "windows"))]
            let acceptor = QuinnListener::new(config.clone(), addr_v6)
                .join(ipv6_listener)
                .bind()
                .await;
            #[cfg(target_os = "windows")]
            let acceptor = QuinnListener::new(config.clone(), addr)
                .join(QuinnListener::new(config, addr_v6))
                .join(ipv6_listener)
                .join(listener)
                .bind()
                .await;
            log::info!("server started at {addr} with tls");
            log::info!("server started at {addr_v6} with tls");
            salvo::server::Server::new(acceptor).serve(app).await;
        } else {
            let acceptor = QuinnListener::new(config.clone(), addr)
                .join(listener)
                .bind()
                .await;
            log::info!("server started at {addr} with tls");
            salvo::server::Server::new(acceptor).serve(app).await;
        };
    } else if let Some(ipv6_addr) = ipv6_addr {
        let addr_v6 = SocketAddr::new(IpAddr::V6(ipv6_addr), port);
        let ipv6_listener = TcpListener::new(addr_v6);
        log::info!("server started at {addr} without tls");
        log::info!("server started at {addr_v6} without tls");
        // On linux, when the IPv6 addr is unspecified, and IPv4 addr is unspecified, that will cause exception "Address in used"
        #[cfg(not(target_os = "windows"))]
        if ipv6_addr.is_unspecified() {
            let acceptor = ipv6_listener.bind().await;
            salvo::server::Server::new(acceptor).serve(app).await;
        } else {
            let acceptor = TcpListener::new(addr).join(ipv6_listener).bind().await;
            salvo::server::Server::new(acceptor).serve(app).await;
        };
        #[cfg(target_os = "windows")]
        {
            let acceptor = TcpListener::new(addr).join(ipv6_listener).bind().await;
            salvo::server::Server::new(acceptor).serve(app).await;
        }
    } else {
        log::info!("server started at {addr} without tls");
        let acceptor = TcpListener::new(addr).bind().await;
        salvo::server::Server::new(acceptor).serve(app).await;
    };
}
