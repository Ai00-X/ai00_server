use std::{
    io::Cursor,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    path::{Path, PathBuf},
    time::Duration,
};

use ai00_core::{model_route, ThreadRequest};
use anyhow::{bail, Result};
use clap::{command, CommandFactory, Parser};
use memmap2::Mmap;
use salvo::{
    affix,
    conn::rustls::{Keycert, RustlsConfig},
    cors::{AllowHeaders, AllowOrigin, Cors},
    http::Method,
    jwt_auth::{ConstDecoder, HeaderFinder, QueryFinder},
    logging::Logger,
    prelude::*,
    serve_static::StaticDir,
    Router,
};
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};

use crate::types::{JwtClaims, ThreadState};

mod api;
mod config;
mod types;

const SLEEP: Duration = Duration::from_millis(500);

pub fn build_path(path: impl AsRef<Path>, name: impl AsRef<Path>) -> Result<PathBuf> {
    let permitted = path.as_ref();
    let name = name.as_ref();
    if name.ancestors().any(|p| p.ends_with(Path::new(".."))) {
        bail!("cannot have \"..\" in names");
    }
    let path = match name.is_absolute() || name.starts_with(permitted) {
        true => name.into(),
        false => permitted.join(name),
    };
    match path.starts_with(permitted) {
        true => Ok(path),
        false => bail!("path not permitted"),
    }
}

pub fn check_path_permitted(path: impl AsRef<Path>, permitted: &[&str]) -> Result<()> {
    let current_path = std::env::current_dir()?;
    for sub in permitted {
        let permitted = current_path.join(sub).canonicalize()?;
        let path = path.as_ref().canonicalize()?;
        if path.starts_with(permitted) {
            return Ok(());
        }
    }
    bail!("path not permitted");
}

pub async fn load_web(path: impl AsRef<Path>, target: &Path) -> Result<()> {
    let file = File::open(path).await?;
    let map = unsafe { Mmap::map(&file)? };
    zip_extract::extract(Cursor::new(&map), target, false)?;
    Ok(())
}

pub async fn load_plugin(path: impl AsRef<Path>, target: &Path, name: &String) -> Result<()> {
    let file = File::open(path).await?;
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

pub async fn load_config(path: impl AsRef<Path>) -> Result<config::Config> {
    let file = File::open(path).await?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
    Ok(toml::from_str(&contents)?)
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long, short, value_name = "FILE")]
    config: Option<PathBuf>,
    #[arg(long, short)]
    ip: Option<IpAddr>,
    #[arg(long, short)]
    port: Option<u16>,
}

#[tokio::main]
async fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ai00_server", log::LevelFilter::Info)
        .with_module_level("ai00_core", log::LevelFilter::Info)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .init()
        .expect("start logger");

    let args = Args::parse();

    let cmd = Args::command();
    let version = cmd.get_version().unwrap_or("0.0.1");
    let bin_name = cmd.get_bin_name().unwrap_or("ai00_server");

    log::info!("{}\tversion: {}", bin_name, version);

    let (sender, receiver) = flume::unbounded::<ThreadRequest>();
    tokio::spawn(model_route(receiver));

    let (listen, config) = {
        let path = args
            .config
            .clone()
            .unwrap_or("assets/configs/Config.toml".into());
        log::info!("reading config {}...", path.to_string_lossy());
        let config = load_config(path).await.expect("load config failed");
        let listen = config.listen.clone();
        (listen, config)
    };

    let request = Box::new(config.clone().try_into().expect("load model failed"));
    let _ = sender.send(ThreadRequest::Reload {
        request,
        sender: None,
    });

    let serve_path = match config.web {
        Some(web) => {
            if Path::new("assets/temp").exists() {
                std::fs::remove_dir_all("assets/temp").expect("delete temp dir failed");
            }

            std::fs::create_dir("assets/temp").expect("create plugins dir failed");
            let path = PathBuf::from("assets/temp");

            load_web(web.path, &path)
                .await
                .expect("load frontend failed");

            // create `assets/www/plugins` if it doesn't exist
            if !Path::new("assets/www/plugins").exists() {
                std::fs::create_dir("assets/www/plugins").expect("create plugins dir failed");
            }

            // extract and load all plugins under `assets/www/plugins`
            match std::fs::read_dir("assets/www/plugins") {
                Ok(dir) => {
                    for x in dir
                        .filter_map(|x| x.ok())
                        .filter(|x| x.path().is_file())
                        .filter(|x| x.path().extension().is_some_and(|ext| ext == "zip"))
                        .filter(|x| x.path().file_stem().is_some_and(|stem| stem != "api"))
                    {
                        let name = x
                            .path()
                            .file_stem()
                            .expect("this cannot happen")
                            .to_string_lossy()
                            .into();
                        match load_plugin(x.path(), &path, &name).await {
                            Ok(_) => log::info!("loaded plugin {}", name),
                            Err(err) => log::error!("failed to load plugin {}, {}", name, err),
                        }
                    }
                }
                Err(err) => {
                    log::error!("failed to read plugin directory: {}", err);
                }
            };

            Some(path)
        }
        None => None,
    };

    let cors = Cors::new()
        .allow_origin(AllowOrigin::any())
        .allow_methods(vec![Method::GET, Method::POST, Method::DELETE])
        .allow_headers(AllowHeaders::any())
        .into_handler();

    let admin_auth: JwtAuth<JwtClaims, _> =
        JwtAuth::new(ConstDecoder::from_secret(config.listen.slot.as_bytes()))
            .finders(vec![
                Box::new(HeaderFinder::new()),
                Box::new(QueryFinder::new("admin_token")),
                // Box::new(CookieFinder::new("jwt_token")),
            ])
            .force_passed(listen.force_pass.unwrap_or_default());

    let admin_router = Router::with_hoop(admin_auth)
        .push(Router::with_path("/models/save").post(api::model::save))
        .push(Router::with_path("/models/load").post(api::model::load))
        .push(Router::with_path("/models/unload").get(api::model::unload))
        .push(Router::with_path("/models/state/load").post(api::model::load_state))
        .push(Router::with_path("/files/unzip").post(api::file::unzip))
        .push(Router::with_path("/files/dir").post(api::file::dir))
        .push(Router::with_path("/files/ls").post(api::file::dir))
        .push(Router::with_path("/files/config/load").post(api::file::load_config))
        .push(Router::with_path("/files/config/save").post(api::file::save_config));
    let api_router = Router::new()
        .push(Router::with_path("/adapters").get(api::adapter::adapters))
        .push(Router::with_path("/models/info").get(api::model::info))
        .push(Router::with_path("/models/list").get(api::file::models))
        .push(Router::with_path("/models/state").get(api::model::state))
        .push(Router::with_path("/oai/models").get(api::oai::models))
        .push(Router::with_path("/oai/v1/models").get(api::oai::models))
        .push(Router::with_path("/oai/completions").post(api::oai::completions))
        .push(Router::with_path("/oai/v1/completions").post(api::oai::completions))
        .push(Router::with_path("/oai/chat/completions").post(api::oai::chat_completions))
        .push(Router::with_path("/oai/v1/chat/completions").post(api::oai::chat_completions))
        .push(Router::with_path("/oai/embeddings").post(api::oai::embeddings))
        .push(Router::with_path("/oai/v1/embeddings").post(api::oai::embeddings))
        .push(Router::with_path("/oai/chooses").post(api::oai::chooses))
        .push(Router::with_path("/oai/v1/chooses").post(api::oai::chooses));

    let app = Router::new()
        //.hoop(CorsLayer::permissive())
        .hoop(Logger::new())
        .hoop(
            affix::inject(ThreadState {
                sender,
                path: config.model.path,
            })
            .insert("listen", listen.clone()),
        )
        .push(
            Router::with_path("/api")
                .push(Router::with_path("/auth/exchange").post(api::auth::exchange))
                .push(api_router),
        )
        .push(Router::with_path("/admin").push(admin_router));

    let doc = OpenApi::new(bin_name, version).merge_router(&app);

    let app = app
        .push(doc.into_router("/api-docs/openapi.json"))
        .push(SwaggerUi::new("/api-docs/openapi.json").into_router("api-docs"));
    // this static serve should be after `swagger`
    let app = match serve_path {
        Some(path) => app
            .push(Router::with_path("<**path>").get(StaticDir::new(path).defaults(["index.html"]))),
        None => app,
    };

    let service = Service::new(app).hoop(cors);
    let ip_addr = args.ip.unwrap_or(listen.ip);
    let (ipv4_addr, ipv6_addr) = match ip_addr {
        IpAddr::V4(addr) => (addr, None),
        IpAddr::V6(addr) => (Ipv4Addr::UNSPECIFIED, Some(addr)),
    };
    let port = args.port.unwrap_or(listen.port);
    let (acme, tls) = match listen.domain.as_str() {
        "local" => (false, listen.tls),
        _ => (listen.acme, true),
    };
    let ipv4_addr = SocketAddr::new(IpAddr::V4(ipv4_addr), port);

    let url = match ip_addr {
        IpAddr::V6(Ipv6Addr::UNSPECIFIED) | IpAddr::V4(Ipv4Addr::UNSPECIFIED) => "localhost".into(),
        IpAddr::V6(addr) => addr.to_string(),
        IpAddr::V4(addr) => addr.to_string(),
    };
    let url = match acme || tls {
        true => format!("https://{url}:{port}"),
        false => format!("http://{url}:{port}"),
    };
    log::info!("visit WebUI at {url}");

    if acme {
        let listener = TcpListener::new(ipv4_addr)
            .acme()
            .cache_path("assets/certs")
            .add_domain(&listen.domain)
            .quinn(ipv4_addr);
        if let Some(ipv6_addr) = ipv6_addr {
            let ipv6_addr = SocketAddr::new(IpAddr::V6(ipv6_addr), port);
            let ipv6_listener = TcpListener::new(ipv6_addr)
                .acme()
                .cache_path("assets/certs")
                .add_domain(&listen.domain)
                .quinn(ipv6_addr);
            #[cfg(not(target_os = "windows"))]
            let acceptor = ipv6_listener.bind().await;
            #[cfg(target_os = "windows")]
            let acceptor = listener.join(ipv6_listener).bind().await;
            log::info!("server started at {ipv6_addr} with acme and tls");
            salvo::server::Server::new(acceptor).serve(service).await;
        } else {
            let acceptor = listener.bind().await;
            log::info!("server started at {ipv4_addr} with acme and tls.");
            salvo::server::Server::new(acceptor).serve(service).await;
        };
    } else if tls {
        let config = RustlsConfig::new(
            Keycert::new()
                .cert_from_path("assets/certs/cert.pem")
                .expect("unable to find cert.pem")
                .key_from_path("assets/certs/key.pem")
                .expect("unable to fine key.pem"),
        );
        let listener = TcpListener::new(ipv4_addr).rustls(config.clone());
        if let Some(ipv6_addr) = ipv6_addr {
            let ipv6_addr = SocketAddr::new(IpAddr::V6(ipv6_addr), port);
            let ipv6_listener = TcpListener::new(ipv6_addr).rustls(config.clone());
            #[cfg(not(target_os = "windows"))]
            let acceptor = QuinnListener::new(config.clone(), ipv6_addr)
                .join(ipv6_listener)
                .bind()
                .await;
            #[cfg(target_os = "windows")]
            let acceptor = QuinnListener::new(config.clone(), ipv4_addr)
                .join(QuinnListener::new(config, ipv6_addr))
                .join(ipv6_listener)
                .join(listener)
                .bind()
                .await;
            log::info!("server started at {ipv6_addr} with tls");
            salvo::server::Server::new(acceptor).serve(service).await;
        } else {
            let acceptor = QuinnListener::new(config.clone(), ipv4_addr)
                .join(listener)
                .bind()
                .await;
            log::info!("server started at {ipv4_addr} with tls");
            salvo::server::Server::new(acceptor).serve(service).await;
        };
    } else if let Some(ipv6_addr) = ipv6_addr {
        let ipv6_addr = SocketAddr::new(IpAddr::V6(ipv6_addr), port);
        let ipv6_listener = TcpListener::new(ipv6_addr);
        log::info!("server started at {ipv6_addr} without tls");
        // on Linux, when the IpV6 addr is unspecified while the IpV4 addr being unspecified, it will cause exception "address in used"
        #[cfg(not(target_os = "windows"))]
        if ipv6_addr.ip().is_unspecified() {
            let acceptor = ipv6_listener.bind().await;
            salvo::server::Server::new(acceptor).serve(service).await;
        } else {
            let acceptor = TcpListener::new(ipv4_addr).join(ipv6_listener).bind().await;
            salvo::server::Server::new(acceptor).serve(service).await;
        };
        #[cfg(target_os = "windows")]
        {
            let acceptor = TcpListener::new(ipv4_addr).join(ipv6_listener).bind().await;
            salvo::server::Server::new(acceptor).serve(service).await;
        }
    } else {
        log::info!("server started at {ipv4_addr} without tls");
        let acceptor = TcpListener::new(ipv4_addr).bind().await;
        salvo::server::Server::new(acceptor).serve(service).await;
    };
}
