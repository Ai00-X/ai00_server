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

use crate::types::JwtClaims;

mod api;
mod config;
mod types;

const SLEEP: Duration = Duration::from_millis(500);

// #[derive(Debug, Deserialize)]
// struct EmbeddingConfig {
//     open_embed: bool,
//     model_name: String,
//     endpoint: String,
//     home_path: String,
// }

// /// Data struct about the available models
// #[derive(Debug, Clone)]
// pub struct ModelInfo {
//     pub model: EmbeddingModel,
//     pub model_code: String,
// }

// pub fn models_list() -> Vec<ModelInfo> {
//     let models_list = vec![
//         ModelInfo {
//             model: EmbeddingModel::AllMiniLML6V2,
//             model_code: String::from("Qdrant/all-MiniLM-L6-v2-onnx"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::AllMiniLML6V2Q,
//             model_code: String::from("Xenova/all-MiniLM-L6-v2"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::BGEBaseENV15,
//             model_code: String::from("Xenova/bge-base-en-v1.5"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::BGEBaseENV15Q,
//             model_code: String::from("Qdrant/bge-base-en-v1.5-onnx-Q"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::BGELargeENV15,
//             model_code: String::from("Xenova/bge-large-en-v1.5"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::BGELargeENV15Q,
//             model_code: String::from("Qdrant/bge-large-en-v1.5-onnx-Q"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::BGESmallENV15,
//             model_code: String::from("Xenova/bge-small-en-v1.5"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::BGESmallENV15Q,
//             model_code: String::from("Qdrant/bge-small-en-v1.5-onnx-Q"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::NomicEmbedTextV1,
//             model_code: String::from("nomic-ai/nomic-embed-text-v1"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::NomicEmbedTextV15,
//             model_code: String::from("nomic-ai/nomic-embed-text-v1.5"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::NomicEmbedTextV15Q,
//             model_code: String::from("nomic-ai/nomic-embed-text-v1.5"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::ParaphraseMLMiniLML12V2Q,
//             model_code: String::from("Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::ParaphraseMLMiniLML12V2,
//             model_code: String::from("Xenova/paraphrase-multilingual-MiniLM-L12-v2"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::ParaphraseMLMpnetBaseV2,
//             model_code: String::from("Xenova/paraphrase-multilingual-mpnet-base-v2"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::BGESmallZHV15,
//             model_code: String::from("Xenova/bge-small-zh-v1.5"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::MultilingualE5Small,
//             model_code: String::from("intfloat/multilingual-e5-small"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::MultilingualE5Base,
//             model_code: String::from("intfloat/multilingual-e5-base"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::MultilingualE5Large,
//             model_code: String::from("Qdrant/multilingual-e5-large-onnx"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::MxbaiEmbedLargeV1,
//             model_code: String::from("mixedbread-ai/mxbai-embed-large-v1"),
//         },
//         ModelInfo {
//             model: EmbeddingModel::MxbaiEmbedLargeV1Q,
//             model_code: String::from("mixedbread-ai/mxbai-embed-large-v1"),
//         },
//     ];
//     models_list
// }

// lazy_static! {
// #[derive(Debug)]
// static ref EMBEDCONFIG: EmbeddingConfig = {
//     // 假设 config.json 在程序根目录
//     let config_str = fs::read_to_string("embed_config.json").expect("Unable to read config file");
//     serde_json::from_str(&config_str).expect("Unable to parse config file")
// };

// #[allow(clippy::let_and_return)]
// static ref EMBEDTOKENIZERS: Tokenizer = {

//     env::set_var("HF_ENDPOINT", EMBEDCONFIG.endpoint.clone());
//     env::set_var("HF_HOME", EMBEDCONFIG.home_path.clone());

//     let models_list = models_list(); // 假设这个函数已经定义并可用
//     let identifier = models_list
//         .iter()
//         .find(|m| m.model == EmbeddingModel::from_name(&EMBEDCONFIG.model_name))
//         .map(|m| m.model_code.clone())
//         .unwrap();

//     let api = Api::new().unwrap();

//     let filename = api
//     .model(identifier)
//     .get("tokenizer.json")
//     .unwrap();

//     Tokenizer::from_file(filename).unwrap()
// };

// static ref EMBEDMODEL: TextEmbedding = {
//     env::set_var("HF_ENDPOINT", EMBEDCONFIG.endpoint.clone());
//     env::set_var("HF_HOME", EMBEDCONFIG.home_path.clone());

//     println!("Loading embed_model: {}", EMBEDCONFIG.model_name);

//     TextEmbedding::try_new(InitOptions {
//         model_name: EmbeddingModel::from_name(&EMBEDCONFIG.model_name),
//         show_download_progress: true,
//         ..Default::default()
//     }).expect("Failed to initialize embed_model")
// };
// }

// trait EmbeddingModelExt {
//     fn from_name(name: &str) -> Self;
// }

// impl EmbeddingModelExt for EmbeddingModel {
//     fn from_name(name: &str) -> Self {
//         match name {
//             "AllMiniLML6V2" => EmbeddingModel::AllMiniLML6V2,
//             "AllMiniLML6V2Q" => EmbeddingModel::AllMiniLML6V2Q,
//             "BGEBaseENV15" => EmbeddingModel::BGEBaseENV15,
//             "BGEBaseENV15Q" => EmbeddingModel::BGEBaseENV15Q,
//             "BGELargeENV15" => EmbeddingModel::BGELargeENV15,
//             "BGELargeENV15Q" => EmbeddingModel::BGELargeENV15Q,
//             "BGESmallENV15" => EmbeddingModel::BGESmallENV15,
//             "BGESmallENV15Q" => EmbeddingModel::BGESmallENV15Q,
//             "NomicEmbedTextV1" => EmbeddingModel::NomicEmbedTextV1,
//             "NomicEmbedTextV15" => EmbeddingModel::NomicEmbedTextV15,
//             "NomicEmbedTextV15Q" => EmbeddingModel::NomicEmbedTextV15Q,
//             "ParaphraseMLMiniLML12V2" => EmbeddingModel::ParaphraseMLMiniLML12V2,
//             "ParaphraseMLMiniLML12V2Q" => EmbeddingModel::ParaphraseMLMiniLML12V2,
//             "ParaphraseMLMpnetBaseV2" => EmbeddingModel::ParaphraseMLMpnetBaseV2,
//             "BGESmallZHV15" => EmbeddingModel::BGESmallZHV15,
//             "MultilingualE5Small" => EmbeddingModel::MultilingualE5Small,
//             "MultilingualE5Base" => EmbeddingModel::MultilingualE5Base,
//             "MultilingualE5Large" => EmbeddingModel::MultilingualE5Large,
//             "MxbaiEmbedLargeV1" => EmbeddingModel::MxbaiEmbedLargeV1,
//             "MxbaiEmbedLargeV1Q" => EmbeddingModel::MxbaiEmbedLargeV1Q,
//             _ => panic!("Unsupported model name"),
//         }
//     }
// }

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

    // println!("{:?}", EMBEDCONFIG);
    // if EMBEDCONFIG.open_embed {
    //     let _emb = EMBEDMODEL
    //         .embed(["ok"].to_vec(), None)
    //         .expect("Failed to get embedding");
    // }

    let config = {
        let path = args
            .config
            .clone()
            .unwrap_or("assets/configs/Config.toml".into());
        log::info!("reading config {}...", path.to_string_lossy());
        load_config(path).await.expect("load config failed")
    };

    let request = Box::new(config.clone().try_into().expect("load model failed"));
    let _ = sender.send(ThreadRequest::Reload {
        request,
        sender: None,
    });

    let serve_path = match config.web.clone() {
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
            .force_passed(config.listen.force_pass.unwrap_or_default());

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
    let api_embed = Router::new()
        .push(Router::with_path("/oai/embeds").post(api::oai::embeds))
        .push(Router::with_path("/oai/v1/embeds").post(api::oai::embeds));

    let app = Router::new()
        //.hoop(CorsLayer::permissive())
        .hoop(Logger::new())
        .hoop(affix::inject(sender).inject(config.clone()))
        .push(
            Router::with_path("/api")
                .push(Router::with_path("/auth/exchange").post(api::auth::exchange))
                .push(api_router)
                .push(api_embed),
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
    let ip_addr = args.ip.unwrap_or(config.listen.ip);
    let (ipv4_addr, ipv6_addr) = match ip_addr {
        IpAddr::V4(addr) => (addr, None),
        IpAddr::V6(addr) => (Ipv4Addr::UNSPECIFIED, Some(addr)),
    };
    let port = args.port.unwrap_or(config.listen.port);
    let (acme, tls) = match config.listen.domain.as_str() {
        "local" => (false, config.listen.tls),
        _ => (config.listen.acme, true),
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
            .add_domain(&config.listen.domain)
            .quinn(ipv4_addr);
        if let Some(ipv6_addr) = ipv6_addr {
            let ipv6_addr = SocketAddr::new(IpAddr::V6(ipv6_addr), port);
            let ipv6_listener = TcpListener::new(ipv6_addr)
                .acme()
                .cache_path("assets/certs")
                .add_domain(&config.listen.domain)
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
