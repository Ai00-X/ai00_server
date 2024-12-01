use std::{
    net::{IpAddr, Ipv4Addr},
    path::PathBuf,
};

use ai00_core::{
    reload::{AdapterOption, BnfOption, Lora, Model, State, Tokenizer},
    ReloadRequest,
};
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::build_path;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: Model,
    pub lora: Vec<Lora>,
    pub state: Vec<State>,
    pub tokenizer: Tokenizer,
    pub bnf: BnfOption,
    pub adapter: AdapterOption,
    pub listen: ListenerOption,
    pub web: Option<WebOption>,
    #[cfg(feature = "embed")]
    pub embed: Option<EmbedOption>,
}

impl TryFrom<Config> for ReloadRequest {
    type Error = anyhow::Error;

    fn try_from(value: Config) -> Result<Self, Self::Error> {
        let Config {
            model:
                Model {
                    name,
                    path,
                    quant,
                    quant_type,
                    precision,
                    token_chunk_size,
                    max_batch,
                    embed_device,
                },
            mut lora,
            mut state,
            tokenizer: Tokenizer {
                path: tokenizer_path,
            },
            bnf,
            adapter,
            ..
        } = value;

        let model_path = build_path(&path, name)?;
        for lora in lora.iter_mut() {
            lora.path = build_path(&path, &lora.path)?;
        }
        for state in state.iter_mut() {
            state.path = build_path(&path, &state.path)?;
        }

        Ok(Self {
            model_path,
            lora,
            state,
            quant,
            quant_type,
            precision,
            token_chunk_size,
            max_batch,
            embed_device,
            tokenizer_path,
            bnf,
            adapter,
        })
    }
}

#[cfg(feature = "embed")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(remote = "fastembed::EmbeddingModel")]
pub enum EmbeddingModel {
    /// sentence-transformers/all-MiniLM-L6-v2
    AllMiniLML6V2,
    /// Quantized sentence-transformers/all-MiniLM-L6-v2
    AllMiniLML6V2Q,
    /// sentence-transformers/all-MiniLM-L12-v2
    AllMiniLML12V2,
    /// Quantized sentence-transformers/all-MiniLM-L12-v2
    AllMiniLML12V2Q,
    /// BAAI/bge-base-en-v1.5
    BGEBaseENV15,
    /// Quantized BAAI/bge-base-en-v1.5
    BGEBaseENV15Q,
    /// BAAI/bge-large-en-v1.5
    BGELargeENV15,
    /// Quantized BAAI/bge-large-en-v1.5
    BGELargeENV15Q,
    /// BAAI/bge-small-en-v1.5 - Default
    BGESmallENV15,
    /// Quantized BAAI/bge-small-en-v1.5
    BGESmallENV15Q,
    /// nomic-ai/nomic-embed-text-v1
    NomicEmbedTextV1,
    /// nomic-ai/nomic-embed-text-v1.5
    NomicEmbedTextV15,
    /// Quantized v1.5 nomic-ai/nomic-embed-text-v1.5
    NomicEmbedTextV15Q,
    /// sentence-transformers/paraphrase-MiniLM-L6-v2
    ParaphraseMLMiniLML12V2,
    /// Quantized sentence-transformers/paraphrase-MiniLM-L6-v2
    ParaphraseMLMiniLML12V2Q,
    /// sentence-transformers/paraphrase-mpnet-base-v2
    ParaphraseMLMpnetBaseV2,
    /// BAAI/bge-small-zh-v1.5
    BGESmallZHV15,
    /// intfloat/multilingual-e5-small
    MultilingualE5Small,
    /// intfloat/multilingual-e5-base
    MultilingualE5Base,
    /// intfloat/multilingual-e5-large
    MultilingualE5Large,
    /// mixedbread-ai/mxbai-embed-large-v1
    MxbaiEmbedLargeV1,
    /// Quantized mixedbread-ai/mxbai-embed-large-v1
    MxbaiEmbedLargeV1Q,
    /// Alibaba-NLP/gte-base-en-v1.5
    GTEBaseENV15,
    /// Quantized Alibaba-NLP/gte-base-en-v1.5
    GTEBaseENV15Q,
    /// Alibaba-NLP/gte-large-en-v1.5
    GTELargeENV15,
    /// Quantized Alibaba-NLP/gte-large-en-v1.5
    GTELargeENV15Q,
    /// Qdrant/clip-ViT-B-32-text
    ClipVitB32,
}

#[cfg(feature = "embed")]
#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct EmbedOption {
    #[serde(with = "EmbeddingModel")]
    #[derivative(Default(value = "fastembed::EmbeddingModel::MultilingualE5Small"))]
    pub model: fastembed::EmbeddingModel,
    #[derivative(Default(value = "\"https://huggingface.co\".into()"))]
    pub endpoint: String,
    #[derivative(Default(value = "\"assets/models/hf\".into()"))]
    pub home: PathBuf,
    #[cfg(target_os = "windows")]
    #[derivative(Default(value = "\"assets/ort/onnxruntime.dll\".into()"))]
    pub lib: PathBuf,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AppKey {
    pub app_id: String,
    pub secret_key: String,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct ListenerOption {
    /// Ip to listen to.
    #[derivative(Default(value = "IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))"))]
    pub ip: IpAddr,
    /// Binding port.
    #[derivative(Default(value = "65530u16"))]
    pub port: u16,
    /// Domain for certs. Certs are stored in `assets/certs/`.
    #[derivative(Default(value = "String::from(\"local\")"))]
    pub domain: String,
    /// Using acme to issue the certs if the domain is not local.
    #[derivative(Default(value = "false"))]
    pub acme: bool,
    /// Force to enable https. When acme is true, tls must be true.
    #[derivative(Default(value = "false"))]
    pub tls: bool,
    /// For JWT Token encoding and decoding.    
    #[derivative(Default(value = "String::from(\"ai00s\")"))]
    pub slot: String,
    /// Whether the identifier is forced to pass even if JWT Token verification fails
    #[derivative(Default(value = "Some(false)"))]
    pub force_pass: Option<bool>,
    /// Token expiration time by second
    #[derivative(Default(value = "Some(86400u32)"))]
    pub expire_sec: Option<u32>,
    /// AppId with SecretKey pairs
    pub app_keys: Vec<AppKey>,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct WebOption {
    #[derivative(Default(value = "\"assets/www/index.zip\".into()"))]
    pub path: PathBuf,
}
