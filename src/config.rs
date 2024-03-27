use std::{
    net::{IpAddr, Ipv4Addr},
    path::PathBuf,
};

use derivative::Derivative;
use serde::{Deserialize, Serialize};
use web_rwkv::model::{EmbedDevice, Quant};

use crate::{build_path, middleware::ReloadRequest};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: Model,
    pub lora: Vec<Lora>,
    pub tokenizer: Tokenizer,
    pub bnf: BnfOption,
    pub adapter: AdapterOption,
    pub listen: ListenerOption,
}

impl TryFrom<Config> for ReloadRequest {
    type Error = anyhow::Error;

    fn try_from(value: Config) -> Result<Self, Self::Error> {
        let Config {
            model:
                Model {
                    model_name,
                    model_path,
                    quant,
                    quant_type,
                    turbo,
                    token_chunk_size,
                    state_chunk_size,
                    max_runtime_batch,
                    max_batch,
                    embed_device,
                },
            mut lora,
            tokenizer: Tokenizer {
                path: tokenizer_path,
            },
            bnf,
            adapter,
            ..
        } = value;

        for lora in lora.iter_mut() {
            lora.path = build_path(&model_path, &lora.path)?;
        }
        let model_path = build_path(&model_path, model_name)?;

        Ok(Self {
            model_path,
            lora,
            quant,
            quant_type,
            turbo,
            token_chunk_size,
            state_chunk_size,
            max_runtime_batch,
            max_batch,
            embed_device,
            tokenizer_path,
            bnf,
            adapter,
        })
    }
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct Model {
    /// Path to the folder containing all models.
    #[derivative(Default(value = "\"assets/models\".into()"))]
    pub model_path: PathBuf,
    /// Name of the model.
    pub model_name: PathBuf,
    /// Specify layers that needs to be quantized.
    pub quant: usize,
    /// Quantization type (Int8 or NF4).
    pub quant_type: Quant,
    /// Whether to use alternative GEMM kernel to speed-up long prompts.
    #[derivative(Default(value = "true"))]
    pub turbo: bool,
    /// Maximum tokens to be processed in parallel at once.
    #[derivative(Default(value = "128"))]
    pub token_chunk_size: usize,
    /// The chunk size of layers in model state.
    #[derivative(Default(value = "4"))]
    pub state_chunk_size: usize,
    /// Maximum number of batches that are active at once.
    #[derivative(Default(value = "8"))]
    pub max_runtime_batch: usize,
    /// Number of states that are cached on GPU.
    #[derivative(Default(value = "16"))]
    pub max_batch: usize,
    /// Device to put the embed tensor.
    pub embed_device: EmbedDevice,
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct Lora {
    /// Path to the LoRA.
    pub path: PathBuf,
    /// Blend factor.
    #[derivative(Default(value = "1.0"))]
    pub alpha: f32,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct Tokenizer {
    #[derivative(Default(value = "\"assets/tokenizer/rwkv_vocab_v20230424.json\".into()"))]
    pub path: PathBuf,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct BnfOption {
    /// Enable the cache that accelerates the expansion of certain short schemas.
    #[derivative(Default(value = "true"))]
    pub enable_bytes_cache: bool,
    /// The initial nonterminal of the BNF schemas.
    #[derivative(Default(value = "\"start\".into()"))]
    pub start_nonterminal: String,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum AdapterOption {
    #[default]
    Auto,
    Economical,
    Manual(usize),
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AppKey {
    pub app_id: String,
    pub secret_key: String,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
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
