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
    pub adapter: AdapterOption,
    pub listen: ListenerOption,
}

impl From<Config> for ReloadRequest {
    fn from(value: Config) -> Self {
        let Config {
            model:
                Model {
                    model_name,
                    model_path,
                    quant,
                    quant_type,
                    turbo,
                    token_chunk_size,
                    head_chunk_size,
                    state_chunk_size,
                    max_runtime_batch,
                    max_batch,
                    embed_device,
                },
            mut lora,
            tokenizer: Tokenizer {
                path: tokenizer_path,
            },
            adapter,
            ..
        } = value;

        for lora in lora.iter_mut() {
            lora.path = build_path(&model_path, &lora.path).expect("error building path");
        }
        let model_path = build_path(&model_path, model_name).expect("error building path");

        Self {
            model_path,
            lora,
            quant,
            quant_type,
            turbo,
            token_chunk_size,
            head_chunk_size,
            state_chunk_size,
            max_runtime_batch,
            max_batch,
            embed_device,
            tokenizer_path,
            adapter,
        }
    }
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct Model {
    /// Name of the model.
    pub model_name: PathBuf,
    /// Path to the folder containing all models.
    #[derivative(Default(value = "String::from(\"assets/models\").into()"))]
    pub model_path: PathBuf,
    /// Specify layers that needs to be quantized.
    pub quant: usize,
    /// Quantization type (Int8 or NF4).
    #[derivative(Default(value = "Quant::Int8"))]
    pub quant_type: Quant,
    /// Whether to use alternative GEMM kernel to speed-up long prompts.
    #[derivative(Default(value = "true"))]
    pub turbo: bool,
    /// Maximum tokens to be processed in parallel at once.
    #[derivative(Default(value = "32"))]
    pub token_chunk_size: usize,
    /// The chunk size for each split of the head matrix.
    #[derivative(Default(value = "8192"))]
    pub head_chunk_size: usize,
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

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Tokenizer {
    pub path: PathBuf,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum AdapterOption {
    #[default]
    Auto,
    Economical,
    Manual(usize),
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
    pub acme: bool,
    /// Force to enable https. When acme is true, tls must be true.
    pub tls: bool,
}
