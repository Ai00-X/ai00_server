use std::path::PathBuf;

use derivative::Derivative;
use serde::{Deserialize, Serialize};
use web_rwkv::model::{EmbedDevice, Quant};

use crate::ReloadRequest;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: Model,
    pub lora: Vec<Lora>,
    pub tokenizer: Tokenizer,
    pub adapter: AdapterOption,
}

impl From<Config> for ReloadRequest {
    fn from(value: Config) -> Self {
        let Config {
            model:
                Model {
                    path: model_path,
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
            lora,
            tokenizer: Tokenizer {
                path: tokenizer_path,
            },
            adapter,
            ..
        } = value;
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
    /// Path to the model.
    pub path: PathBuf,
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
