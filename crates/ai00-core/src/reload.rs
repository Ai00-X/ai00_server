use std::path::PathBuf;

use derivative::Derivative;
use salvo::oapi::ToSchema;
use serde::{Deserialize, Serialize};
use web_rwkv::runtime::model::{EmbedDevice, Quant};

use crate::StateId;

#[derive(Debug, Clone, Derivative, Serialize, Deserialize, ToSchema)]
#[derivative(Default)]
#[serde(default)]
pub struct Model {
    /// Path to the folder containing all models.
    #[derivative(Default(value = "\"assets/models\".into()"))]
    #[serde(alias = "model_path")]
    #[salvo(schema(value_type = String))]
    pub path: PathBuf,
    /// Name of the model.
    #[serde(alias = "model_name")]
    #[salvo(schema(value_type = String))]
    pub name: PathBuf,
    /// Specify layers that needs to be quantized.
    pub quant: usize,
    /// Quantization type (`Int8` or `NF4`).
    #[salvo(schema(value_type = super::sealed::Quant))]
    pub quant_type: Quant,
    /// Precision for intermediate tensors (`Fp16` or `Fp32`).
    pub precision: Precision,
    /// Maximum tokens to be processed in parallel at once.
    #[derivative(Default(value = "128"))]
    pub token_chunk_size: usize,
    /// Number of states that are cached on GPU.
    #[derivative(Default(value = "8"))]
    pub max_batch: usize,
    /// Device to put the embed tensor.
    #[salvo(schema(value_type = super::sealed::EmbedDevice))]
    pub embed_device: EmbedDevice,
}

/// Low-rank adaptor.
#[derive(Debug, Clone, Derivative, Serialize, Deserialize, ToSchema)]
#[derivative(Default)]
#[serde(default)]
pub struct Lora {
    /// Path to the LoRA.
    #[salvo(schema(value_type = String))]
    pub path: PathBuf,
    /// Blend factor.
    #[derivative(Default(value = "1.0"))]
    pub alpha: f32,
}

/// State-tuned initial state.
#[derive(Debug, Clone, Derivative, Serialize, Deserialize, ToSchema)]
#[derivative(Default)]
#[serde(default)]
pub struct State {
    /// Path to the initial state.
    #[salvo(schema(value_type = String))]
    pub path: PathBuf,
    /// Given name for the state.
    pub name: Option<String>,
    /// UUID for this state.
    #[serde(default = "StateId::new")]
    pub id: StateId,
    /// If this state should be loaded on startup.
    pub default: bool,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize, ToSchema)]
#[derivative(Default)]
#[serde(default)]
pub struct Tokenizer {
    #[derivative(Default(value = "\"assets/tokenizer/rwkv_vocab_v20230424.json\".into()"))]
    #[salvo(schema(value_type = String))]
    pub path: PathBuf,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize, ToSchema)]
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

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, ToSchema)]
pub enum Precision {
    #[default]
    Fp16,
    Fp32,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, ToSchema)]
pub enum AdapterOption {
    #[default]
    Auto,
    Economical,
    Manual(usize),
}
