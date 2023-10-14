use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::ReloadRequest;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub tokenizer: TokenizerConfig,
    pub adapter: AdapterOption,
}

impl From<Config> for ReloadRequest {
    fn from(value: Config) -> Self {
        let ModelConfig {
            path: model_path,
            quant,
            token_chunk_size,
            head_chunk_size,
            max_runtime_batch,
            max_batch,
            embed_layer,
        } = value.model;
        let TokenizerConfig {
            path: tokenizer_path,
        } = value.tokenizer;
        let adapter = value.adapter;

        Self {
            model_path,
            quant,
            token_chunk_size,
            head_chunk_size,
            max_runtime_batch,
            max_batch,
            embed_layer,
            tokenizer_path,
            adapter,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the model.
    pub path: PathBuf,
    /// Specify layers that needs to be quantized.
    pub quant: Vec<usize>,
    /// Maximum tokens to be processed in parallel at once.
    pub token_chunk_size: usize,
    /// The chunk size for each split of the head matrix.
    pub head_chunk_size: usize,
    /// Maximum number of batches that are active at once.
    pub max_runtime_batch: usize,
    /// Number of states that are cached on GPU.
    pub max_batch: usize,
    /// the (reversed) number of layer at which the output is as embedding.
    pub embed_layer: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub path: PathBuf,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum AdapterOption {
    #[default]
    Auto,
    Economical,
    Manual(usize),
}
