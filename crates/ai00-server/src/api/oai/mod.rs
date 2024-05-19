use std::sync::Arc;

use ai00_core::sampler::{
    mirostat::{MirostatParams, MirostatSampler},
    nucleus::{NucleusParams, NucleusSampler},
    Sampler,
};
use salvo::oapi::ToSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

pub mod chat;
pub mod completion;
pub mod embedding;
pub mod info;

pub use chat::chat_completions;
pub use completion::completions;
pub use embedding::embeddings;
pub use info::models;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum SamplerParams {
    Mirostat(MirostatParams),
    Nucleus(NucleusParams),
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self::Nucleus(Default::default())
    }
}

impl From<SamplerParams> for Arc<RwLock<dyn Sampler + Send + Sync>> {
    fn from(value: SamplerParams) -> Self {
        match value {
            SamplerParams::Mirostat(params) => Arc::new(RwLock::new(MirostatSampler::new(params))),
            SamplerParams::Nucleus(params) => Arc::new(RwLock::new(NucleusSampler::new(params))),
        }
    }
}
