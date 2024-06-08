use std::sync::Arc;

use ai00_core::sampler::{
    mirostat::{MirostatParams, MirostatSampler},
    nucleus::{NucleusParams, NucleusSampler},
    typical::{TypicalParams, TypicalSampler},
    Sampler,
};
use salvo::oapi::ToSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

mod chat;
mod choose;
mod completion;
mod embedding;
mod info;

pub use chat::chat_completions;
pub use choose::chooses;
pub use completion::completions;
pub use embedding::embeddings;
pub use info::models;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type")]
pub enum SamplerParams {
    Mirostat(MirostatParams),
    Typical(TypicalParams),
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
            SamplerParams::Typical(params) => Arc::new(RwLock::new(TypicalSampler::new(params))),
            SamplerParams::Nucleus(params) => Arc::new(RwLock::new(NucleusSampler::new(params))),
        }
    }
}
