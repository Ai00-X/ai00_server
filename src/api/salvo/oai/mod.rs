use std::sync::Arc;

use salvo::{macros::Extractible, oapi::ToSchema};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

pub mod chat;
pub mod completion;
pub mod embedding;
pub mod models;

pub use chat::salvo_oai_chat_completions;
pub use completion::salvo_oai_completions;
pub use embedding::salvo_oai_embeddings;
pub use models::salvo_oai_models;

use crate::sampler::{
    mirostat::{MirostatParams, MirostatSampler},
    nucleus::{NucleusParams, NucleusSampler},
    Sampler,
};

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum SamplerParams {
    Nucleus(NucleusParams),
    Mirostat(MirostatParams),
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self::Nucleus(Default::default())
    }
}

impl From<SamplerParams> for Arc<RwLock<dyn Sampler + Send + Sync>> {
    fn from(value: SamplerParams) -> Self {
        match value {
            SamplerParams::Nucleus(params) => Arc::new(RwLock::new(NucleusSampler::new(params))),
            SamplerParams::Mirostat(params) => Arc::new(RwLock::new(MirostatSampler::new(params))),
        }
    }
}
