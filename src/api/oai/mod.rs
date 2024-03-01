use std::sync::Arc;

use serde::Deserialize;
use tokio::sync::RwLock;

pub mod chat;
pub mod completion;
pub mod embedding;
pub mod models;

pub use chat::chat_completions;
pub use completion::completions;
pub use embedding::embeddings;
pub use models::models;

use crate::sampler::{
    mirostat::{MirostatParams, MirostatSampler},
    nucleus::{NucleusParams, NucleusSampler},
    Sampler,
};

#[derive(Debug, Default, Clone, Deserialize)]
pub struct SamplerParams {
    #[serde(flatten)]
    nucleus: Option<NucleusParams>,
    #[serde(flatten)]
    mirostat: Option<MirostatParams>,
}

impl From<SamplerParams> for Arc<RwLock<dyn Sampler + Send + Sync>> {
    fn from(value: SamplerParams) -> Self {
        let SamplerParams { nucleus, mirostat } = value;
        match (nucleus, mirostat) {
            (None, None) => Arc::new(RwLock::new(NucleusSampler::new(Default::default()))),
            (None, Some(params)) => Arc::new(RwLock::new(MirostatSampler::new(params))),
            (Some(params), _) => Arc::new(RwLock::new(NucleusSampler::new(params))),
        }
    }
}
