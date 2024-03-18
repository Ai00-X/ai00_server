use std::collections::HashMap;

use derivative::Derivative;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::Sampler;

#[cfg(feature = "salvo-api")]
use salvo::oapi::ToSchema;

#[cfg(feature = "salvo-api")]
#[derive(Debug, Clone, Derivative, Serialize, Deserialize, ToSchema)]
#[derivative(Default)]
pub struct NucleusParams {
    #[derivative(Default(value = "1.0"))]
    pub top_p: f32,
    #[derivative(Default(value = "1.0"))]
    pub temperature: f32,
    #[derivative(Default(value = "0.0"))]
    pub presence_penalty: f32,
    #[derivative(Default(value = "0.0"))]
    pub frequency_penalty: f32,
    #[derivative(Default(value = "0.99654026"))]
    #[serde(default = "default_penalty_decay")]
    pub penalty_decay: f32,
}

#[cfg(feature = "axum-api")]
#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct NucleusParams {
    #[derivative(Default(value = "1.0"))]
    pub top_p: f32,
    #[derivative(Default(value = "1.0"))]
    pub temperature: f32,
    #[derivative(Default(value = "0.0"))]
    pub presence_penalty: f32,
    #[derivative(Default(value = "0.0"))]
    pub frequency_penalty: f32,
    #[derivative(Default(value = "0.99654026"))]
    #[serde(default = "default_penalty_decay")]
    pub penalty_decay: f32,
}

fn default_penalty_decay() -> f32 {
    NucleusParams::default().penalty_decay
}

#[derive(Debug, Default, Clone)]
pub struct NucleusState {
    pub penalties: HashMap<u16, f32>,
}

#[derive(Debug, Default, Clone)]
pub struct NucleusSampler {
    pub params: NucleusParams,
    pub state: NucleusState,
}

impl NucleusSampler {
    pub fn new(params: NucleusParams) -> Self {
        Self {
            params,
            state: Default::default(),
        }
    }
}

impl Sampler for NucleusSampler {
    fn init(&mut self, model_tokens: &[u16]) {
        let NucleusSampler { params, state } = self;
        for (index, token) in model_tokens.iter().rev().enumerate() {
            let ap = params.presence_penalty;
            let af = params.frequency_penalty;
            let ad = params.penalty_decay;
            let mut penalty = state.penalties.remove(token).unwrap_or(ap);
            penalty += af * ad.powf(index as f32);
            state.penalties.insert(*token, penalty);
        }
    }

    fn transform(&self, output: &mut [f32]) {
        self.state
            .penalties
            .iter()
            // .filter(|(token, _)| !penalty_free_tokens.contains(token))
            .for_each(|(token, penalty)| output[*token as usize] -= penalty)
    }

    fn sample(&mut self, probs: &[f32]) -> u16 {
        let NucleusSampler { params, state } = self;

        let sorted = probs
            .iter()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                if *cum > params.top_p {
                    None
                } else {
                    *cum += x;
                    Some((id, *cum, *x))
                }
            })
            .map(|(id, _, x)| (id, x.powf(1.0 / params.temperature)))
            .collect_vec();

        let sum: f32 = sorted.iter().map(|(_, x)| x).sum();
        let sorted = sorted
            .into_iter()
            .map(|(id, x)| (id, x / sum))
            .scan((0, 0.0), |(_, cum), (id, x)| {
                *cum += x;
                Some((id, *cum))
            })
            .collect_vec();

        let rand = fastrand::f32();
        let token = sorted
            .into_iter()
            .find_or_first(|&(_, cum)| rand <= cum)
            .map(|(id, _)| id)
            .unwrap_or_default();
        let token = token as u16;

        state
            .penalties
            .iter_mut()
            .for_each(|(_, penalty)| *penalty *= params.penalty_decay);

        let penalty = match state.penalties.get(&token) {
            Some(penalty) => penalty + params.frequency_penalty,
            None => params.presence_penalty,
        };
        state.penalties.insert(token, penalty);

        token
    }
}
