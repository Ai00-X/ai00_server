use std::collections::HashMap;

use derivative::Derivative;
use itertools::Itertools;
use salvo::oapi::ToSchema;
use serde::{Deserialize, Serialize};
use voracious_radix_sort::RadixSort;

use super::{radix, Sampler};

#[derive(Debug, Clone, Derivative, Serialize, Deserialize, ToSchema)]
#[derivative(Default)]
#[serde(default)]
pub struct TypicalParams {
    #[derivative(Default(value = "0.5"))]
    pub tau: f32,
    #[derivative(Default(value = "128"))]
    pub top_k: usize,
    #[derivative(Default(value = "1.0"))]
    pub temperature: f32,
    #[derivative(Default(value = "0.3"))]
    pub presence_penalty: f32,
    #[derivative(Default(value = "0.3"))]
    pub frequency_penalty: f32,
    #[derivative(Default(value = "0.99654026"))]
    pub penalty_decay: f32,
}

#[derive(Debug, Default, Clone)]
pub struct TypicalState {
    pub penalties: HashMap<u16, f32>,
}

#[derive(Debug, Default, Clone)]
pub struct TypicalSampler {
    pub params: TypicalParams,
    pub state: TypicalState,
}

impl TypicalSampler {
    pub fn new(params: TypicalParams) -> Self {
        Self {
            params,
            state: Default::default(),
        }
    }
}

impl Sampler for TypicalSampler {
    fn init(&mut self, model_tokens: &[u16]) {
        let TypicalSampler { params, state } = self;
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
        let TypicalSampler { params, state } = self;

        let probs = probs
            .iter()
            .enumerate()
            .filter(|(_, &x)| x > 0.0)
            .map(|(id, &x)| (id, x, -x.ln()))
            .collect_vec();
        let entropy = probs.iter().map(|(_, x, y)| x * y).sum::<f32>();
        let mut sorted = probs
            .into_iter()
            .map(|(id, x, y)| radix::DoubleF32WithIndex(id, x, (y - entropy).abs()))
            .collect_vec();
        sorted.voracious_sort();
        let sorted = sorted
            .into_iter()
            .map(|radix::DoubleF32WithIndex(id, x, _)| (id, x))
            .take(params.top_k)
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                if *cum > params.tau {
                    None
                } else {
                    *cum += x;
                    Some((id, *cum, x))
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
