use super::{radix, Sampler};
use derivative::Derivative;
use itertools::Itertools;
use salvo::oapi::ToSchema;
use serde::{Deserialize, Serialize};
use voracious_radix_sort::RadixSort;

#[derive(Debug, Clone, Derivative, Serialize, Deserialize, ToSchema)]
#[derivative(Default)]
#[serde(default)]
pub struct MirostatParams {
    #[derivative(Default(value = "3.0"))]
    pub tau: f32,
    #[derivative(Default(value = "0.1"))]
    #[serde(alias = "learning_rate")]
    pub rate: f32,
}

#[derive(Debug, Clone, Default)]
pub struct MirostatState {
    pub max_surprise: f32,
}

#[derive(Debug, Clone, Default)]
pub struct MirostatSampler {
    pub params: MirostatParams,
    pub state: MirostatState,
}

impl MirostatSampler {
    pub fn new(params: MirostatParams) -> Self {
        let state = MirostatState {
            max_surprise: params.tau * 2.0,
        };
        Self { params, state }
    }
}

impl Sampler for MirostatSampler {
    fn init(&mut self, _model_tokens: &[u32]) {}

    fn transform(&self, _output: &mut [f32]) {}

    fn sample(&mut self, probs: &[f32]) -> u32 {
        let MirostatSampler { params, state } = self;

        // sort the surprise values and truncate
        let mut sorted = probs
            .iter()
            .copied()
            .enumerate()
            .map(|(id, x)| radix::F32WithIndex(id, x))
            .collect_vec();
        sorted.voracious_sort();
        let sorted = sorted
            .into_iter()
            .rev()
            .scan((0, 0.0, 0.0), |(_, cum, _), radix::F32WithIndex(id, x)| {
                // if *cum > params.top_p {
                //     None
                // } else {
                //     *cum += x;
                //     Some((id, *cum, *x))
                // }
                *cum += x;
                Some((id, *cum, x))
            })
            .collect_vec();
        let k = sorted
            .iter()
            .find_position(|&(_, _, x)| -x.log2() > state.max_surprise)
            .map(|(k, _)| k + 1)
            .unwrap_or(sorted.len());
        let sorted = sorted.into_iter().take(k).collect_vec();

        // normalize the probs
        let sum = sorted.last().map(|(_, x, _)| *x).unwrap();
        let rand = fastrand::f32() * sum;
        let (token, _, prob) = sorted
            .into_iter()
            .find_or_first(|&(_, cum, _)| rand <= cum)
            .unwrap();

        let token_surprise = sum.log2() - prob.log2();
        let error_surprise = token_surprise - params.tau;
        state.max_surprise -= params.rate * error_surprise;
        state.max_surprise = state.max_surprise.min(4.0 * params.tau);

        token as u32
    }
}
