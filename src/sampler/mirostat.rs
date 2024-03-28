use super::Sampler;
use derivative::Derivative;
use itertools::Itertools;
use salvo::oapi::ToSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Derivative, Serialize, Deserialize, ToSchema)]
#[derivative(Default)]
pub struct MirostatParams {
    #[derivative(Default(value = "3.0"))]
    pub tau: f32,
    #[derivative(Default(value = "0.1"))]
    #[serde(alias = "learning_rate")]
    pub rate: f32,
    #[derivative(Default(value = "128"))]
    #[serde(default = "default_threshold")]
    pub threshold: usize,
}

fn default_threshold() -> usize {
    MirostatParams::default().threshold
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

    fn estimate_s(&self, probs: &[f32]) -> f32 {
        assert!(probs.len() >= self.params.threshold);
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..self.params.threshold {
            let b = probs[i] / probs[i + 1];
            let t = (i + 2) as f32 / (i + 1) as f32;
            num += b.ln() * t.ln();
            den += t.ln().powi(2);
        }
        num / den
    }

    fn compute_k(&self, probs: &[f32], s: f32) -> usize {
        let n = probs.len() as f32;
        let tau = self.state.max_surprise;
        let eps = s - 1.0;
        let k = (eps * 2.0_f32.powf(tau) / (1.0 - n.powf(-eps))).powf(1.0 / s);
        k.round() as usize
    }
}

impl Sampler for MirostatSampler {
    fn init(&mut self, _model_tokens: &[u16]) {}

    fn transform(&self, _output: &mut [f32]) {}

    fn sample(&mut self, probs: &[f32]) -> u16 {
        let sorted = probs
            .iter()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                *cum += x;
                Some((id, *cum, *x))
            })
            .collect_vec();
        let sorted_probs = sorted.iter().map(|x| x.2).collect_vec();

        let s = self.estimate_s(&sorted_probs);
        let k = self.compute_k(&sorted_probs, s) + 1;
        let k = k.min(probs.len() - 1);

        let sum = sorted.get(k).map(|&(_, cum, _)| cum).unwrap_or_default();
        let rand = fastrand::f32() * sum;
        let (token, _, prob) = sorted
            .into_iter()
            .find_or_first(|&(_, cum, _)| rand <= cum)
            .unwrap_or_default();

        let token_surprise = (1.0 / prob).log2();
        let error_surprise = token_surprise - self.params.tau;
        self.state.max_surprise -= self.params.rate * error_surprise;

        token as u16
    }
}
