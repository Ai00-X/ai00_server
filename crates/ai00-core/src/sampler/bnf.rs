use bit_set::BitSet;
use bnf_sampler::sampler::{AcceptTokenResult, PossibleTokensResult, Sampler};

use super::Transformer;

#[derive(Debug)]
pub struct BnfSampler {
    sampler: Sampler,
    current_token_ids: BitSet,
}

impl BnfSampler {
    pub fn new(mut sampler: Sampler) -> Self {
        let current_token_ids = match sampler.all_possible_next_tokens(None) {
            Ok(PossibleTokensResult::Continue(tokens)) => tokens.clone(),
            _ => BitSet::new(),
        };
        Self {
            sampler,
            current_token_ids,
        }
    }

    #[inline]
    pub fn current_token_ids(&self) -> &BitSet {
        &self.current_token_ids
    }
}

impl std::ops::Deref for BnfSampler {
    type Target = Sampler;

    fn deref(&self) -> &Self::Target {
        &self.sampler
    }
}

impl std::ops::DerefMut for BnfSampler {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.sampler
    }
}

impl Transformer for BnfSampler {
    type Output = AcceptTokenResult;

    fn transform(&self, output: &mut [f32]) {
        output
            .iter_mut()
            .enumerate()
            .filter(|&(token, _)| !self.current_token_ids().contains(token))
            .for_each(|(_, logits)| *logits = f32::MIN)
    }

    fn update(&mut self, token: u16) -> AcceptTokenResult {
        let token = Some(token as u32);
        let res = self.accept_a_token(token).expect("invalid input token");
        self.current_token_ids = match self.sampler.all_possible_next_tokens(None) {
            Ok(PossibleTokensResult::Continue(tokens)) => tokens.clone(),
            _ => BitSet::new(),
        };
        res
    }
}
