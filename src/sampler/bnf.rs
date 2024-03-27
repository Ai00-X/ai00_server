use anyhow::Result;
use bit_set::BitSet;
use bnf_sampler::sampler::{PossibleTokensResult, Sampler};

#[derive(Debug)]
pub struct BnfSampler {
    sampler: Sampler,
    current_token_ids: BitSet,
}

impl BnfSampler {
    pub fn new(mut sampler: Sampler) -> Result<Self> {
        let current_token_ids = match sampler.all_possible_next_tokens(None)? {
            PossibleTokensResult::Continue(token_ids) => token_ids.clone(),
            _ => unreachable!(),
        };
        Ok(Self {
            sampler,
            current_token_ids,
        })
    }

    #[inline]
    pub fn current_token_ids(&self) -> &BitSet {
        &self.current_token_ids
    }

    pub fn update(&mut self) -> Result<()> {
        self.current_token_ids = match self.sampler.all_possible_next_tokens(None)? {
            PossibleTokensResult::Continue(tokens) => tokens.clone(),
            _ => unreachable!(),
        };
        Ok(())
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
