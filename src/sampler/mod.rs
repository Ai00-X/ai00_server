pub mod bnf;
pub mod mirostat;
pub mod nucleus;

#[allow(unused_variables)]
pub trait Sampler {
    /// Initialize the sampler state.
    fn init(&mut self, model_tokens: &[u16]) {}
    /// Update the raw model output.
    fn transform(&self, output: &mut [f32]) {}
    /// Select one token from the distribution, and also update the state.
    fn sample(&mut self, probs: &[f32]) -> u16;
}
