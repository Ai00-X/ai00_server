pub mod mirostat;
pub mod nucleus;

pub trait Sampler {
    /// Initialize the sampler state.
    fn init(&mut self, model_tokens: &[u16]);
    /// Update the raw model output.
    fn transform(&self, output: &mut [f32]);
    /// Select one token from the distribution.
    fn sample(&self, probs: &[f32]) -> u16;
    /// Update the sampler state after a token is chosen.
    fn update(&mut self, token: u16);
}
