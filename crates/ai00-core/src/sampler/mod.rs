pub mod bnf;
pub mod mirostat;
pub mod nucleus;
pub mod typical;
mod utils;
pub trait Sampler {
    /// Initialize the sampler state.
    fn init(&mut self, model_tokens: &[u16]);
    /// Update the raw model output.
    fn transform(&self, output: &mut [f32]);
    /// Select one token from the distribution, and also update the state.
    fn sample(&mut self, probs: &[f32]) -> u16;
}

pub trait Formatter {
    /// Update the raw model output.
    fn transform(&self, output: &mut [f32]);
    /// Update the internal state after a token is chosen. Return if the state machine is halt.
    fn update(&mut self, token: u16) -> bool;
}
