use derivative::Derivative;

#[derive(Debug, Clone, Derivative)]
#[derivative(Default)]
pub struct MirostatParams {
    #[derivative(Default(value = "3.0"))]
    pub tau: f32,
}
