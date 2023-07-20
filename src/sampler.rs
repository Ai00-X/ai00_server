use itertools::Itertools;

#[derive(Debug, Clone)]
pub struct Sampler {
    pub top_p: f32,
    pub temperature: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Sampler {
    fn softmax(data: Vec<f32>) -> Vec<f32> {
        let exp = data.into_iter().map(f32::exp).collect_vec();
        let sum: f32 = exp.iter().sum();
        exp.into_iter().map(|x| x / sum).collect()
    }

    pub fn sample(&self, logits: Vec<f32>) -> u16 {
        let probs = Self::softmax(logits);
        let sorted = probs
            .into_iter()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                if *cum > self.top_p {
                    None
                } else {
                    *cum += x;
                    Some((id, *cum, x))
                }
            })
            .map(|(id, _, x)| (id, x.powf(1.0 / self.temperature)))
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
        token as u16
    }
}
