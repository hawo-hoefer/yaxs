use super::Parameter;
use crate::background::Background;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BackgroundSpec {
    Chebyshev {
        coefs: Vec<Parameter<f32>>,
        scale: Parameter<f32>,
    },
    Exponential {
        slope: Parameter<f32>,
        scale: Parameter<f32>,
    },
}

impl BackgroundSpec {
    pub fn generate_bkg(&self, rng: &mut impl Rng) -> Background {
        match self {
            BackgroundSpec::Chebyshev { ref coefs, scale } => {
                let mut coef = Vec::<f32>::with_capacity(coefs.len());
                for param in coefs.iter() {
                    coef.push(param.generate(rng));
                }
                let scale = scale.generate(rng);
                Background::Chebyshev { coef, scale }
            }
            BackgroundSpec::Exponential { slope, scale } => Background::Exponential {
                slope: slope.generate(rng),
                scale: scale.generate(rng),
            },
        }
    }
}
