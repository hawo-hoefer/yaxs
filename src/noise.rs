use rand::Rng;

fn std_normal_box_muller_tf(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();

    return (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
}

#[derive(Debug, Clone)]
pub enum Noise {
    Gaussian { sigma: f64 },
    Uniform { min: f64, max: f64 },
}

impl Noise {
    pub fn apply(&self, intensities: &mut [f32], rng: &mut impl Rng) {
        match self {
            Noise::Gaussian { sigma } => {
                for i in intensities.iter_mut() {
                    *i += (std_normal_box_muller_tf(rng) * sigma) as f32;
                }
            }
            Noise::Uniform { min, max } => {
                for i in intensities.iter_mut() {
                    *i += (rng.random::<f64>() * (max - min) + min) as f32;
                }
            }
        }
    }
}
