use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::pattern::{ImpurityPeak, Peak};

use super::{Parameter, Probability};

#[derive(Serialize, Debug, Clone)]
pub struct ImpuritySpec {
    d_hkl_ams: Parameter<f64>,
    intensity: Parameter<f64>,
    eta: Parameter<f64>,
    mean_ds_nm: Parameter<f64>,
    probability: Option<f64>,
    n_peaks: Option<usize>,
}

impl ImpuritySpec {
    pub fn new(
        d_hkl_ams: Parameter<f64>,
        intensity: Parameter<f64>,
        eta: Parameter<f64>,
        mean_ds_nm: Parameter<f64>,
        probability: Option<Probability>,
        n_peaks: Option<usize>,
    ) -> Self {
        Self {
            d_hkl_ams,
            intensity,
            eta,
            mean_ds_nm,
            probability: probability.map(|p| p.into()),
            n_peaks,
        }
    }
}

impl<'de> Deserialize<'de> for ImpuritySpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct _ImpuritySpec {
            d_hkl_ams: Parameter<f64>,
            intensity: Parameter<f64>,
            eta: Parameter<f64>,
            mean_ds_nm: Parameter<f64>,
            probability: Option<Probability>,
            n_peaks: Option<usize>,
        }

        let _ImpuritySpec {
            d_hkl_ams,
            intensity,
            eta,
            mean_ds_nm,
            probability,
            n_peaks,
        } = _ImpuritySpec::deserialize(deserializer)?;

        Ok(ImpuritySpec::new(
            d_hkl_ams,
            intensity,
            eta,
            mean_ds_nm,
            probability,
            n_peaks,
        ))
    }
}

pub fn generate_impurities(
    impurity_specs: &[ImpuritySpec],
    rng: &mut impl Rng,
) -> Box<[ImpurityPeak]> {
    let mut impurity_peaks = Vec::new();
    for spec in impurity_specs.iter() {
        for _ in 0..spec.n_peaks.unwrap_or(1) {
            if !spec.probability.map(|p| rng.random_bool(p)).unwrap_or(true) {
                continue;
            }
            let d_hkl = spec.d_hkl_ams.generate(rng);
            let i_hkl = spec.intensity.generate(rng);
            impurity_peaks.push(ImpurityPeak {
                peak: Peak {
                    d_hkl,
                    i_hkl,
                    hkls: Vec::new(),
                },
                eta: spec.eta.generate(rng),
                mean_ds_nm: spec.mean_ds_nm.generate(rng),
            })
        }
    }
    impurity_peaks.into()
}
