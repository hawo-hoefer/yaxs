use log::{error, warn};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::pattern::{ImpurityPeak, Peak};

use super::{Parameter, Probability};

#[derive(Serialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
/// Specification for a phenomenological impurity peak
///
/// * `d_hkl_ams`: d_hkl for the peak
/// * `intensity`: peak intensity
/// * `eta`: peak eta for peak shape
/// * `mean_ds_nm`: mean domain size for peak shape
/// * `probability`: probability for a single peak to be in output. For each peak, presence will be
/// evaluated separately
/// * `n_peaks`: number of peaks to generate
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

    pub fn validate_d_hkl_or_adjust(&mut self, lb: f64, ub: f64) -> Option<()> {
        match self.d_hkl_ams {
            Parameter::Fixed(d_hkl_ams) => {
                if d_hkl_ams > ub || d_hkl_ams < lb {
                    error!(
                        "Impurity position {} outside of visible range [{}, {}]",
                        d_hkl_ams, lb, ub
                    );
                    return None;
                }
            }
            Parameter::Range(ref mut lo, ref mut hi) => {
                if *lo < lb {
                    warn!("Invalid impurity definition. Lower d_hkl_ams bound {lo} may put impurity outside of visible area. Adjusting to {lb}...",
                    );
                    *lo = lb;
                }

                if *hi > ub {
                    warn!("Invalid impurity definition. Upper d_hkl_ams bound {hi} may put impurity outside of visible area. Adjusting to {ub}...",
                    );
                    *hi = ub;
                }
            }
        }
        Some(())
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
