use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::Parameter;
use crate::math::linalg::{ColVec, Mat, Mat4, Vec3, Vec4};
use crate::math::quaternion::Quaternion;
use crate::math::stats::{
    sample_unit_quaternion_subgroup_algorithm, BinghamDistribution, HitAndRunPolytopeSampler,
};
use crate::preferred_orientation::{BinghamParams, KDEBinghamODF};

#[derive(Deserialize, Serialize, Copy, Clone, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub struct KDEApprox {
    pub n: usize,
    pub kappa: f64,
}

impl KDEApprox {
    pub fn normalization_constant(&self) -> f64 {
        self.kappa
            / (std::f64::consts::TAU * (self.kappa.exp() - (-self.kappa).exp()))
            / self.n as f64
    }
}

impl Default for KDEApprox {
    fn default() -> Self {
        Self {
            n: 2048,
            kappa: 20.0,
        }
    }
}

#[derive(PartialEq, Debug, Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub enum POCfg {
    FullEpitaxialGrowth {
        k_max: f64,
        strength: Parameter<f64>,
        sampling: Option<KDEApprox>,
    },
    /// maximum value for the bingham parameter distribution
    /// will sample a distribution with one axis being aliged and the
    /// other two axes having the same distribution (on a fuzzy circle)
    SingleAxis {
        concentration: Parameter<f64>,
        sampling: Option<KDEApprox>,
    },
    DirectBingham {
        k: Vec3<f64>,
        orientation: Vec4<f64>,
        sampling: Option<KDEApprox>,
    },
}

#[derive(Clone)]
pub enum POGenerator {
    FullEpitaxialGrowth {
        sampler: HitAndRunPolytopeSampler<10, 3>,
        sampling: KDEApprox,
    },
    SingleAxis {
        concentration: Parameter<f64>,
        sampling: KDEApprox,
    },
    Exact {
        k: Vec3<f64>,
        orientation: Vec4<f64>,
        sampling: KDEApprox,
    },
}

impl POGenerator {
    pub fn sampling_parameters(&self) -> KDEApprox {
        match self {
            POGenerator::FullEpitaxialGrowth { sampling, .. } => *sampling,
            POGenerator::SingleAxis { sampling, .. } => *sampling,
            POGenerator::Exact { sampling, .. } => *sampling,
        }
    }

    fn sample_hyper(
        &mut self,
        rng: &mut impl Rng,
    ) -> (BinghamDistribution<4>, Vec4<f64>, &mut KDEApprox) {
        let (k, orientation, sampling) = match self {
            POGenerator::FullEpitaxialGrowth { sampler, sampling } => {
                let orientation = sample_unit_quaternion_subgroup_algorithm(rng);
                let ks = sampler.sample(rng);
                (ks, orientation, sampling)
            }
            POGenerator::SingleAxis {
                concentration,
                sampling,
            } => {
                // k4 = k2 + k3 - k1
                // k4 = 0
                // k3 = k1 - k2
                let k = rng.random_range(concentration.lower_bound()..=concentration.upper_bound());
                let orientation = sample_unit_quaternion_subgroup_algorithm(rng);
                let ks = Vec3::new(k, k, 0.0);
                (ks, orientation, sampling)
            }
            POGenerator::Exact {
                k,
                orientation,
                sampling,
            } => (k.clone(), orientation.clone(), sampling),
        };

        let mut indices = [0, 1, 2];
        indices.shuffle(rng);
        let [i1, i2, i3] = indices;

        let k = Vec4::new(k[i1], k[i2], k[i3], 0.0);

        let bingham_dist =
            BinghamDistribution::try_new(k, Mat4::identity()).expect("identity matrix is OK");

        (bingham_dist, orientation, sampling)
    }

    pub fn sample_into(&mut self, rng: &mut impl Rng, dst: &mut Vec<Quaternion>) -> BinghamParams {
        let (bingham_dist, orientation, params) = self.sample_hyper(rng);

        for _ in 0..params.n {
            let sample = bingham_dist.sample(rng);
            dst.push(sample.into());
        }

        BinghamParams {
            orientation: orientation.into(),
            ks: bingham_dist.ks().clone().into(),
        }
    }

    pub fn sample(&mut self, rng: &mut impl Rng) -> KDEBinghamODF {
        let (bingham_dist, orientation, sampling) = self.sample_hyper(rng);

        let mut bingham_samples = Vec::with_capacity(sampling.n);
        for _ in 0..bingham_samples.capacity() {
            let sample = bingham_dist.sample(rng);
            bingham_samples.push(sample.into());
        }

        KDEBinghamODF::new(
            orientation.into(),
            bingham_dist.ks().clone().into(),
            bingham_samples,
            sampling.kappa,
            sampling.normalization_constant(),
        )
    }
}

enum IneqOp {
    LessOrEq,
    GreaterOrEq,
}

struct IneqConstraint<const NPARAMS: usize> {
    coefs: [f64; NPARAMS],
    rhs: f64,
    op: IneqOp,
}

impl<const N_PARAMS: usize> IneqConstraint<N_PARAMS> {
    pub fn ge(coefs: [f64; N_PARAMS], rhs: f64) -> Self {
        Self {
            coefs,
            rhs,
            op: IneqOp::GreaterOrEq,
        }
    }

    pub fn le(coefs: [f64; N_PARAMS], rhs: f64) -> Self {
        Self {
            coefs,
            rhs,
            op: IneqOp::LessOrEq,
        }
    }

    pub fn to_coef_rhs(&self) -> ([f64; N_PARAMS], f64) {
        // the solver wants everything as less-or-equal constraints
        match self.op {
            IneqOp::LessOrEq => {
                // just return everything as is
                let lhs = self.coefs;
                let rhs = self.rhs;
                return (lhs, rhs);
            }
            IneqOp::GreaterOrEq => {
                // a1 k1 + a2 k2 ... >= rhs
                // -a1 k1 - a2 k2 ... <= -rhs
                let mut lhs = self.coefs;
                let rhs = -self.rhs;
                for c in lhs.iter_mut() {
                    *c = -*c;
                }

                return (lhs, rhs);
            }
        }
    }
}

struct Constraints<const N_CONSTR: usize, const N_PARAMS: usize>(
    pub [IneqConstraint<N_PARAMS>; N_CONSTR],
);

impl<const N_CONSTR: usize, const N_PARAMS: usize> Constraints<N_CONSTR, N_PARAMS> {
    pub fn to_lhs_rhs(self) -> (Mat<f64, N_CONSTR, N_PARAMS>, ColVec<f64, N_CONSTR>) {
        let mut lhs_rows = [[0.0; N_PARAMS]; N_CONSTR];
        let mut rhs_col = [0.0; N_CONSTR];

        for (i, c) in self.0.iter().enumerate() {
            let (coef, rhs) = c.to_coef_rhs();
            lhs_rows[i] = coef;
            rhs_col[i] = rhs;
        }

        (Mat::from_rows(lhs_rows), ColVec::from_col(rhs_col))
    }
}

impl POCfg {
    pub fn try_into_generator(&self, rng: &mut impl Rng) -> Result<POGenerator, String> {
        use POCfg::*;
        // NOTE: I don't want to implement and understand the math to randomly sample
        // an orthogonal 4x4 matrix. So to avoid that, we generate an axis-aligned
        // Bingham distribution instead, and rotate hkl vectors using a randomly sampled
        // orientation.

        // NOTE: We rely on the description of the bingham distribution from the MTEX
        // toolbox (https://mtex-toolbox.github.io/BinghamODFs.html). Some playing
        // around produced sensible results for distributions over orientations,
        // so we're going with that until we encounter problems.
        //
        // TODO: currently, strength must be a range, otherwise the solver will fail - can we do something about that?
        match self {
            FullEpitaxialGrowth {
                k_max,
                strength,
                sampling,
            } => {
                // corresponds to the bipolar mode of the bingham distribution
                //
                let k_max = *k_max;
                let s_lo = strength.lower_bound();
                let _ = strength.upper_bound();
                let constraints = Constraints([
                    // k1 in [0, k_max]
                    IneqConstraint::ge([1.0, 0.0, 0.0], 0.0),
                    IneqConstraint::le([1.0, 0.0, 0.0], k_max),
                    // k2 in [0, k_max]
                    IneqConstraint::ge([0.0, 1.0, 0.0], 0.0),
                    IneqConstraint::le([0.0, 1.0, 0.0], k_max),
                    // k3 in [0, k_max]
                    IneqConstraint::ge([0.0, 0.0, 1.0], 0.0),
                    IneqConstraint::le([0.0, 0.0, 1.0], k_max),
                    // coefficient ordering k1 >= k2 >= k3
                    // k1 >= k2
                    IneqConstraint::ge([1.0, -1.0, 0.0], 0.0),
                    // k2 >= k3
                    IneqConstraint::ge([0.0, 1.0, -1.0], 0.0),
                    // bipolar distribution if k1 + k4 > k2 + k3 (see MTEX link above)
                    // k4 = 0
                    // so k1 > k2 + k3
                    // concentration of distribution is stronger, the larger k1 is compared to k2 + k3
                    // we use strength to set the ratio of those two values
                    // k1 / (k2 + k3) > strength_low  => k1 -  strength_low * k2 -  strength_low * k3 > 0
                    // k1 / (k2 + k3) < strength_high => k1 - strength_high * k2 - strength_high * k3 < 0
                    IneqConstraint::ge([1.0, -s_lo, -s_lo], 0.0),
                    IneqConstraint::ge([1.0, -s_lo, -s_lo], 0.0),
                ]);

                let (a, b) = constraints.to_lhs_rhs();
                let sampler = HitAndRunPolytopeSampler::try_new(a, b, 10000, 100, rng)?;
                Ok(POGenerator::FullEpitaxialGrowth {
                    sampler,
                    sampling: sampling.unwrap_or_default(),
                })
            }
            SingleAxis {
                concentration,
                sampling,
            } => Ok(POGenerator::SingleAxis {
                concentration: concentration.clone(),
                sampling: sampling.unwrap_or_default(),
            }),
            DirectBingham {
                k,
                orientation,
                sampling,
            } => Ok(POGenerator::Exact {
                k: k.clone(),
                orientation: orientation.clone(),
                sampling: sampling.unwrap_or_default(),
            }),
        }
    }
}
