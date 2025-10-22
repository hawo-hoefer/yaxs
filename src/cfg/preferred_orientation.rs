use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::Parameter;
use crate::math::linalg::{ColVec, Mat, Mat4, Vec4};
use crate::math::stats::{
    sample_unit_quaternion_subgroup_algorithm, BinghamDistribution, HitAndRunPolytopeSampler,
};
use crate::pattern::{
    uniform_sample_no_replacement_knuth, uniform_sample_no_replacement_knuth_arr,
};
use crate::preferred_orientation::{BinghamODF, BinghamParams, N_SAMPLES};

#[derive(PartialEq, Debug, Serialize, Deserialize, Clone)]
pub enum POCfg {
    FullEpitaxialGrowth {
        k_max: f64,
        strength: Parameter<f64>,
    },
    SingleAxis {
        k_max: f64,
        strength: Parameter<f64>,
    },
    DirectBingham {
        k: Vec4<f64>,
        orientation: Vec4<f64>,
    },
}

#[derive(Clone)]
pub enum POGenerator {
    FullEpitaxialGrowth {
        sampler: HitAndRunPolytopeSampler<13, 4>,
    },
    SingleAxis {
        sampler: HitAndRunPolytopeSampler<14, 3>,
    },
    Exact {
        k: Vec4<f64>,
        orientation: Vec4<f64>,
    },
}

impl POGenerator {
    pub fn sample(&mut self, rng: &mut impl Rng) -> BinghamODF {
        let (k, orientation) = match self {
            POGenerator::FullEpitaxialGrowth { sampler } => {
                let orientation = sample_unit_quaternion_subgroup_algorithm(rng);
                (sampler.sample(rng), orientation)
            }
            POGenerator::SingleAxis { sampler } => {
                // k4 = k2 + k3 - k1
                let k123 = sampler.sample(rng);
                let orientation = sample_unit_quaternion_subgroup_algorithm(rng);
                (k123.extend(k123[1] + k123[2] - k123[0]), orientation)
            }
            POGenerator::Exact { k, orientation } => (k.clone(), orientation.clone()),
        };

        let mut indices = [0, 1, 2, 3];
        indices.shuffle(rng);
        let [i1, i2, i3, i4] = indices;

        let k = Vec4::new(k[i1], k[i2], k[i3], k[i4]);

        let bingham_dist = BinghamDistribution::try_new(k.clone(), Mat4::identity())
            .expect("identity matrix is OK");
        let mut bingham_samples = Vec::with_capacity(N_SAMPLES);
        for _ in 0..bingham_samples.capacity() {
            bingham_samples.push(bingham_dist.sample(rng));
        }

        BinghamODF {
            params: BinghamParams { orientation, ks: k },
            axis_aligned_bingham_dist_samples: bingham_samples,
        }
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
        match self {
            FullEpitaxialGrowth { k_max, strength } => {
                #[rustfmt::skip]
                let a = Mat::from_rows([
                    // upper bounds
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],

                    // // lower bounds
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],


                    // ordering of ks: k1 >= k2 >= k3 >= k4
                    // -k1 + k2 <= 0
                    [-1.0, 1.0, 0.0, 0.0],
                    // -k2 + k3 <= 0
                    [0.0, -1.0, 1.0, 0.0],
                    // -k3 + k4 <= 0
                    [0.0, 0.0, -1.0, 1.0],

                    // strength_low * (k2 + k3) <= (k1 + k4)
                    [-1.0, strength.lower_bound(), strength.lower_bound(), -1.0],
                    // -(k1 + k4) + strength * (k2 + k3) <= 0

                    // strength_high * (k2 + k3) >= (k1 + k4)
                    [1.0, -strength.upper_bound(), -strength.upper_bound(), 1.0],
                    // (k1 + k4) - strength_high * (k2 + k3) <= 0
                ]);

                let k_max = *k_max;
                let b = ColVec::from_col([
                    k_max, k_max, k_max, k_max, // upper bounds
                    0.0, 0.0, 0.0, 0.0, // lower bounds
                    0.0, 0.0, 0.0, // ordering
                    0.0, 0.0, // bipolar
                ]);
                let sampler = HitAndRunPolytopeSampler::try_new(a, b, 10000, 100, rng)?;
                Ok(POGenerator::FullEpitaxialGrowth { sampler })
            }
            SingleAxis { k_max, strength } => {
                // corresponds to the circular mode of the bingham distribution
                // k1 + k4 = k2 + k3
                // k4 = k2 + k3 - k1
                //
                // sample k1, k2, k3, and compute k4
                #[rustfmt::skip]
                let a = Mat::from_rows([
                    // upper bounds
                    [ 1.0,      0.0,      0.0],
                    [ 0.0,      1.0,      0.0],
                    [ 0.0,      0.0,      1.0],

                    // lower bounds
                    [-1.0,      0.0,      0.0],
                    [ 0.0,     -1.0,      0.0],
                    [ 0.0,      0.0,     -1.0],

                    // actual constraints

                    // k1 >= k2 >= k3 >= k4
                    [-1.0,      1.0,      0.0],
                    // -k1 + k2 <= 0
                    [ 0.0,     -1.0,      1.0],
                    // -k2 + k3 <= 0

                    // additional constraint from the transformation due to the equality
                    // k4 <= k_max
                    // -k1 + k2 + k3 <= k_max 
                    [-1.0, 1.0, 1.0],

                    // k4 >= 0
                    // - k4 <= 0
                    // -(-k1 + k2 + k3) <= 0 
                    // k1 - k2 - k3 <= 0
                    [1.0, -1.0, -1.0],

                    // something with strength
                    // k1 >= strength_lo * k4
                    // k4 = k2 + k3 - k1
                    // k1 >= strength_lo * k2 + strength_lo * k3 - strength_lo * k1
                    // k1 * (1 + strength_lo) >= strength_lo * k2 + strength_lo * k3
                    // k1 * (1 + strength_lo) - k2 * strength_lo - k3 * strength_lo >= 0
                    // -k1 * (1 + strength_lo) + k2 * strength_lo + k3 * strength_lo <= 0
                    [-(1.0 + strength.lower_bound()), strength.lower_bound(), strength.lower_bound()],

                    // k1 <= strength_hi * k4
                    // k4 = k2 + k3 - k1
                    // k1 <= strength_hi * k2 + strength_hi * k3 - strength_hi * k1
                    // k1 * (1 + strength_hi) <= strength_hi * k2 + strength_hi * k3
                    // k1 * (1 + strength_hi) - k2 * strength_hi - k3 * strength_hi <= 0
                    [(1.0 + strength.upper_bound()), -strength.upper_bound(), -strength.upper_bound()],

                    // k2 >= strength_lo * k3
                    // -k2 + strength_lo * k3 <= 0
                    [0.0, -1.0, strength.lower_bound()],

                    // k2 <= strength_hi * k3
                    // k2 - strength_hi * k3 <= 0
                    [0.0, 1.0, -strength.upper_bound()],
                ]);

                let k_max = *k_max;
                let b = ColVec::from_col([
                    k_max, k_max, k_max, // upper bounds
                    0.0, 0.0, 0.0, // lower bounds
                    k_max, 0.0, // bounds for implicit k4
                    0.0, 0.0, // ordering
                    0.0, 0.0, // circular
                    0.0, 0.0, // circular
                ]);
                let sampler = HitAndRunPolytopeSampler::try_new(a, b, 1000000, 1000, rng)?;
                Ok(POGenerator::SingleAxis { sampler })
            }
            DirectBingham { k, orientation } => Ok(POGenerator::Exact {
                k: k.clone(),
                orientation: orientation.clone(),
            }),
        }
    }
}
