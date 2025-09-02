use rand::Rng;
use serde::{Deserialize, Serialize};

use super::Parameter;
use crate::math::linalg::Vec3;
use crate::preferred_orientation::MarchDollase;

#[derive(PartialEq, Debug, Serialize, Deserialize, Clone)]
pub struct MarchDollaseCfg {
    pub hkl: Vec3<f64>,
    pub r: Parameter<f64>,
}

impl MarchDollaseCfg {
    pub fn generate(&self, rng: &mut impl Rng) -> MarchDollase {
        MarchDollase {
            hkl: self.hkl.clone(),
            r: self.r.generate(rng),
        }
    }
}
