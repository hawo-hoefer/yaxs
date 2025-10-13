use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::structure::{Strain, Structure};

use super::{POCfg, Parameter, VolumeFraction};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub enum StrainCfg {
    Maximum(f64),
    Ortho([Parameter<f64>; 3]),
    Full([Parameter<f64>; 6]),
}

pub fn apply_strain_cfg(
    cfg: &Option<StrainCfg>,
    s: &Structure,
    rng: &mut impl Rng,
) -> Option<(Structure, Strain)> {
    use StrainCfg::*;
    match cfg {
        Some(Maximum(max_strain)) => Some(s.permute(*max_strain, rng)),
        Some(Ortho(params)) => {
            let strain = Strain::from_diag(
                params[0].generate(rng),
                params[1].generate(rng),
                params[2].generate(rng),
            );
            Some((s.apply_strain(&strain), strain))
        }
        Some(Full(params)) => {
            let strain = Strain::new_verified([
                params[0].generate(rng),
                params[1].generate(rng),
                params[2].generate(rng),
                params[3].generate(rng),
                params[4].generate(rng),
                params[5].generate(rng),
            ])?;
            Some((s.apply_strain(&strain), strain))
        }
        None => Some((s.clone(), Strain::none())),
    }
}

#[derive(PartialEq, Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct StructureDef {
    pub path: String,
    pub preferred_orientation: Option<POCfg>,
    pub strain: Option<StrainCfg>,
    pub volume_fraction: Option<VolumeFraction>,
    pub mean_ds_nm: Parameter<f64>,
}
