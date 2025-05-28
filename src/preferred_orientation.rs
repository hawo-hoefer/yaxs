use nalgebra::Vector3;
use rand::Rng;
use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};

use crate::structure::Lattice;

const HKL_NORM_TOL: f64 = 1e-3;

#[derive(PartialEq, Debug, Serialize, Deserialize, Clone)]
pub struct MarchDollaseCfg {
    hkl: Vector3<f64>,
    r: (f64, f64),
}

impl MarchDollaseCfg {
    pub fn generate(&self, rng: &mut impl Rng) -> MarchDollase {
        let r = rng.random_range(self.r.0..=self.r.1);
        MarchDollase { hkl: self.hkl, r }
    }
}

#[derive(PartialEq, Debug, Serialize, Clone)]
pub struct MarchDollase {
    // miller index h
    hkl: Vector3<f64>,
    // march parameter
    r: f64,
}

/// compute the march pole density function
///
/// * `alpha`: angle to HKL vector in radians
/// * `r`: shape parameter of the march functino
fn march(alpha: f64, r: f64) -> f64 {
    (r.powi(2) * alpha.cos().powi(2) + alpha.sin().powi(2) / r).powf(-1.5)
}

impl MarchDollase {
    pub fn new(hkl: Vector3<f64>, r: f64) -> Result<Self, String> {
        Ok(Self { hkl, r })
    }

    /// compute march-dollase scaling of peak intensities
    /// Using equation (1) from Zolotoyabko, E. (2009). J. Appl. Cryst. 42, 513-518.
    /// https://doi.org/10.1107/S0021889809013727
    ///
    /// * `hkl`: hkl vector for scaling
    /// * `lat`: lattice to scale for
    pub fn weight(&self, hkl: &Vector3<f64>, lat: &Lattice) -> f64 {
        if self.r == 1.0 {
            // short circuit if no preferred orientation is given via the r parameter
            return 1.0;
        }

        let hkl_real = lat.mat * hkl;
        let direction_real = lat.mat * self.hkl;

        let num = hkl_real.dot(&direction_real);
        let denom = direction_real.norm() * hkl_real.norm();

        let alpha_rad = if ((num / denom).abs() - 1.0).abs() < HKL_NORM_TOL {
            0.0
        } else {
            (num / denom).acos()
        };

        march(alpha_rad, self.r)
    }
}

impl<'de> Deserialize<'de> for MarchDollase {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Hkl,
            R,
        }

        struct MarchDollaseVisitor;
        impl<'de> Visitor<'de> for MarchDollaseVisitor {
            type Value = MarchDollase;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct MarchDollase")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut hkl = None;
                let mut r = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::R => {
                            if r.is_some() {
                                return Err(de::Error::duplicate_field("r"));
                            }
                            r = Some(map.next_value()?);
                        }
                        Field::Hkl => {
                            if hkl.is_some() {
                                return Err(de::Error::duplicate_field("hkl"));
                            }
                            hkl = Some(map.next_value()?);
                        }
                    }
                }

                let hkl = hkl.ok_or_else(|| de::Error::missing_field("hkl"))?;
                let r = r.ok_or_else(|| de::Error::missing_field("r"))?;
                if r < 0.0 {
                    return Err(de::Error::invalid_value(
                        de::Unexpected::Float(r),
                        &"r needs to be larger than 0.",
                    ));
                }

                let ret = MarchDollase::new(hkl, r).map_err(|err| {
                    de::Error::invalid_value(de::Unexpected::Float(r), &err.as_str())
                })?;

                Ok(ret)
            }
        }
        const FIELDS: &[&str] = &["hkl", "r"];
        deserializer.deserialize_struct("MarchDollase", FIELDS, MarchDollaseVisitor)
    }
}
