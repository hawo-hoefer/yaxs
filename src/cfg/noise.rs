use super::Parameter;
use crate::noise::Noise;
use rand::Rng;
use serde::de::Visitor;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Clone)]
pub enum NoiseSpec {
    Gaussian {
        sigma: Parameter<f64>,
    },
    Uniform {
        min: Parameter<f64>,
        max: Parameter<f64>,
    },
}

impl NoiseSpec {
    pub fn generate(&self, rng: &mut impl Rng) -> Noise {
        match self {
            NoiseSpec::Gaussian { sigma } => Noise::Gaussian {
                sigma: sigma.generate(rng),
            },
            NoiseSpec::Uniform { min, max } => Noise::Uniform {
                min: min.generate(rng),
                max: max.generate(rng),
            },
        }
    }
}

impl<'de> Deserialize<'de> for NoiseSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Sigma,
            Min,
            Max,
        }

        struct NoiseVisitor;
        impl<'de> Visitor<'de> for NoiseVisitor {
            type Value = NoiseSpec;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Noise")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                use serde::de;

                let mut sigma: Option<Parameter<f64>> = None;
                let mut min: Option<Parameter<f64>> = None;
                let mut max: Option<Parameter<f64>> = None;

                fn inv_field_msg(name: &'static str) -> String {
                    format!("invalid field '{}' expected either 'sigma' for gaussian distribution, or 'min' and 'max' for uniform distribution", name)
                }

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Sigma => {
                            if sigma.is_some() {
                                return Err(de::Error::duplicate_field("sigma"));
                            }

                            if min.is_some() || max.is_some() {
                                return Err(de::Error::custom(inv_field_msg("sigma")));
                            }

                            let v: Parameter<f64> = map.next_value()?;
                            match v {
                                Parameter::Fixed(v) if v <= 0.0 => {
                                    return Err(de::Error::invalid_value(
                                        de::Unexpected::Float(v),
                                        &"a value larger than 0.0",
                                    ));
                                }
                                Parameter::Range(lo, _) if lo <= 0.0 => {
                                    return Err(de::Error::invalid_value(
                                        de::Unexpected::Float(lo),
                                        &"lower bound of gaussian standard deviation range needs to be larger than 0.0",
                                    ));
                                }
                                _ => (),
                            }
                            sigma = Some(v);
                        }
                        Field::Min => {
                            if min.is_some() {
                                return Err(de::Error::duplicate_field(""));
                            }

                            if sigma.is_some() {
                                return Err(de::Error::custom(inv_field_msg("min")));
                            }

                            let maybe_min: Parameter<f64> = map.next_value()?;
                            if let Some(max) = max {
                                if max.lower_bound() < maybe_min.upper_bound() {
                                    return Err(de::Error::invalid_value(
                                        de::Unexpected::Float(maybe_min.upper_bound()),
                                        &"a value smaller than max (or it's lower bound)",
                                    ));
                                }
                            }

                            min = Some(maybe_min);
                        }
                        Field::Max => {
                            if max.is_some() {
                                return Err(de::Error::duplicate_field(""));
                            }

                            if sigma.is_some() {
                                return Err(de::Error::custom(inv_field_msg("max")));
                            }

                            let maybe_max: Parameter<f64> = map.next_value()?;
                            if let Some(min) = min {
                                if maybe_max.lower_bound() < min.upper_bound() {
                                    return Err(de::Error::invalid_value(
                                        de::Unexpected::Float(maybe_max.lower_bound()),
                                        &"a value larger than min (or it's upper bound)",
                                    ));
                                }
                            }
                            max = Some(maybe_max);
                        }
                    }
                }

                if sigma.is_some() {
                    let sigma = sigma.ok_or_else(|| de::Error::missing_field("sigma"))?;

                    return Ok(NoiseSpec::Gaussian { sigma });
                }

                if min.is_some() || max.is_some() {
                    let min: Parameter<f64> = min.ok_or_else(|| de::Error::missing_field("min"))?;
                    let max: Parameter<f64> = max.ok_or_else(|| de::Error::missing_field("max"))?;

                    return Ok(NoiseSpec::Uniform { min, max });
                }

                return Err(de::Error::custom(
                    "Could not parse noise from no information",
                ));
            }
        }

        const FIELDS: &[&str] = &["sigma", "scale", "min", "max"];
        deserializer.deserialize_struct("Noise", FIELDS, NoiseVisitor)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use Parameter::*;

    #[test]
    fn deserialize_noise_uniform_fixed() {
        let noise: NoiseSpec = serde_yaml::from_str("min: 0.0\nmax: 1.0").expect("valid noise");
        assert!(matches!(
            noise,
            NoiseSpec::Uniform {
                min: Fixed(0.0),
                max: Fixed(1.0)
            }
        ))
    }

    #[test]
    fn deserialize_noise_uniform_ranges() {
        let noise: NoiseSpec =
            serde_yaml::from_str("min: [0.0, 0.1]\nmax: [0.11, 0.9]").expect("valid noise");
        assert!(matches!(
            noise,
            NoiseSpec::Uniform {
                min: Range(0.0, 0.1),
                max: Range(0.11, 0.9)
            }
        ))
    }

    #[test]
    fn deserialize_noise_uniform_mixed() {
        let noise: NoiseSpec =
            serde_yaml::from_str("min: 0.1\nmax: [0.11, 0.9]").expect("valid noise");
        assert!(matches!(
            noise,
            NoiseSpec::Uniform {
                min: Fixed(0.1),
                max: Range(0.11, 0.9)
            }
        ))
    }

    #[test]
    fn deserialize_noise_gaussian_fixed() {
        let noise: NoiseSpec = serde_yaml::from_str("sigma: 1.0").expect("valid noise");
        assert!(matches!(noise, NoiseSpec::Gaussian { sigma: Fixed(1.0) }))
    }

    #[test]
    fn deserialize_noise_uniform_range_err_fixed() {
        let _err = serde_yaml::from_str::<NoiseSpec>("min: 2.0\nmax: 1.0")
            .expect_err("invalid min/max range");
    }

    #[test]
    fn deserialize_noise_uniform_range_err_min_range() {
        let _err = serde_yaml::from_str::<NoiseSpec>("min: [0.0, 1.1]\nmax: 1.0")
            .expect_err("invalid noise");
    }

    #[test]
    fn deserialize_noise_uniform_range_err_max_range() {
        let _err = serde_yaml::from_str::<NoiseSpec>("max: [0.0, 1.1]\nmin: 0.1")
            .expect_err("invalid noise");
    }

    #[test]
    fn deserialize_noise_uniform_range_err_both_range() {
        let _err = serde_yaml::from_str::<NoiseSpec>("min: [0.0, 1.1]\nmax: [1.0, 2.0]")
            .expect_err("invalid noise");
    }

    #[test]
    fn deserialize_noise_gaussian_ranges() {
        let noise: NoiseSpec = serde_yaml::from_str("sigma: [1.0, 2.0]\n").expect("valid noise");
        assert!(matches!(
            noise,
            NoiseSpec::Gaussian {
                sigma: Range(1.0, 2.0),
            }
        ))
    }

    #[test]
    fn deserialize_noise_mixed_kinds_sigma_max() {
        let _err = serde_yaml::from_str::<NoiseSpec>("sigma: [1.0, 2.0]\nmax: 5.0")
            .expect_err("invalid noise");
    }

    #[test]
    fn deserialize_noise_mixed_kinds_sigma_min() {
        let _err = serde_yaml::from_str::<NoiseSpec>("sigma: [1.0, 2.0]\nmin: 5.0")
            .expect_err("invalid noise");
    }

    #[test]
    fn deserialize_noise_mixed_kinds_scale_max() {
        let _err = serde_yaml::from_str::<NoiseSpec>("scale: [1.0, 2.0]\nmax: 5.0")
            .expect_err("invalid noise");
    }

    #[test]
    fn deserialize_noise_mixed_kinds_scale_min() {
        let _err = serde_yaml::from_str::<NoiseSpec>("scale: [1.0, 2.0]\nmin: 5.0")
            .expect_err("invalid noise");
    }

    #[test]
    fn deserialize_noise_mixed_kinds_min_scale() {
        let _err = serde_yaml::from_str::<NoiseSpec>("min: [1.0, 2.0]\nscale: 5.0")
            .expect_err("invalid noise");
    }

    #[test]
    fn deserialize_noise_mixed_kinds_max_scale() {
        let _err = serde_yaml::from_str::<NoiseSpec>("max: [1.0, 2.0]\nscale: 5.0")
            .expect_err("invalid noise");
    }

    #[test]
    fn deserialize_noise_mixed_kinds_min_sigma() {
        let _err = serde_yaml::from_str::<NoiseSpec>("min: [1.0, 2.0]\nsigma: 5.0")
            .expect_err("invalid noise");
    }

    #[test]
    fn deserialize_noise_mixed_kinds_max_sigma() {
        let _err = serde_yaml::from_str::<NoiseSpec>("max: [1.0, 2.0]\nsigma: 5.0")
            .expect_err("invalid noise");
    }
}
