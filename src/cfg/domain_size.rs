use crate::math::linalg::{Mat3, Vec3, Vec4};
use crate::math::quaternion::Quaternion;
use crate::util::{
    deserialize_positive_parameter_list_3, deserialize_symmetric_3x3_to_aniso_ellipse,
    deserialize_unit_quaternion,
};
use log::error;
use rand::Rng;
use serde::Deserialize;

use super::Parameter;

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum EllipsoidalInner {
    #[serde(deserialize_with = "deserialize_symmetric_3x3_to_aniso_ellipse")]
    Exact {
        q_ori: Quaternion,
        orientation: Mat3<f64>,
        strength: Vec3<f64>,
    },
    /// for a specific orientation, roll axis lengths of the anisotropy ellipsoid
    /// orientation needs to be a unit quaternion
    OrientationAndStrength {
        #[serde(deserialize_with = "deserialize_unit_quaternion")]
        orientation: Quaternion,
        strength: [Parameter<f64>; 3],
    },
    /// roll a random orientation and strength in the parameter range for each sample
    /// this effectively creates an ellipsoid with random orientation and main axis
    /// lengths specified in the strength parameter
    #[serde(deserialize_with = "deserialize_positive_parameter_list_3")]
    StrengthRandomOri([Parameter<f64>; 3]),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum DomainSize {
    Uniform(Parameter<f64>),
    Ellipsoidal(EllipsoidalInner),
}

impl DomainSize {
    pub fn generate(&self, rng: &mut impl Rng) -> crate::domain_size::DomainSize {
        use crate::domain_size::DomainSize as DS;
        let ret = match self {
            DomainSize::Uniform(v) => DS::Isotropic(v.generate(rng)),
            DomainSize::Ellipsoidal(inner) => match inner {
                EllipsoidalInner::Exact {
                    orientation,
                    strength,
                    q_ori,
                } => DS::Ellipsoidal {
                    orientation: orientation.clone(),
                    main_sizes: strength.clone(),
                    q_ori: q_ori.clone(),
                },
                EllipsoidalInner::OrientationAndStrength {
                    orientation: q_ori,
                    strength,
                } => {
                    let orientation = q_ori.to_rotation_matrix().map(|x| *x as f64);
                    DS::Ellipsoidal {
                        orientation,
                        q_ori: q_ori.clone(),
                        main_sizes: Vec3::from_col([
                            strength[0].generate(rng),
                            strength[1].generate(rng),
                            strength[2].generate(rng),
                        ]),
                    }
                }
                EllipsoidalInner::StrengthRandomOri([a, b, c]) => {
                    let ori = crate::math::stats::sample_sphere_unif::<4>(rng);
                    let q_ori =
                        Quaternion::new(ori[0] as f32, ori[1] as f32, ori[2] as f32, ori[3] as f32);
                    let orientation = q_ori.to_rotation_matrix().map(|x| *x as f64);
                    let strength = Vec3::new(a.generate(rng), b.generate(rng), c.generate(rng));
                    DS::Ellipsoidal {
                        orientation,
                        main_sizes: strength,
                        q_ori,
                    }
                }
            },
        };

        ret
    }

    pub fn mean(&self) -> f64 {
        match self {
            DomainSize::Uniform(v) => v.mean(),
            _ => unimplemented!("no mean domain size"),
        }
    }

    pub fn upper_bound(&self) -> f64 {
        use EllipsoidalInner::*;
        match self {
            DomainSize::Uniform(v) => v.upper_bound(),
            DomainSize::Ellipsoidal(inner) => match inner {
                Exact { strength, .. } => *strength
                    .iter_values()
                    .max_by(|&a, &b| a.partial_cmp(b).expect("not nan"))
                    .expect("three values in iterator"),
                OrientationAndStrength { strength, .. } | StrengthRandomOri(strength) => strength
                    .iter()
                    .map(|x| x.upper_bound())
                    .max_by(|a, b| a.partial_cmp(b).expect("not nan"))
                    .expect("three values in iterator"),
            },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn parse_test<T: for<'a> Deserialize<'a> + PartialEq + std::fmt::Debug>(
        input: &str,
        target: T,
    ) -> Result<(), (T, T)> {
        let v: T = serde_yaml::from_str(input).expect("valid input");

        if v == target {
            Ok(())
        } else {
            Err((v, target))
        }
    }

    #[test]
    fn parse_uniform() {
        parse_test("100", DomainSize::Uniform(Parameter::Fixed(100.0))).unwrap();
        parse_test(
            "[100, 1000]",
            DomainSize::Uniform(Parameter::Range(100.0, 1000.0)),
        )
        .unwrap();
    }

    #[test]
    fn parse_ellipsoidal_exact() {
        let m = Mat3::sym_from_tri_lo(100.0, 100.0, 300.0, 100.0, 200.0, 10.0);
        let (evals, evecs) = m.symmetric_eigendecomp();
        let q_ori = Quaternion::from_rotation_matrix(&evecs);

        parse_test(
            "[100, 100, 300, 100, 200, 10]",
            DomainSize::Ellipsoidal(EllipsoidalInner::Exact {
                q_ori: q_ori.clone(),
                orientation: evecs.clone(),
                strength: Vec3::from_col(evals),
            }),
        )
        .unwrap();

        parse_test(
            "{ s11: 100, s22: 100, s33: 300, s21: 100, s31: 200, s32: 10 }",
            DomainSize::Ellipsoidal(EllipsoidalInner::Exact {
                q_ori,
                orientation: evecs,
                strength: Vec3::from_col(evals),
            }),
        )
        .unwrap();
    }

    #[test]
    fn parse_ellipsoidal_uniform_direction() {
        parse_test(
            "[100, [100, 200], 300]",
            DomainSize::Ellipsoidal(EllipsoidalInner::StrengthRandomOri([
                Parameter::Fixed(100.0),
                Parameter::Range(100.0, 200.0),
                Parameter::Fixed(300.0),
            ])),
        )
        .unwrap();
    }

    #[test]
    fn parse_ellipsoidal_selected_direction() {
        parse_test(
            "{orientation: [0, 1, 0, 0], strength: [100, [100, 200], 300]}",
            DomainSize::Ellipsoidal(EllipsoidalInner::OrientationAndStrength {
                orientation: Quaternion::new(0.0, 1.0, 0.0, 0.0),
                strength: [
                    Parameter::Fixed(100.0),
                    Parameter::Range(100.0, 200.0),
                    Parameter::Fixed(300.0),
                ],
            }),
        )
        .unwrap();
    }
}
