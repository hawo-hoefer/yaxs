use crate::math::linalg::{ColVec, Mat3, Vec3, Vec4};
use crate::math::quaternion::Quaternion;
use crate::util::{
    deserialize_positive_parameter, deserialize_positive_parameter_list_3,
    deserialize_symmetric_3x3_to_aniso_ellipse, deserialize_unit_quaternion,
};
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
        main_sizes: Vec3<f64>,
    },
    /// for a specific orientation, roll axis lengths of the anisotropy ellipsoid
    /// orientation needs to be a unit quaternion
    OrientationAndSizes {
        #[serde(deserialize_with = "deserialize_unit_quaternion")]
        orientation: Quaternion,
        main_sizes: [Parameter<f64>; 3],
    },
    /// roll a random orientation and axis lengths in the parameter range for each sample
    /// this effectively creates an ellipsoid with random orientation and main axis
    /// lengths specified in the strength parameter
    #[serde(deserialize_with = "deserialize_positive_parameter_list_3")]
    SizesRandomOri([Parameter<f64>; 3]),
    /// roll a random orientation and axis length in the parameter range for each sample
    /// this effectively creates an ellipsoid with random orientation and main axis
    /// lengths specified in the strength parameter
    SingleSizeRandomOri {
        #[serde(deserialize_with = "deserialize_positive_parameter")]
        ani: Parameter<f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum DomainSize {
    Uniform(Parameter<f64>),
    Ellipsoidal(EllipsoidalInner),
}

fn ellipsoidal(q_ori: Quaternion, main_sizes: Vec3<f64>) -> crate::domain_size::DomainSize {
    let orientation = q_ori.to_rotation_matrix().map(|x| *x as f64);
    let evals = main_sizes.map(cs_to_eval);
    let spd_mat = orientation
        .transpose()
        .matmul_diag(&evals)
        .matmul(&orientation);

    crate::domain_size::DomainSize::Ellipsoidal {
        orientation,
        q_ori: q_ori.clone(),
        evals,
        main_sizes,
        mat: ColVec::from_col([
            spd_mat[(0, 0)],
            spd_mat[(1, 0)],
            spd_mat[(1, 1)],
            spd_mat[(2, 0)],
            spd_mat[(2, 1)],
            spd_mat[(2, 2)],
        ]),
    }
}

fn cs_to_eval(x: &f64) -> f64 {
    x.powi(2).recip()
}

impl DomainSize {
    pub fn generate(&self, rng: &mut impl Rng) -> crate::domain_size::DomainSize {
        use crate::domain_size::DomainSize as DS;
        let ret = match self {
            DomainSize::Uniform(v) => DS::Isotropic(v.generate(rng)),
            DomainSize::Ellipsoidal(inner) => match inner {
                EllipsoidalInner::Exact {
                    orientation,
                    main_sizes,
                    q_ori,
                } => ellipsoidal(q_ori.clone(), main_sizes.clone()),
                EllipsoidalInner::OrientationAndSizes {
                    orientation: q_ori,
                    main_sizes: strength,
                } => {
                    let main_sizes = Vec3::new(
                        strength[0].generate(rng),
                        strength[1].generate(rng),
                        strength[2].generate(rng),
                    );
                    ellipsoidal(q_ori.clone(), main_sizes)
                }
                EllipsoidalInner::SizesRandomOri([a, b, c]) => {
                    let q_ori = Quaternion::from(crate::math::stats::sample_sphere_unif::<4>(rng));
                    let main_sizes = Vec3::new(a.generate(rng), b.generate(rng), c.generate(rng));
                    ellipsoidal(q_ori, main_sizes)
                }
                EllipsoidalInner::SingleSizeRandomOri { ani: a } => {
                    let q_ori = Quaternion::from(crate::math::stats::sample_sphere_unif::<4>(rng));
                    let main_sizes = Vec3::new(a.generate(rng), a.generate(rng), a.generate(rng));
                    ellipsoidal(q_ori, main_sizes)
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
                Exact {
                    main_sizes: strength,
                    ..
                } => *strength
                    .iter_values()
                    .max_by(|&a, &b| a.partial_cmp(b).expect("not nan"))
                    .expect("three values in iterator"),
                OrientationAndSizes {
                    main_sizes: strength,
                    ..
                }
                | SizesRandomOri(strength) => strength
                    .iter()
                    .map(|x| x.upper_bound())
                    .max_by(|a, b| a.partial_cmp(b).expect("not nan"))
                    .expect("three values in iterator"),
                SingleSizeRandomOri { ani } => ani.upper_bound(),
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
                main_sizes: Vec3::from_col(evals),
            }),
        )
        .unwrap();

        parse_test(
            "{ s11: 100, s22: 100, s33: 300, s21: 100, s31: 200, s32: 10 }",
            DomainSize::Ellipsoidal(EllipsoidalInner::Exact {
                q_ori,
                orientation: evecs,
                main_sizes: Vec3::from_col(evals),
            }),
        )
        .unwrap();
    }

    #[test]
    fn parse_ellipsoidal_uniform_direction() {
        parse_test(
            "[100, [100, 200], 300]",
            DomainSize::Ellipsoidal(EllipsoidalInner::SizesRandomOri([
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
            "{orientation: [0, 1, 0, 0], main_sizes: [100, [100, 200], 300]}",
            DomainSize::Ellipsoidal(EllipsoidalInner::OrientationAndSizes {
                orientation: Quaternion::new(0.0, 1.0, 0.0, 0.0),
                main_sizes: [
                    Parameter::Fixed(100.0),
                    Parameter::Range(100.0, 200.0),
                    Parameter::Fixed(300.0),
                ],
            }),
        )
        .unwrap();
    }

    #[test]
    fn parse_ellipsoidal_equal_all_dirs() {
        parse_test(
            "{ani: [10, 100]}",
            DomainSize::Ellipsoidal(EllipsoidalInner::SingleSizeRandomOri {
                ani: Parameter::Range(10.0, 100.0),
            }),
        )
        .unwrap();
    }
}
