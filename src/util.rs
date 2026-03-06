use std::ops::{AddAssign, Sub};

use num_traits::{ConstOne, Float, FloatConst};
use serde::{Deserialize, Deserializer};

use crate::cfg::Parameter;
use crate::math::linalg::{Mat3, Vec3, Vec4};
use crate::math::quaternion::Quaternion;

pub fn deserialize_nonzero_float<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Float + Deserialize<'de>,
{
    let val = T::deserialize(deserializer)?;
    if val.is_zero() {
        return Err(serde::de::Error::invalid_value(
            serde::de::Unexpected::Other("0.0"),
            &"a nonzero number",
        ));
    }

    Ok(val)
}

macro_rules! dppl {
    ($n:literal, $name:ident) => {
        pub fn $name<'de, T, D>(deserializer: D) -> Result<[Parameter<T>; $n], D::Error>
        where
            D: Deserializer<'de>,
            T: Deserialize<'de> + std::fmt::Debug + num_traits::Zero + PartialOrd,
        {
            let val = <[Parameter<T>; _]>::deserialize(deserializer)?;

            for v in val.iter() {
                match v {
                    Parameter::Fixed(ref v) => {
                        if v < &T::zero() {
                            return Err(serde::de::Error::invalid_value(
                                serde::de::Unexpected::Other("fixed zero parameter"),
                                &"a positive value",
                            ));
                        }
                    }
                    Parameter::Range(ref lo, _) => {
                        if lo < &T::zero() {
                            return Err(serde::de::Error::invalid_value(
                                serde::de::Unexpected::Other("lower parameter range"),
                                &"a positive value",
                            ));
                        }
                    }
                }
            }

            Ok(val)
        }
    };
}

dppl!(3, deserialize_positive_parameter_list_3);
dppl!(6, deserialize_positive_parameter_list_6);

pub fn deserialize_positive_parameter<'de, T, D>(deserializer: D) -> Result<Parameter<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + std::fmt::Debug + num_traits::Zero + PartialOrd,
{
    let val = <Parameter<T>>::deserialize(deserializer)?;

    match val {
        Parameter::Fixed(ref v) => {
            if v < &T::zero() {
                return Err(serde::de::Error::invalid_value(
                    serde::de::Unexpected::Other("fixed zero parameter"),
                    &"a positive value",
                ));
            }
        }
        Parameter::Range(ref lo, _) => {
            if lo < &T::zero() {
                return Err(serde::de::Error::invalid_value(
                    serde::de::Unexpected::Other("lower parameter range"),
                    &"a positive value",
                ));
            }
        }
    }

    Ok(val)
}

pub fn deserialize_nonzero_usize<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    let val = usize::deserialize(deserializer)?;
    if val == 0 {
        return Err(serde::de::Error::invalid_value(
            serde::de::Unexpected::Other("0"),
            &"a positive, nonzero integer",
        ));
    }

    Ok(val)
}

pub fn deserialize_range<'de, D>(deserializer: D) -> Result<(f64, f64), D::Error>
where
    D: Deserializer<'de>,
{
    let (lo, hi) = <(f64, f64)>::deserialize(deserializer)?;

    if lo == hi {
        return Err(serde::de::Error::invalid_value(
            serde::de::Unexpected::Other("empty range"),
            &"(lo, hi) with lo < hi",
        ));
    }

    if lo > hi {
        return Err(serde::de::Error::invalid_value(
            serde::de::Unexpected::Other("lower limit larger than uper limit"),
            &"(lo, hi) with lo < hi",
        ));
    }

    Ok((lo, hi))
}

pub fn deserialize_unit_quaternion<'de, D>(deserializer: D) -> Result<Quaternion, D::Error>
where
    D: Deserializer<'de>,
{
    const ATOL: f32 = 1e-6;

    let v = Vec4::<f32>::deserialize(deserializer)?;
    if !(v.magnitude() - 1.0 < ATOL) {
        return Err(serde::de::Error::invalid_value(
            serde::de::Unexpected::Seq,
            &"a unit quaternion (magnitude must be 1)",
        ));
    }

    Ok(Quaternion {
        w: v[0],
        x: v[1],
        y: v[2],
        z: v[3],
    })
}

pub fn deserialize_symmetric_3x3_to_aniso_ellipse<'de, D>(
    deserializer: D,
) -> Result<(Quaternion, Mat3<f64>, Vec3<f64>), D::Error>
where
    D: Deserializer<'de>,
{
    use serde::Deserialize;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SymmetricMat3 {
        List([f64; 6]),
        Named {
            s11: f64,
            s22: f64,
            s33: f64,
            s21: f64,
            s31: f64,
            s32: f64,
        },
    }

    let mat = SymmetricMat3::deserialize(deserializer)?;
    match mat {
        SymmetricMat3::List([s11, s22, s33, s21, s31, s32])
        | SymmetricMat3::Named {
            s11,
            s22,
            s33,
            s21,
            s31,
            s32,
        } => {
            let mat = Mat3::from_rows([[s11, s21, s31], [s21, s22, s32], [s31, s32, s33]]);
            let (evals, evecs) = mat.symmetric_eigendecomp();

            let q = Quaternion::from_rotation_matrix(&evecs);

            return Ok((q, evecs, Vec3::from_col(evals)));
        }
    }
}

pub fn deserialize_angle_rad_to_deg<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let angle = f64::deserialize(deserializer)?;
    Ok(angle.to_radians())
}

pub fn deserialize_positive_float<'de, D, T: Float>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let v = T::deserialize(deserializer)?;
    if v.is_sign_negative() {
        return Err(serde::de::Error::invalid_value(
            serde::de::Unexpected::Float(v.to_f64().expect("not nan")),
            &"a non-negative number",
        ));
    }

    return Ok(v);
}
