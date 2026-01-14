use num_traits::Float;
use serde::{Deserialize, Deserializer};

use crate::cfg::Parameter;

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

pub fn deserialize_angle_rad_to_deg<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let angle = f64::deserialize(deserializer)?;
    Ok(angle.to_radians())
}
