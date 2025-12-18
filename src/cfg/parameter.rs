use num_traits::Float;
use rand::distr::uniform::SampleUniform;
use rand::distr::Uniform;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Clone, Copy, PartialEq)]
#[serde(untagged)]
pub enum Parameter<T> {
    Fixed(T),
    Range(T, T),
    // Choice(Vec<T>),
    // ChoiceWithWeights(Vec<T>, Vec<f32>)
}

impl<T> Parameter<T> {
    pub fn range_checked(lo: T, hi: T) -> Result<Self, String>
    where
        T: std::fmt::Debug + PartialOrd,
    {
        if lo > hi {
            return Err(format!(
                "lower bound needs to be smaller than or equal to upper bound. got {lo:?} > {hi:?}"
            ));
        }

        Ok(Self::Range(lo, hi))
    }

    pub fn upper_bound(&self) -> T
    where
        T: Copy,
    {
        match self {
            Parameter::Fixed(v) => *v,
            Parameter::Range(_, v) => *v,
        }
    }

    pub fn lower_bound(&self) -> T
    where
        T: Copy,
    {
        match self {
            Parameter::Fixed(v) => *v,
            Parameter::Range(v, _) => *v,
        }
    }

    pub fn mean(&self) -> T
    where
        T: Float,
    {
        (self.upper_bound() + self.lower_bound()) / (T::one() + T::one())
    }
}

impl<'de, T> Deserialize<'de> for Parameter<T>
where
    T: PartialOrd + std::fmt::Debug + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Inner<T>
        where
            T: PartialOrd + std::fmt::Debug,
        {
            Fixed(T),
            Range(T, T),
        }

        let p = Inner::deserialize(deserializer)?;

        match p {
            Inner::Fixed(v) => Ok(Parameter::Fixed(v)),
            Inner::Range(lo, hi) => {
                Parameter::range_checked(lo, hi).map_err(|err| serde::de::Error::custom(err))
            }
        }
    }
}

impl<T> Parameter<T>
where
    T: SampleUniform + PartialOrd + Copy + std::fmt::Debug,
{
    pub fn generate(&self, rng: &mut impl Rng) -> T {
        match self {
            Parameter::Fixed(v) => *v,
            Parameter::Range(lo, hi) => rng.random_range(*lo..=*hi),
        }
    }

    pub fn sampler(&self) -> Result<Uniform<T>, T> {
        match self {
            Parameter::Fixed(v) => Err(*v),
            Parameter::Range(lo, hi) => Ok(Uniform::try_from(*lo..=*hi)
                .unwrap_or_else(|_| unreachable!("proper initialization should prevent this"))),
        }
    }
}

impl<T> Default for Parameter<T>
where
    T: Default,
{
    fn default() -> Self {
        Parameter::Fixed(T::default())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn deserialize_parameter_bounds_nok() {
        serde_yaml::from_str::<Parameter<f64>>("[1.2, 0.8]").expect_err("invalid range");
    }

    #[test]
    fn deserialize_parameter_list_bounds_nok() {
        let _ = serde_yaml::from_str::<Vec<Parameter<f64>>>("[1.0, [1.2, 0.8]]")
            .expect_err("invalid range");
    }

    #[test]
    fn deserialize_parameter_struct_bounds_nok() {
        #[derive(Deserialize, Debug)]
        #[allow(unused)]
        struct DummyCfg {
            data: f64,
            coefs: [Parameter<f64>; 3],
        }

        let _ = serde_yaml::from_str::<DummyCfg>(
            "data: 1.2
coefs: [0.1, [1.2, 0.8], 0.8]
",
        )
        .expect_err("invalid range");
    }

    #[test]
    fn deserialize_parameter_enum_bounds_nok() {
        #[derive(Deserialize, Debug)]
        #[serde(untagged)]
        #[allow(unused)]
        enum DummyCfg {
            V0(f64),
            V1([Parameter<f64>; 2]),
        }

        let _err =
            serde_yaml::from_str::<DummyCfg>("[[1.2, 0.8], 1.2]").expect_err("invalid range");
    }

    #[test]
    fn deserialize_parameter_enum_should_fail() {
        let _err =
            serde_yaml::from_str::<Parameter<f64>>("[-0.0, -0.25]").expect_err("invalid range");
    }
}
