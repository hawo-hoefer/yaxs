use log::error;
use rand::distr::uniform::SampleUniform;
use rand::distr::Uniform;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Parameter<T>
where
    T: PartialOrd + std::fmt::Debug,
{
    Fixed(T),
    Range(T, T),
    // Choice(Vec<T>),
    // ChoiceWithWeights(Vec<T>, Vec<f32>)
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
            Inner::Fixed(v) => return Ok(Parameter::Fixed(v)),
            Inner::Range(lo, hi) => {
                if lo > hi {
                    return Err(serde::de::Error::custom(&format!( "lower bound needs to be smaller than or equal to upper bound. got {lo:?} > {hi:?}" )));
                }

                Ok(Parameter::Range(lo, hi))
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
            Parameter::Range(lo, hi) => Ok(Uniform::try_from(*lo..=*hi).unwrap_or_else(|err| {
                error!("Could not sample mean domain size: {err}");
                std::process::exit(1);
            })),
        }
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

        let err = serde_yaml::from_str::<DummyCfg>(
            "data: 1.2
coefs: [0.1, [1.2, 0.8], 0.8]
",
        )
        .expect_err("invalid range");
        println!("{err}");
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

        let err = serde_yaml::from_str::<DummyCfg>("[[1.2, 0.8], 1.2]").expect_err("invalid range");
        println!("{err}");
        panic!()
    }
}
