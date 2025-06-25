use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Clone, Copy)]
pub struct Probability(f64);

impl Probability {
    pub fn new(p: f64) -> Option<Self> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }

        Some(Self(p))
    }
}

impl From<Probability> for f64 {
    fn from(val: Probability) -> Self {
        val.0
    }
}

impl<'de> Deserialize<'de> for Probability {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ProbVisitor;
        impl serde::de::Visitor<'_> for ProbVisitor {
            type Value = Probability;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Probability")
            }

            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let p = Probability::new(v);
                p.ok_or(serde::de::Error::invalid_value(
                    serde::de::Unexpected::Float(v),
                    &"value in the range [0, 1]",
                ))
            }
        }

        deserializer.deserialize_f64(ProbVisitor)
    }
}
