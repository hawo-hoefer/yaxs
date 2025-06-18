use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Clone, Copy)]
pub struct Probability(f64);

impl Probability {
    pub fn new(p: f64) -> Option<Self> {
        if p < 0.0 || p > 1.0 {
            return None;
        }

        Some(Self(p))
    }
}

impl Into<f64> for Probability {
    fn into(self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Probability {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ProbVisitor;
        impl<'de> serde::de::Visitor<'de> for ProbVisitor {
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
