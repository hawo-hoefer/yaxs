use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, PartialEq, Clone, Copy)]
#[serde(transparent)]
pub struct VolumeFraction(pub f64);

impl VolumeFraction {
    pub fn new(p: f64) -> Option<Self> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }

        Some(Self(p))
    }
}

impl<'de> Deserialize<'de> for VolumeFraction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct VFVisitor;
        impl serde::de::Visitor<'_> for VFVisitor {
            type Value = VolumeFraction;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "struct VolumeFraction")
            }

            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let p = VolumeFraction::new(v);
                p.ok_or(serde::de::Error::invalid_value(
                    serde::de::Unexpected::Float(v),
                    &"value in the range [0, 1]",
                ))
            }
        }

        deserializer.deserialize_f64(VFVisitor)
    }
}
