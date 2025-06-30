use crate::math::Vec3;

use crate::species::Species;

#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub coords: Vec3<f64>,
    pub species: Species,
    pub occu: f64,
}

impl Site {
    pub fn normalized(&self) -> Site {
        let coords = self.coords.map(|x| {
            let mut x = x - x.round();
            if x < 0.0 {
                // map negative positions to positive end of unit cell
                x += 1.0
            }
            x
        });
        Self {
            coords,
            species: self.species.clone(),
            occu: self.occu,
        }
    }
}

#[cfg(test)]
mod test {
    use std::str::FromStr;

    use super::*;

    #[test]
    fn normalization() {
        let s = Site {
            coords: Vec3::new(1.52, 0.2, -1.2),
            species: Species::from_str("Fe+").unwrap(),
            occu: 1.0,
        };
        let s2 = s.normalized();
        assert_eq!(s2.coords, Vec3::new(0.52, 0.2, 0.8))
    }
}
