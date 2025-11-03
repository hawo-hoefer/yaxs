use crate::math::linalg::Vec3;

use crate::species::Species;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AtomicDisplacement {
    // https://www.iucr.org/resources/commissions/crystallographic-nomenclature/adp
    Uiso(f64),
    Biso(f64),
    // Uani([f64; 6]),
    // Uovl,
    // Umpe,
    // Bani,
    // Bovl,
}

impl AtomicDisplacement {
    pub fn debye_waller_factor(&self, g_hkl: f64) -> f64 {
        match self {
            AtomicDisplacement::Uiso(u) => {
                // original formula from link above is
                // T(|h|) = exp(- 8 pi^2 * <u^2> (sin^2 theta / lambda^2) )
                // in bragg condition, n \lambda = 2 d sin theta
                // and using n = 1, we get
                // sin theta / lambda = 1 / (2 d).
                // therefore, T(|h|) = exp(-8 pi^2 * <u^2> / 4d^2)
                use std::f64::consts::PI;
                let x = (-8.0 * PI * PI * u / 4.0 * g_hkl.powi(2)).exp();
                x
            }
            AtomicDisplacement::Biso(b) => {
                // the same as Uiso, only that b = u / (8 pi^2)
                let x = (-b / 4.0 * g_hkl.powi(2)).exp();
                x
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub coords: Vec3<f64>,
    pub species: Species,
    pub occu: f64,
    pub displacement: Option<AtomicDisplacement>,
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
            displacement: self.displacement,
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
            displacement: None,
        };
        let s2 = s.normalized();
        assert_eq!(s2.coords, Vec3::new(0.52, 0.2, 0.8))
    }
}
