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
