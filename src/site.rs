use nalgebra::Vector3;

use crate::species::Species;

#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub coords: Vector3<f64>,
    pub species: Species,
    pub occu: f64,
}

impl Site {
    pub fn normalized(&self) -> Site {
        let coords = self.coords.map(|mut x| {
            x = x - x.round();
            if x < 0.0 {
                // map negative positions to positive end of unit cell
                x = 1.0 + x
            }
            return x;
        });
        Self {
            coords,
            species: self.species.clone(),
            occu: self.occu,
        }
    }
}
