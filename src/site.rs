use crate::math::linalg::Vec3;

use crate::species::Species;

#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub coords: Vec3<f64>,
    pub species: Species,
    pub occu: f64,
}

impl Site {
    pub fn normalized(&self) -> Site {
        todo!()
        // let coords = self.coords.iter().map(|mut x| {
        //     x = x - x.round();
        //     if x < 0.0 {
        //         // map negative positions to positive end of unit cell
        //         x = 1.0 + x
        //     }
        //     return x;
        // });
        // Self {
        //     coords,
        //     species: self.species.clone(),
        //     occu: self.occu,
        // }
    }
}
