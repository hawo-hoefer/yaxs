use crate::math::linalg::Mat3;

#[derive(Debug, PartialEq, Clone)]
pub struct Strain(pub [f64; 6]);

impl std::fmt::Display for Strain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Strain(")?;
        for (i, a) in self.0.iter().enumerate() {
            if i != self.0.len() - 1 {
                write!(f, "{}, ", a)?;
            } else {
                write!(f, "{}", a)?;
            }
        }
        write!(f, ")")
    }
}

impl Strain {
    pub fn from_diag(a: f64, b: f64, c: f64) -> Self {
        Self([a, 0.0, b, 0.0, 0.0, c])
    }

    pub fn new_verified(data: [f64; 6]) -> Option<Self> {
        // TODO: find a better way to do this. we may not actually need to try calculating the inverse

        // strain is ok if we can take the inverse of the strain matrix
        // use this to verify user input
        let v = Self(data);
        if v.to_mat3().try_inverse().is_some() {
            return Some(v);
        }
        None
    }

    pub fn from_mat3(mat: &Mat3<f64>) -> Self {
        Self([
            mat[(0, 0)],
            mat[(1, 0)],
            mat[(1, 1)],
            mat[(2, 0)],
            mat[(2, 1)],
            mat[(2, 2)],
        ])
    }

    pub fn none() -> Self {
        Self([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    }

    pub fn to_mat3(&self) -> Mat3<f64> {
        let mut ret = Mat3::zeros();
        ret[(0, 0)] = self.0[0];

        ret[(1, 0)] = self.0[1];
        ret[(0, 1)] = self.0[1];

        ret[(1, 1)] = self.0[2];

        ret[(2, 0)] = self.0[3];
        ret[(0, 2)] = self.0[3];

        ret[(2, 1)] = self.0[4];
        ret[(1, 2)] = self.0[4];

        ret[(2, 2)] = self.0[5];

        ret
    }
}


