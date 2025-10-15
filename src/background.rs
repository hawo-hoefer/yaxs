#[derive(Clone, PartialEq, Debug)]
pub enum Background {
    None,
    Polynomial { poly_coef: Vec<f32>, scale: f32 },
    Exponential { slope: f32, scale: f32 },
}

fn cheb2poly(chebyshev_coefs: &[f32]) -> Vec<f32> {
    let mut poly_coef = Vec::with_capacity(chebyshev_coefs.len());
    poly_coef.resize(chebyshev_coefs.len(), 0.0);

    poly_coef[0] = chebyshev_coefs[0];
    if chebyshev_coefs.len() == 1 {
        return poly_coef;
    }
    poly_coef[1] = chebyshev_coefs[1];

    let mut cn2 = Vec::with_capacity(chebyshev_coefs.len());
    cn2.resize(chebyshev_coefs.len(), 0.0f32);
    cn2[0] = 1.0;

    let mut cn1 = Vec::with_capacity(chebyshev_coefs.len());
    cn1.resize(chebyshev_coefs.len(), 0.0f32);
    cn1[1] = 1.0;

    let mut cn = Vec::with_capacity(chebyshev_coefs.len());
    cn.resize(chebyshev_coefs.len(), 0.0f32);
    #[allow(clippy::needless_range_loop)]
    for i in 2..chebyshev_coefs.len() {
        // cn[:] = 0
        cn.iter_mut().for_each(|coef| *coef = 0.0);

        // cn[1 : i + 1] = 2 * cn1[:i]
        for j in 1..(i + 1) {
            cn[j] = 2.0 * cn1[j - 1];
        }

        // cn[:i] -= cn2[:i]
        for j in 0..i {
            cn[j] -= cn2[j];
        }

        // out += (coefs[:, i, None] * cn).astype(coefs.dtype)  # type: ignore
        poly_coef.iter_mut().enumerate().for_each(|(j, coef)| {
            *coef += chebyshev_coefs[i] * cn[j];
        });
        // swap cns
        // cn, cn1, cn2 = cn2, cn, cn1

        std::mem::swap(&mut cn, &mut cn2);
        // cn becomes cn2 -> all good
        // cn2 now is cn
        std::mem::swap(&mut cn1, &mut cn2);
        // cn now moved to cn1 -> all good
        // cn2 now is cn1 -> all good
    }
    poly_coef
}
impl Background {
    pub fn render(&self, intensities: &mut [f32], positions: &[f32]) {
        // assume that intensities is zeroed
        let iterator = intensities.iter_mut().zip(positions);

        match self {
            Background::None => (),
            Background::Polynomial { poly_coef, scale } => {
                let mut d_max = std::f32::MIN;
                let mut d_min = std::f32::MAX;
                for (i, (intensity, _)) in iterator.enumerate() {
                    let p = (i as f32 / (positions.len() - 1) as f32 - 0.5) * 2.0;
                    // map to [-1, 1]
                    let d = polynomial_at(poly_coef, p);

                    d_max = d_max.max(d);
                    d_min = d_min.min(d);

                    *intensity += d;
                }

                for i in intensities.iter_mut() {
                    *i = (*i - d_min) / (d_max - d_min) * scale;
                }
            }
            Background::Exponential { slope, scale } => {
                let mut d_max = std::f32::MIN;
                let mut d_min = std::f32::MAX;
                for (intensity, pos) in iterator {
                    let d = exp_at(*slope, *pos);
                    d_max = d_max.max(d);
                    d_min = d_min.min(d);
                    *intensity += d;
                }

                for i in intensities.iter_mut() {
                    *i = (*i - d_min) / (d_max - d_min) * scale;
                }
            }
        }
    }

    pub fn chebyshev_polynomial(chebyshev_coefs: &[f32], scale: f32) -> Self {
        if chebyshev_coefs.is_empty() {
            return Self::None;
        }

        Self::Polynomial {
            poly_coef: cheb2poly(chebyshev_coefs),
            scale,
        }
    }
}

fn exp_at(slope: f32, pos: f32) -> f32 {
    (slope * pos).exp()
}

fn polynomial_at(polynomial_coefficients: &[f32], pos: f32) -> f32 {
    polynomial_coefficients
        .iter()
        .enumerate()
        .map(|(power, c)| {
            c * pos.powi(
                i32::try_from(power).expect("unreasonably large vector of polynomial coefficients"),
            )
        })
        .sum()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn polynomial_at_simple_quadratic() {
        let coefs = [0.0, 0.0, 1.0];
        assert!((polynomial_at(&coefs, 10.0) - 100.0).abs() < 1e-5);
    }

    #[test]
    fn cheb2poly_single_coef() {
        assert_eq!(cheb2poly(&[1.0]), [1.0]);
        assert_eq!(cheb2poly(&[42.0]), [42.0]);
    }

    #[test]
    fn cheb2poly_two_coefs() {
        assert_eq!(cheb2poly(&[1.0, 1.0]), [1.0, 1.0]);
        assert_eq!(cheb2poly(&[34.0, 35.0]), [34.0, 35.0]);
    }

    #[rustfmt::skip]
    #[test]
    fn cheb2poly_base_cases() {
        assert_eq!(cheb2poly(&[ 1.0,  0.0, 0.0,    0.0,   0.0]), [  1.0,  0.0,  0.0,   0.0,   0.0]);
        assert_eq!(cheb2poly(&[ 1.0,  0.0, 0.0,    0.0,   0.0]), [  1.0,  0.0,  0.0,   0.0,   0.0]);
        assert_eq!(cheb2poly(&[ 0.0,  1.0, 0.0,    0.0,   0.0]), [  0.0,  1.0,  0.0,   0.0,   0.0]);
        assert_eq!(cheb2poly(&[ 0.0,  0.0, 1.0,    0.0,   0.0]), [ -1.0,  0.0,  2.0,   0.0,   0.0]);
        assert_eq!(cheb2poly(&[ 0.0,  0.0, 0.0,    1.0,   0.0]), [  0.0, -3.0,  0.0,   4.0,   0.0]);
        assert_eq!(cheb2poly(&[ 0.0,  0.0, 0.0,    0.0,   1.0]), [  1.0,  0.0, -8.0,   0.0,   8.0]);
        assert_eq!(cheb2poly(&[0.12, 0.34, 0.47, -0.22, -0.13]), [-0.48,  1.0, 1.98, -0.88, -1.04]);
    }
}
