#[derive(Clone, PartialEq, Debug)]
pub enum Background {
    None,
    Polynomial { coef: Vec<f64> },
    Exponential(f64),
}

impl Background {
    pub fn render(&self, pat: &mut [f64], two_thetas: &[f64]) {
        let iterator = pat.iter_mut().zip(two_thetas);

        match self {
            Background::None => return,
            Background::Polynomial { coef: poly_coefs } => {
                for (intensity, pos) in iterator {
                    *intensity += chebyshev_at(poly_coefs, *pos);
                }
            }
            Background::Exponential(slope) => {
                for (intensity, pos) in iterator {
                    *intensity += exp_at(*slope, *pos);
                }
            }
        }
    }

    pub fn chebyshev_polynomial(chebyshev_coefs: &[f64]) -> Self {
        // TODO: Test this
        if chebyshev_coefs.is_empty() {
            return Self::Polynomial { coef: Vec::new() };
        }

        let mut poly_coefs = Vec::with_capacity(chebyshev_coefs.len());
        poly_coefs.resize(chebyshev_coefs.len(), 0.0f64);

        for coef in poly_coefs.iter_mut() {
            *coef += chebyshev_coefs[0] * 1.0;
        }
        if chebyshev_coefs.len() == 1 {
            return Background::Polynomial { coef: poly_coefs };
        }

        let mut cn2 = Vec::with_capacity(chebyshev_coefs.len());
        cn2.resize(chebyshev_coefs.len(), 0.0f64);
        cn2[0] = 1.0;

        let mut cn1 = Vec::with_capacity(chebyshev_coefs.len());
        cn1.resize(chebyshev_coefs.len(), 0.0f64);
        cn1[1] = 1.0;

        for poly_coef in poly_coefs.iter_mut() {
            *poly_coef += chebyshev_coefs[1];
        }

        if poly_coefs.len() == 2 {
            return Background::Polynomial { coef: poly_coefs };
        }

        let mut cn = Vec::with_capacity(chebyshev_coefs.len());
        cn.resize(chebyshev_coefs.len(), 0.0f64);
        for i in 2..chebyshev_coefs.len() {
            // cn[:] = 0
            cn.iter_mut().for_each(|coef| *coef = 0.0);

            // cn[1 : i + 1] = 2 * cn1[:i]
            for j in 1..(i + 1) {
                cn[j] -= 2.0 * cn1[j];
            }

            // cn[:i] -= cn2[:i]
            for j in 0..i {
                cn[j] -= cn2[j];
            }

            // out += (coefs[:, i, None] * cn).astype(coefs.dtype)  # type: ignore
            poly_coefs.iter_mut().enumerate().for_each(|(j, coef)| {
                *coef += chebyshev_coefs[j] * cn[j];
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

        Self::Polynomial { coef: poly_coefs }
    }
}

fn exp_at(slope: f64, pos: f64) -> f64 {
    (slope * pos).exp()
}

fn chebyshev_at(polynomial_coefficients: &[f64], pos: f64) -> f64 {
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
