use itertools::Itertools;
use rand::Rng;
use serde::Serialize;

use crate::math::linalg::{ColVec, Mat};

use super::linalg::Vec4;

/// sample uniformly from the surface of the DIMS-sphere
///
/// * `n`: number of samples
/// * `rng`: underlying random number generator
pub fn sample_sphere_unif<const DIMS: usize>(rng: &mut impl Rng) -> ColVec<f64, DIMS> {
    let mut ret = ColVec::<f64, DIMS>::zeros();
    for v in ret.iter_values_mut() {
        *v = std_normal_box_muller_tf(rng);
    }
    ret.normalize_inplace();

    ret
}

/// Sample a uniformly random unit quaternion using the subgroup algorithm described in
///
/// Shoemake, Ken. "Uniform random rotations." Graphics Gems III (IBM Version).
/// Morgan Kaufmann, 1992. 124-132.
/// DOI: <https://doi.org/10.1016/B978-0-08-050755-2.50036-1>
///
/// * `rng`: underlying random number generator
pub fn sample_unit_quaternion_subgroup_algorithm(rng: &mut impl Rng) -> Vec4<f64> {
    let x_0 = rng.random_range(0.0..1.0);
    let x_1 = rng.random_range(0.0..1.0);
    let x_2 = rng.random_range(0.0..1.0);

    let theta_1 = 2.0 * std::f64::consts::PI * x_1;
    let theta_2 = 2.0 * std::f64::consts::PI * x_2;

    let r_1 = (1.0f64 - x_0).sqrt();
    let r_2 = x_0.sqrt();

    Vec4::new(
        theta_1.sin() * r_1,
        theta_1.cos() * r_1,
        theta_2.sin() * r_2,
        theta_2.cos() * r_2,
    )
}

/// sample from a standard normal distribution using the box-muller transform
///
/// * `rng`: rng to use
pub fn std_normal_box_muller_tf(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();

    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A bingham distribution over elements of $S^{N-1}$
#[derive(PartialEq, Debug, Clone, Serialize)]
pub struct BinghamDistribution<const N: usize> {
    a: Mat<f64, N, N>,
    k: ColVec<f64, N>,
    acg_envelope: ACGDistribution<N>,
}

/// Angular Centered Gaussian Distribution
/// also called projected normal distribution
#[derive(Debug, Clone, Serialize, PartialEq)]
struct ACGDistribution<const N: usize> {
    omega: Mat<f64, N, N>,
    l: Mat<f64, N, N>,
}

impl<const N: usize> ACGDistribution<N> {
    fn new(omega: Mat<f64, N, N>) -> Result<Self, String> {
        let l = omega.cholesky_decompose()?;
        Ok(Self { omega, l })
    }

    /// Compute the unnormalized probability density function
    ///
    /// * `x`: point to compute pdf for. Is assumed to have length 1 (/ be on $S^{N-1}$)
    pub fn pdf_unnorm(&self, x: &ColVec<f64, N>) -> f64 {
        x.transpose().matmul(&self.omega).matmul(x).item().powi(-2)
    }

    /// compute the logarithm of the unnormalized probability density function
    ///
    /// * `x`: value n $S^{N-1}$, needs to have length 1
    pub fn log_pdf_unnorm(&self, x: &ColVec<f64, N>) -> f64 {
        self.pdf_unnorm(x).ln()
    }

    /// Sample from Angular Centered Gaussian distribution
    ///
    /// we first sample a vector of standard normally distributed values using
    /// the Box-Muller Transform. They are mapped to a multivariate gaussian using
    /// the method described on page 315 (PDF page 328) in
    /// Gentle, James E. Computational statistics. Vol. 308. New York: Springer, 2009.
    /// https://doi.org/10.1007/978-0-387-98144-4
    ///
    ///
    /// * `rng`: Random number generator
    pub fn sample(&self, rng: &mut impl Rng) -> ColVec<f64, N> {
        let mut v = ColVec::zeros();
        for i in v.iter_values_mut() {
            *i = std_normal_box_muller_tf(rng);
        }

        // multivariate gaussian distributed
        let x = self.l.matmul(&v);
        // map onto sphere

        x.normalize()
    }
}

impl<const N: usize> BinghamDistribution<N> {
    /// Create a new BinghamDistribution from decomposition of the underlying
    /// gaussian distribution's covariance matrix A
    ///
    /// $$A = u K u^T$$
    ///
    /// where the orthogonal matrix $u \in R^{N \times N}$ contains an orthonormal basis
    /// with eigenvectors of $A$, and the vector $k$ contains the eigenvalues
    ///
    /// * `k`: eigenvalues of decomposition
    /// * `u`: orthonormal basis of eigenvectors of decomposition
    pub fn try_new(k: ColVec<f64, N>, u: Mat<f64, N, N>) -> Result<Self, String> {
        let u_t = u.transpose();
        let a = u_t.matmul_diag(&k).matmul(&u);
        let omega = Mat::<f64, N, N>::identity() + a.scale(2.0f64);
        let acg_envelope = ACGDistribution::new(omega)?;

        Ok(BinghamDistribution { a, acg_envelope, k })
    }

    pub fn ks(&self) -> &ColVec<f64, N> {
        &self.k
    }

    /// compute the logarithm of the unnormalized probability density function
    ///
    /// * `x`: value n $S^{N-1}$, needs to have length 1
    pub fn log_pdf_unnorm(&self, x: &ColVec<f64, N>) -> f64 {
        x.transpose().matmul(&self.a).matmul(x).item()
    }

    /// compute the unnormalized probability density function
    ///
    /// * `x`: value n $S^{N-1}$, needs to have length 1
    pub fn pdf_unnorm(&self, x: &ColVec<f64, N>) -> f64 {
        self.log_pdf_unnorm(x).exp()
    }

    /// Sample from the distribution
    ///
    /// * `rng`: underlying rng
    pub fn sample(&self, rng: &mut impl Rng) -> ColVec<f64, N> {
        const GAMMA_FN_THREE_HALFS: f64 = 0.8862269254527579;
        const MAXITER: usize = 100_000;

        for _ in 0..MAXITER {
            use std::f64::consts::{E, TAU};
            let s = self.acg_envelope.sample(rng);
            let w: f64 = rng.random_range(0.0..=1.0);
            let m = (TAU * E) * (3.0 / (2.0 * E)).powf(1.5) / GAMMA_FN_THREE_HALFS;

            // from rejection sampling method
            // w < f_bingham / (m * f_acg)
            // log(w) < log(f_bingham / (m * f_acg))
            // log(w) < log(f_bingham) - log(m) - log(f_acg)
            // log(w) + log(m) < log(f_bingham) - log(f_acg)

            let log_f_acg = self.acg_envelope.log_pdf_unnorm(&s);
            let log_f_bingham = self.log_pdf_unnorm(&s);
            if w.ln() + m.ln() < log_f_bingham - log_f_acg {
                // accept sample
                return s;
            }
        }
        panic!("Could not sample from bingham distribution after {MAXITER} tries")
    }
}

#[derive(Clone)]
pub struct HitAndRunPolytopeSampler<const N: usize, const M: usize> {
    a: Mat<f64, N, M>,
    b: ColVec<f64, N>,
    x: ColVec<f64, M>,
    n_thinning: usize,
}

fn highs_model_status_to_str(status: highs::HighsModelStatus) -> &'static str {
    use highs::HighsModelStatus::*;
    match status {
        Optimal => "Optimal",
        NotSet => "NotSet",
        LoadError => "LoadError",
        ModelError => "ModelError",
        PresolveError => "PresolveError",
        SolveError => "SolveError",
        PostsolveError => "PostsolveError",
        ModelEmpty => "ModelEmpty",
        Infeasible => "Infeasible",
        UnboundedOrInfeasible => "UnboundedOrInfeasible",
        Unbounded => "Unbounded",
        ObjectiveBound => "ObjectiveBound",
        ObjectiveTarget => "ObjectiveTarget",
        ReachedTimeLimit => "ReachedTimeLimit",
        ReachedIterationLimit => "ReachedIterationLimit",
        Unknown => "Unknown",
    }
}

fn find_interior_point<const N: usize, const M: usize>(
    a: &Mat<f64, N, M>,
    b: &ColVec<f64, N>,
) -> Result<ColVec<f64, M>, String> {
    use highs::{RowProblem, Sense};

    let create_problem = || {
        let mut pb = RowProblem::new();
        let variables = (0..M)
            .map(|_| pb.add_column::<f64, _>(0.0, ..))
            .collect_vec();
        let slack = pb.add_column::<f64, _>(-1.0, ..);
        for i in 0..N {
            pb.add_row(
                ..=b[i],
                a.col(i)
                    .iter_values()
                    .zip(&variables)
                    .chain([(&2.0, &slack)])
                    .map(|(&fac, &var)| (var, fac)),
            );
        }

        pb.add_row(..=0.0, [(slack, -1.0)]);
        (variables, slack, pb)
    };

    let (_, _, pb) = create_problem();
    let sol_model = pb
        .try_optimise(Sense::Minimise)
        .map_err(|err| format!("Could not find interior point: {err:?}"))?
        .try_solve()
        .map_err(|err| format!("Could not find interior point: {err:?}"))?;

    use highs::HighsModelStatus::*;

    let sol_model = match sol_model.status() {
        Infeasible => return Err("Infeasible model".to_string()),
        Optimal => {
            // all good
            sol_model
        }
        Unbounded => {
            let (_, slack, mut pb) = create_problem();

            pb.add_row(..=1.0, &[(slack, 1.0)]);

            let sol_model = pb
                .try_optimise(Sense::Minimise)
                .map_err(|err| format!("Could not find interior point: {err:?}"))?
                .try_solve()
                .map_err(|err| format!("Could not find interior point: {err:?}"))?;

            if sol_model.status() != Optimal {
                return Err(format!(
                    "Could not find interior point for bounded problem: highs returned {s}",
                    s = highs_model_status_to_str(sol_model.status())
                ));
            }
            sol_model
        }
        _ => {
            return Err(format!(
                "Could not find interior point for bounded problem: highs returned {s}",
                s = highs_model_status_to_str(sol_model.status())
            ));
        }
    };

    let mut res = ColVec::<_, M>::zeros();
    // this automatically truncates the slack value which we want to ignore anyway
    for (v, r) in sol_model
        .get_solution()
        .columns()
        .iter()
        .zip(res.iter_values_mut())
    {
        *r = *v;
    }

    Ok(res)
}

impl<const N: usize, const M: usize> HitAndRunPolytopeSampler<N, M> {
    pub fn try_new(
        a: Mat<f64, N, M>,
        b: ColVec<f64, N>,
        n_warmup: usize,
        n_thinning: usize,
        rng: &mut impl Rng,
    ) -> Result<Self, String> {
        let x0 = find_interior_point(&a, &b)?;
        assert!(x0.iter_values().all(|x| *x >= 0.0), "{}", x0);

        let mut ret = Self {
            x: x0,
            a,
            b,
            n_thinning,
        };

        for _ in 0..n_warmup {
            _ = ret.update_mcmc_position(rng)
        }

        Ok(ret)
    }

    pub fn sample(&mut self, rng: &mut impl Rng) -> ColVec<f64, M> {
        for _ in 0..self.n_thinning - 1 {
            _ = self.update_mcmc_position(rng);
        }
        self.update_mcmc_position(rng)
    }

    /// return the next position of the markov chain
    ///
    /// * `rng`:
    pub fn update_mcmc_position(&mut self, rng: &mut impl Rng) -> ColVec<f64, M> {
        let mut w = &self.b - &self.a.matmul(&self.x);
        assert!(w.iter_values().all(|x| *x >= 0.0), "{}", w);

        let dk = sample_sphere_unif::<M>(rng);
        let ar = self.a.matmul(&dk);

        // from BoTorch code:
        //
        // Given x, the next point in the chain is x+scale*d.
        //
        // It must satisfy A(x+scale*r)<=b, which implies A*scale*r<=b-Ax,
        // so scale<=(b-Ax)/ar for ar>0, and scale>=(b-Ax)/ar for ar<0.
        //
        // If x is at the boundary, b - Ax = 0. If ar > 0, then we must
        // have scale <= 0. If ar < 0, we must have scale >= 0.
        // ar == 0 is an unlikely event that provides no signal.
        // b - A @ x is always >= 0, clamping for numerical tolerances.
        //
        // w = (b - A @ x).squeeze().clamp(min=0.0) / ar
        w = &w.map(|x| x.clamp(0.0, std::f64::MAX)) / &ar;

        // # Find upper bound for scale. If there are no constraints on
        // # the upper bound of scale, set it to a large value.
        // pos = w > 0
        // scale_max = w[pos].min().item() if pos.any() else large_constant
        //
        // # Find lower bound for scale.
        // neg = w < 0
        // scale_min = w[neg].max().item() if neg.any() else -large_constant

        //              0         1
        // x0 --------->|-------->|
        // |------------|
        //   s_plus
        let mut s_plus = w
            .iter_values()
            .filter(|&&w_i| w_i > 0.0)
            .min_by(|&&a, &&b| a.partial_cmp(&b).expect("not nan"))
            .unwrap_or(&std::f64::MAX)
            .to_owned();

        let mut s_minus = w
            .iter_values()
            .filter(|&&w_i| w_i < 0.0)
            .max_by(|&&a, &&b| a.partial_cmp(&b).expect("not nan"))
            .unwrap_or(&std::f64::MIN)
            .to_owned();

        let w_eq_0 = w.map(|&x| x == 0.0);
        if w_eq_0
            .zip(&ar.map(|x| *x > 0.0))
            .any(|(&w_is_0, &ar_gt_0)| w_is_0 && ar_gt_0)
        {
            s_plus = s_plus.min(0.0f64);
        }

        if w_eq_0
            .zip(&ar.map(|x| *x < 0.0))
            .any(|(&w_i_is_0, &ar_lt_0)| w_i_is_0 && ar_lt_0)
        {
            s_minus = s_minus.max(0.0f64);
        }

        let scale = rng.random_range(s_minus..=s_plus);
        self.x += dk.scale(scale);

        self.x.clone()
    }
}
