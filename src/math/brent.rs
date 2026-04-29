pub fn brent(
    f: impl Fn(f64) -> f64,
    mut a: f64,
    mut b: f64,
    atol: f64,
    maxiter: usize,
) -> Result<f64, String> {
    let mut fa = f(a);
    let mut fb = f(b);

    if fa == 0.0 {
        return Ok(a);
    }

    if fb == 0.0 {
        return Ok(b);
    }

    if fa * fb >= 0.0 {
        return Err(format!("Invalid Interval not brackenting root"));
    }

    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
    }

    let mut c = a;
    let mut fc;
    let mut d = 0.0;

    let mut mflag = true;

    for _ in 0..maxiter {
        fc = f(c);
        fb = f(b);
        fa = f(a);

        if fb == 0.0 {
            return Ok(b);
        }
        let mut s = if fa != fc && fb != fc {
            (a * fb * fc) / ((fa - fb) * (fa - fc))
                + (b * fa * fc) / ((fb - fa) * (fb - fc))
                + (c * fa * fb) / ((fc - fa) * (fc - fb))
        } else {
            b - fb * (b - a) / (fb - fc)
        };

        let cond1 = !((3.0 * a + b) / 4.0 < s && s < b);
        let cond2 = mflag && (s - b).abs() * 2.0 >= (b - c).abs();
        let cond3 = !mflag && (s - b).abs() * 2.0 >= (c - d).abs();
        let cond4 = mflag && (b - c).abs() < atol.abs();
        let cond5 = !mflag && (c - d).abs() < atol.abs();

        if cond1 || cond2 || cond3 || cond4 || cond5 {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = f(s);
        d = c; // d will not be used in the above conditions before setting here because on the first iteration mflag is set
        c = b;

        if fa * fs < 0.0 {
            b = s;
        } else {
            a = s;
        }

        if fa.abs() > fb.abs() {
            std::mem::swap(&mut a, &mut b);
        }

        if (b - a).abs() < atol {
            return Ok((a + b) / 2.0);
        }
    }

    Ok((a + b) / 2.0)
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn test_basic() {
        let f = |x: f64| x.powi(2) - 4.0;

        let brent_sol = brent(f, 0.0, 100.0, 1e-6, 200).expect("should not fail");

        assert!((brent_sol - 2.0) < 1e-6)
    }

    #[test]
    fn test_1_over_x() {
        let f = |x: f64| (1.0 / (x + 2.0) + 1.0 / x) - 1.0;

        let brent_sol = brent(f, 0.0, 5.0, 1e-6, 200).expect("should not fail");

        println!("{}", f(brent_sol));

        assert!(f(brent_sol).abs() < 1e-6)
    }
}
