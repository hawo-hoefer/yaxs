use std::str::FromStr;

use itertools::Itertools;

use crate::math::{Mat4, Vec3};

pub struct SymOp {
    mat: Mat4<f64>,
}

impl SymOp {
    pub fn apply(&self, pos: Vec3<f64>) -> Vec3<f64> {
        let pos = self.mat.homog_mul(pos);
        Vec3::new(pos.x, pos.y, pos.z)
    }
}

pub fn chop_integer(s: &str) -> Option<(i32, &str)> {
    let num_len: usize = s
        .chars()
        .take_while(|x| x.is_ascii_digit())
        .map(|x| x.len_utf8())
        .sum();

    if num_len == 0 {
        return None;
    }

    let num =
        std::str::from_utf8(&s.as_bytes()[..num_len]).expect("unrecoverable: non-utf8 char in cif");
    let rest =
        std::str::from_utf8(&s.as_bytes()[num_len..]).expect("unrecoverable: non-utf8 char in cif");

    Some((num.parse().expect("we only collected ascii numbers"), rest))
}

/// parse coefficient in an expression like 1x + 2y -3/4z +1/2
fn parse_coef(mut s: &str) -> Result<(f64, &str), String> {
    let sign = match s.chars().next() {
        Some('-') => {
            s = s
                .split_once('-')
                .expect("we know '-' is the first character")
                .1;
            -1.0
        }
        Some('+') => {
            s = s
                .split_once('+')
                .expect("we know '-' is the first character")
                .1;
            1.0
        }
        Some(_) => 1.0,
        None => return Err("Cannot parse empty coef".to_string()),
    };

    // find the delimiter character and put it in the rest
    let (num, mut rest) = chop_integer(s).unwrap_or((1, s));
    let num = num as f64;

    if rest.is_empty() {
        return Ok((num * sign, rest));
    }

    if rest.starts_with('/') {
        // parse a fraction
        rest = rest.split_once('/').expect("first char is '/'").1;
        let (den, r) = chop_integer(rest).ok_or_else(|| {
            format!("Expected integer denominator of fraction while parsing symmetry operation coefficient. Got '{s}'")
        })?;
        rest = r;
        let den = den as f64;
        return Ok((num / den * sign, rest));
    }

    Ok((num * sign, rest))
}

fn parse_xyz(mut s: &str) -> Result<[f64; 4], String> {
    let mut output = [0.0, 0.0, 0.0, 0.0];
    loop {
        // sequence of (+|-)?(num)?(/num)?(x|y|z)?
        // parse number
        // check '/'
        // parse number?
        // parse xyz?
        // put at correct place in vector
        s = s.trim_start();
        if s.is_empty() {
            break;
        }
        let (coef, rest) = parse_coef(s)?;
        s = rest;
        s = s.trim_start();
        match s.chars().next() {
            Some(c) if matches!(c as u8, b'x'..=b'z') => {
                let idx = (c as u8 - b'x') as usize;
                s = s.split_once(c).expect("we know the first character").1;
                output[idx] += coef;
            }
            Some(' ') => {
                s = s.split_once(' ').expect("we know the first character").1;
                output[3] += coef;
            }
            None => {
                // we're at the end
                output[3] += coef;
                break;
            }
            Some(a) => {
                return Err(format!(
                    "Invalid direction. Expected x, y, or z, but got {a}"
                ))
            }
        }
    }

    Ok(output)
}

impl FromStr for SymOp {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let Some((a, b, c)) = s.split(',').collect_tuple::<(&str, &str, &str)>() else {
            return Err(format!("Invalid number of components in symop: '{s}'"));
        };
        let [w11, w12, w13, wx] = parse_xyz(a)?;
        let [w21, w22, w23, wy] = parse_xyz(b)?;
        let [w31, w32, w33, wz] = parse_xyz(c)?;
        Ok(Self {
            #[rustfmt::skip]
            mat: Mat4::new(
                w11, w12, w13,  wx,
                w21, w22, w23,  wy,
                w31, w32, w33,  wz,
                0.0, 0.0, 0.0, 1.0,
            ),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_identity() {
        let op: SymOp = "x,y,z".parse().expect("valid symop");
        assert_eq!(op.mat, Mat4::<f64>::identity())
    }

    #[test]
    fn parse_ok() {
        let op: SymOp = "x+1/3,-y+x,-3z".parse().expect("valid symop");
        #[rustfmt::skip]
        let exp = Mat4::new(
            1.0,  0.0,  0.0, 1.0 / 3.0,
            1.0, -1.0,  0.0,       0.0,
            0.0,  0.0, -3.0,       0.0,
            0.0,  0.0,  0.0,       1.0,
        );
        assert_eq!(op.mat, exp);
    }

    #[test]
    fn parse_coef_err() {
        let coef = parse_coef("1/x").expect_err("invalid coef");
        assert_eq!(coef, "Expected integer denominator of fraction while parsing symmetry operation coefficient. Got '1/x'")
    }

    #[test]
    fn parse() {
        let symops = [
            "x, z, y",
            "z, -x+1/4, -y+1/4",
            "-z+1/4, -x+1/4, y",
            "-z+1/4, x, -y+1/4",
            "z, x, y",
            "-y+1/4, z, -x+1/4",
            "y, -z+1/4, -x+1/4",
            "-y+1/4, -z+1/4, x",
            "y, z, x",
            "-x+1/4, -y+1/4, z",
            "-x+1/4, y, -z+1/4",
            "x, -y+1/4, -z+1/4",
            "x, y, z",
            "-z, y+1/4, x+1/4",
            "z+3/4, -y+1/2, x+1/4",
            "z+3/4, y+1/4, -x+1/2",
            "-z, -y+1/2, -x+1/2",
            "y+3/4, x+1/4, -z+1/2",
            "-y, x+1/4, z+1/4",
        ];
        #[rustfmt::skip]
        let mats = [
            [1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0, 0.0, 0.0, 1.0],
            [0.0,  0.0,  1.0,  0.0,  -1.0, 0.0,  0.0,  0.25, 0.0,  -1.0, 0.0,  0.25, 0.0, 0.0, 0.0, 1.0],
            [0.0,  0.0,  -1.0, 0.25, -1.0, 0.0,  0.0,  0.25, 0.0,  1.0,  0.0,  0.0,  0.0, 0.0, 0.0, 1.0],
            [0.0,  0.0,  -1.0, 0.25, 1.0,  0.0,  0.0,  0.0,  0.0,  -1.0, 0.0,  0.25, 0.0, 0.0, 0.0, 1.0],
            [0.0,  0.0,  1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0, 0.0, 0.0, 1.0],
            [0.0,  -1.0, 0.0,  0.25, 0.0,  0.0,  1.0,  0.0,  -1.0, 0.0,  0.0,  0.25, 0.0, 0.0, 0.0, 1.0],
            [0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  -1.0, 0.25, -1.0, 0.0,  0.0,  0.25, 0.0, 0.0, 0.0, 1.0],
            [0.0,  -1.0, 0.0,  0.25, 0.0,  0.0,  -1.0, 0.25, 1.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 1.0],
            [0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0,  0.0,  0.25, 0.0,  -1.0, 0.0,  0.25, 0.0,  0.0,  1.0,  0.0,  0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0,  0.0,  0.25, 0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  -1.0, 0.25, 0.0, 0.0, 0.0, 1.0],
            [1.0,  0.0,  0.0,  0.0,  0.0,  -1.0, 0.0,  0.25, 0.0,  0.0,  -1.0, 0.25, 0.0, 0.0, 0.0, 1.0],
            [1.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0, 0.0, 0.0, 1.0],
            [0.0,  0.0,  -1.0, 0.0,  0.0,  1.0,  0.0,  0.25, 1.0,  0.0,  0.0,  0.25, 0.0, 0.0, 0.0, 1.0],
            [0.0,  0.0,  1.0,  0.75, 0.0,  -1.0, 0.0,  0.5,  1.0,  0.0,  0.0,  0.25, 0.0, 0.0, 0.0, 1.0],
            [0.0,  0.0,  1.0,  0.75, 0.0,  1.0,  0.0,  0.25, -1.0, 0.0,  0.0,  0.5,  0.0, 0.0, 0.0, 1.0],
            [0.0,  0.0,  -1.0, 0.0,  0.0,  -1.0, 0.0,  0.5,  -1.0, 0.0,  0.0,  0.5,  0.0, 0.0, 0.0, 1.0],
            [0.0,  1.0,  0.0,  0.75, 1.0,  0.0,  0.0,  0.25, 0.0,  0.0,  -1.0, 0.5,  0.0, 0.0, 0.0, 1.0],
            [0.0,  -1.0, 0.0,  0.0,  1.0,  0.0,  0.0,  0.25, 0.0,  0.0,  1.0,  0.25, 0.0, 0.0, 0.0, 1.0],
        ];

        for (i, (op_str, coefs)) in symops.iter().zip(mats).enumerate() {
            let op: SymOp = op_str.parse().expect("valid symop");
            assert_eq!((i, op.mat), (i, Mat4::from_slice(&coefs)));
        }
    }
}
