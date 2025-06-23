use rand::{Rng, RngCore, SeedableRng};
use rand_xoshiro::{SplitMix64, Xoshiro256PlusPlus};

fn std_normal_box_muller_tf(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();

    return (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
}

#[derive(Debug, Clone)]
pub enum Noise {
    Gaussian { sigma: f64 },
    Uniform { min: f64, max: f64 },
}

pub fn get_xoshiro256_seed(seed: u64) -> [u8; 32] {
    let mut seed_gen = SplitMix64::seed_from_u64(seed);
    let mut seed = [0u8; 32];
    seed_gen.fill_bytes(seed.as_mut());
    seed
}

impl Noise {
    pub fn apply(&self, intensities: &mut [f32], seed: u64) {
        let seed = get_xoshiro256_seed(seed);
        let mut rng = Xoshiro256PlusPlus::from_seed(seed);
        match self {
            Noise::Gaussian { sigma } => {
                for i in intensities.iter_mut() {
                    *i += (std_normal_box_muller_tf(&mut rng) * sigma) as f32;
                }
            }
            Noise::Uniform { min, max } => {
                for i in intensities.iter_mut() {
                    *i += (rng.random::<f64>() * (max - min) + min) as f32;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /* Generate the next random u64 using the Xoshiro256++ rng
     * as implemented by David Blackman and Sebastiano Vigna
     * from https://prng.di.unimi.it/xoshiro256plusplus.c
     *
     * this is just a dummy implementation mirroring the GPU version
     * to test whether the seed generation is working and whether they
     * produce the same values
     */
    fn xoshiro_dummy(s: &mut [u64; 4]) -> u64 {
        fn rotl(x: u64, k: i32) -> u64 {
            (x << k) | (x >> (64 - k))
        }

        let result = rotl(s[0].overflowing_add(s[3]).0, 23)
            .overflowing_add(s[0])
            .0;
        let t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 45);

        result
    }

    fn xoshiro256_double(state: &mut [u64; 4]) -> f64 {
        let result = xoshiro_dummy(state);
        return result as f64 / std::u64::MAX as f64;
    }

    #[test]
    fn seed_u64_gen() {
        let base_seed = 1234;
        let mut ref_rng = Xoshiro256PlusPlus::seed_from_u64(base_seed);
        let mut rng_state: [u64; 4] =
            unsafe { core::mem::transmute(get_xoshiro256_seed(base_seed)) };
        for _ in 0..1000 {
            let cuda_dummy_rand = xoshiro_dummy(&mut rng_state);
            let ref_rand: u64 = ref_rng.random();
            assert_eq!(ref_rand, cuda_dummy_rand);
        }
    }

    #[test]
    fn double_generation() {
        let base_seed = 1234;
        let mut ref_rng = Xoshiro256PlusPlus::seed_from_u64(base_seed);
        let mut rng_state: [u64; 4] =
            unsafe { core::mem::transmute(get_xoshiro256_seed(base_seed)) };
        for _ in 0..1000 {
            let cuda_dummy_rand = xoshiro256_double(&mut rng_state);
            let ref_rand: f64 = ref_rng.random_range(0.0..1.0);
            // can't test for equality due to rounding issues (probably)
            assert!((ref_rand - cuda_dummy_rand).abs() < 1e-15);
        }
    }
}
