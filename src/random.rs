pub trait SeededDistribution {
    fn new(seed: &str) -> Self;

    /// returns from range -1.0 to 1.0
    fn next(&mut self) -> f64;
}

pub struct UniformDistribution {
    state: u128,
}

impl SeededDistribution for UniformDistribution {
    fn new(seed: &str) -> Self {
        Self {
            state: fnv_hash(seed) | 1, // Lehmer state must be odd
        }
    }

    fn next(&mut self) -> f64 {
        self.state = lehmer_step(self.state);
        construct_float((self.state >> 64) as u64)
    }
}

/// Pin the exponent to the [2.0, 4.0) range, fill the mantissa with the
/// top 52 random bits, then shift down to [-1.0, 1.0).
fn construct_float(r: u64) -> f64 {
    // 0x4000... = sign 0, exponent 1024 (biased), empty mantissa == 2.0
    f64::from_bits(0x4000_0000_0000_0000 | (r >> 12)) - 3.0
}

const FNV_PRIME: u128 = 309_485_009_821_345_068_724_781_371;
const FNV_OFFSET_BASIS: u128 = 144_066_263_297_769_815_596_495_629_667_062_367_629;

pub fn fnv_hash(seed: &str) -> u128 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in seed.bytes() {
        hash ^= byte as u128;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

const LEHMER_A: u128 = 92563704562804186071655587898373606109;

pub fn lehmer_step(state: u128) -> u128 {
    state.wrapping_mul(LEHMER_A)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_uniform_distribution() {
        let mut seen = HashSet::<[u8; 8]>::new();

        let mut source = UniformDistribution::new("seed");

        for _ in 0..1000 {
            let value = source.next();

            if value > 1.0 || value < -1.0 {
                panic!("Value {} out of range", value);
            }

            let prev = seen.insert(value.to_be_bytes());

            if !prev {
                panic!("We already saw value {}", value)
            }
        }
    }
}
