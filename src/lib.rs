use std::{fmt::Debug, io::Cursor};

use murmur3::murmur3_32;
use rand::{rng, rngs::StdRng, Rng, SeedableRng};

pub struct SimpleBloomFilter {
    epsilon: f32,
    n: usize,
    m: usize,
    k: usize,

    rng_seed: u64,
    hash_seed: u64,
    bit_array: Vec<u64>,
}

impl SimpleBloomFilter {
    pub fn new(n: usize, epsilon: f32) -> Self {
        let rng_seed: u64 = rng().random();

        Self::new_with_seed(n, epsilon, rng_seed)
    }

    pub fn new_with_seed(n: usize, epsilon: f32, rng_seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(rng_seed);

        let k = (-(f32::ln(epsilon) / f32::ln(2.0))).round() as usize;
        let m = (-(n as f32 * f32::ln(epsilon)) / (f32::ln(2.0).powi(2))).round() as usize;

        let hash_seed = rng.random();
        let bit_array = vec![0; (m + 63) / 64];

        Self {
            epsilon,
            n,
            m,
            k,
            hash_seed,
            bit_array,
            rng_seed,
        }
    }

    pub fn insert<T: std::convert::AsRef<[u8]>>(&mut self, value: T) {
        let indexes = self.get_indexes(&value);

        for i in indexes {
            self.set_bit(i);
        }
    }

    pub fn test<T: std::convert::AsRef<[u8]>>(&self, value: T) -> bool {
        self.get_indexes(&value)
            .into_iter()
            .all(|i| self.get_bit(i) == true)
    }

    pub fn union(&mut self, other: &Self) {
        assert!(
            self.rng_seed == other.rng_seed,
            "Bloom Filters have different hash functions distributions"
        );
        assert!(
            self.epsilon == other.epsilon,
            "Bloom Filters have different failure probability"
        );
        assert!(
            self.n == other.n,
            "Bloom Filters have different expected size"
        );

        for (a, b) in self.bit_array.iter_mut().zip(&other.bit_array) {
            *a |= *b;
        }
    }

    pub fn intersect(&mut self, other: &Self) {
        assert!(
            self.rng_seed == other.rng_seed,
            "Bloom Filters have different hash functions distributions"
        );
        assert!(
            self.epsilon == other.epsilon,
            "Bloom Filters have different failure probability"
        );
        assert!(
            self.n == other.n,
            "Bloom Filters have different expected size"
        );

        for (a, b) in self.bit_array.iter_mut().zip(&other.bit_array) {
            *a &= *b;
        }
    }

    // Murmur3 hash function with Kirsch-Mitzenmacher optimization
    fn get_indexes<T: std::convert::AsRef<[u8]>>(&self, value: &T) -> Vec<usize> {
        let h1 = murmur3_32(&mut Cursor::new(value), (self.hash_seed >> 32) as u32).unwrap() as u64;
        let h2 = murmur3_32(&mut Cursor::new(value),(self.hash_seed & 0xFFFF_FFFF) as u32).unwrap() as u64;

        (0..self.k)
            .map(|i| (h1.wrapping_add(i as u64 * h2) as usize) % self.m)
            .collect()
    }

    fn set_bit(&mut self, i: usize) {
        let (word, bit) = (i / 64, i % 64);
        self.bit_array[word] |= 1 << bit;
    }

    fn get_bit(&self, i: usize) -> bool {
        let (word, bit) = (i / 64, i % 64);
        (self.bit_array[word] & (1 << bit)) != 0
    }
}

impl Debug for SimpleBloomFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleBloomFilter")
            .field("epsilon", &self.epsilon)
            .field("n", &self.n)
            .field("m", &self.m)
            .field("k", &self.k)
            .field("hash_seeds", &self.hash_seed)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_test() {
        let mut bf = SimpleBloomFilter::new(1000, 0.01);
        bf.insert("hello");
        bf.insert("world");

        assert!(bf.test("hello"));
        assert!(bf.test("world"));
        assert!(!bf.test("other"));
    }

    #[test]
    fn test_false_positive_rate() {
        let n = 1000;
        let epsilon = 0.05;
        let mut bf = SimpleBloomFilter::new(n, epsilon);

        for i in 0..n {
            bf.insert(format!("key{}", i));
        }

        let mut false_positives = 0;
        let test_count = 10_000;

        for i in n..n + test_count {
            if bf.test(format!("key{}", i)) {
                false_positives += 1;
            }
        }

        let fp_rate = false_positives as f32 / test_count as f32;
        println!("False positive rate: {}", fp_rate);
        assert!(fp_rate <= epsilon * 1.5);
    }

    #[test]
    fn test_union() {
        let mut bf1 = SimpleBloomFilter::new_with_seed(1000, 0.01, 42);
        let mut bf2 = SimpleBloomFilter::new_with_seed(1000, 0.01, 42);

        bf1.insert("a");
        bf2.insert("b");

        assert!(bf1.test("a"));
        assert!(!bf1.test("b"));
        assert!(bf2.test("b"));
        assert!(!bf2.test("a"));

        bf1.union(&bf2);

        assert!(bf1.test("a"));
        assert!(bf1.test("b"));
    }

    #[test]
    fn test_intersect() {
        let mut bf1 = SimpleBloomFilter::new_with_seed(1000, 0.01, 42);
        let mut bf2 = SimpleBloomFilter::new_with_seed(1000, 0.01, 42);

        bf1.insert("common");
        bf1.insert("only1");

        bf2.insert("common");
        bf2.insert("only2");

        bf1.intersect(&bf2);

        assert!(bf1.test("common"));
        assert!(!bf1.test("only1"));
        assert!(!bf1.test("only2"));
    }

    #[test]
    #[should_panic(expected = "Bloom Filters have different hash functions distributions")]
    fn test_union_incompatible_seed() {
        let mut bf1 = SimpleBloomFilter::new_with_seed(1000, 0.01, 1);
        let bf2 = SimpleBloomFilter::new_with_seed(1000, 0.01, 2);
        bf1.union(&bf2);
    }

    #[test]
    #[should_panic(expected = "Bloom Filters have different failure probability")]
    fn test_union_incompatible_epsilon() {
        let mut bf1 = SimpleBloomFilter::new_with_seed(1000, 0.01, 42);
        let bf2 = SimpleBloomFilter::new_with_seed(1000, 0.02, 42);
        bf1.union(&bf2);
    }

    #[test]
    #[should_panic(expected = "Bloom Filters have different expected size")]
    fn test_union_incompatible_n() {
        let mut bf1 = SimpleBloomFilter::new_with_seed(1000, 0.01, 42);
        let bf2 = SimpleBloomFilter::new_with_seed(2000, 0.01, 42);
        bf1.union(&bf2);
    }
}
