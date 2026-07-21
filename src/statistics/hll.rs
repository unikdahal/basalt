//! HyperLogLog — probabilistic distinct-count (NDV) estimation. See
//! design-docs/basalt-phase3-lld.md §2.4.
//!
//! Exact distinct counts need a hash set holding every distinct value —
//! unbounded memory, and completely unnecessary for an optimizer that only
//! needs the right order of magnitude. Fixed memory: `2^precision` registers
//! of one byte each. Standard error ≈ `1.04 / sqrt(2^precision)`.
//!
//! Algorithm: hash each value, use the top `precision` bits to select a
//! register, count the leading zeros in the remaining bits, and keep the
//! maximum per register. A hash with `k` leading zeros appears roughly once
//! per `2^k` distinct values, so the maximum observed run length is a noisy
//! log of the cardinality — averaging across many registers (harmonic mean,
//! with bias correction) cuts the noise.
//!
//! Uses a fast, well-distributed hash (`std`'s `SipHash` via `DefaultHasher`
//! here, since this project has no other dependency on a faster hash) —
//! HyperLogLog isn't defending against adversarial keys, so a
//! cryptographically-motivated hash isn't required, but avoiding a new
//! dependency for it is a reasonable simplification for Phase 3.

use std::hash::{Hash, Hasher};

/// Probabilistic distinct-count estimator.
#[derive(Clone, Debug)]
pub struct HyperLogLog {
    registers: Vec<u8>,
    precision: u8,
}

impl HyperLogLog {
    /// `precision` selects `2^precision` registers; 14 is a common default
    /// (16 KB, ~0.8% standard error).
    ///
    /// # Panics
    /// Panics if `precision` is outside `4..=16` — outside that range the
    /// register-selection/hash-bit split below stops being sound.
    pub fn new(precision: u8) -> Self {
        assert!(
            (4..=16).contains(&precision),
            "HyperLogLog precision must be in 4..=16, got {precision}"
        );
        HyperLogLog {
            registers: vec![0u8; 1usize << precision],
            precision,
        }
    }

    /// Feeds one already-computed 64-bit hash into the sketch. Callers hash
    /// their own values (so any hash function/quality tradeoff is theirs to
    /// make); this only does the register-selection/leading-zeros step.
    pub fn add_hash(&mut self, hash: u64) {
        let m = self.registers.len() as u64;
        let index = (hash % m) as usize;
        // Leading zeros in the bits *not* used to select the register,
        // plus one (a run of zero leading zeros still counts as "seen").
        let remaining = hash / m;
        let rank = (remaining.leading_zeros() as u8 - self.precision + 1).max(1);
        self.registers[index] = self.registers[index].max(rank);
    }

    /// Convenience: hashes `value` with `std`'s default hasher and feeds it.
    pub fn add<T: Hash>(&mut self, value: &T) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hasher);
        self.add_hash(hasher.finish());
    }

    /// The estimated distinct count.
    pub fn estimate(&self) -> usize {
        let m = self.registers.len() as f64;
        let alpha = match self.registers.len() {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let sum: f64 = self.registers.iter().map(|&r| 2f64.powi(-(r as i32))).sum();
        let raw = alpha * m * m / sum;

        // Small-range correction: linear counting when many registers are
        // still zero, which is far more accurate than the raw estimate for
        // small cardinalities.
        let zeros = self.registers.iter().filter(|&&r| r == 0).count();
        if raw <= 2.5 * m && zeros > 0 {
            (m * (m / zeros as f64).ln()).round() as usize
        } else {
            raw.round() as usize
        }
    }

    /// Merges `other` into `self` via register-wise max — the property
    /// that matters for Phase 4: per-partition sketches computed
    /// independently combine into a global sketch with no re-scan of the
    /// underlying data, the same shape as `Accumulator::merge_batch` from
    /// Phase 2.
    ///
    /// # Errors
    /// Errors if `self` and `other` have different precisions (register
    /// counts), which would make a register-wise max meaningless.
    pub fn merge(&mut self, other: &Self) -> crate::error::Result<()> {
        if self.precision != other.precision {
            return Err(crate::error::BasaltError::Internal(format!(
                "cannot merge HyperLogLog sketches with different precision: {} vs {}",
                self.precision, other.precision
            )));
        }
        for (a, b) in self.registers.iter_mut().zip(other.registers.iter()) {
            *a = (*a).max(*b);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_is_within_reasonable_error_of_true_ndv() {
        let mut hll = HyperLogLog::new(12);
        let true_ndv = 10_000;
        for i in 0..true_ndv {
            hll.add(&i);
        }
        let estimate = hll.estimate() as f64;
        let error = (estimate - true_ndv as f64).abs() / true_ndv as f64;
        assert!(
            error < 0.1,
            "estimate {estimate} too far from true NDV {true_ndv} (error {error})"
        );
    }

    #[test]
    fn duplicate_values_do_not_inflate_the_estimate() {
        let mut hll = HyperLogLog::new(10);
        for _ in 0..1000 {
            hll.add(&"same-value");
        }
        assert!(
            hll.estimate() <= 5,
            "expected ~1 distinct value, got {}",
            hll.estimate()
        );
    }

    #[test]
    fn merge_matches_scanning_the_union_directly() {
        let mut a = HyperLogLog::new(12);
        let mut b = HyperLogLog::new(12);
        for i in 0..5000 {
            a.add(&i);
        }
        for i in 4000..9000 {
            b.add(&i);
        }
        let mut merged = a.clone();
        merged.merge(&b).unwrap();

        let mut direct = HyperLogLog::new(12);
        for i in 0..9000 {
            direct.add(&i);
        }

        let merged_est = merged.estimate() as f64;
        let direct_est = direct.estimate() as f64;
        let rel_diff = (merged_est - direct_est).abs() / direct_est;
        assert!(
            rel_diff < 0.05,
            "merged {merged_est} vs direct {direct_est}"
        );
    }

    #[test]
    fn merge_rejects_mismatched_precision() {
        let mut a = HyperLogLog::new(10);
        let b = HyperLogLog::new(12);
        assert!(a.merge(&b).is_err());
    }

    #[test]
    #[should_panic(expected = "precision must be in 4..=16")]
    fn rejects_out_of_range_precision() {
        HyperLogLog::new(20);
    }
}
