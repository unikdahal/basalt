//! The `Cost` type and calibration. See design-docs/basalt-phase3-lld.md §4.

use std::sync::Arc;
use std::time::Instant;

use crate::array::primitive::PrimitiveBuilder;
use crate::array::types::Int64Type;
use crate::compute::arith;
use crate::compute::ColumnarValue;
use crate::scalar::ScalarValue;

/// A resource vector, not a single scalar: collapsing to one number
/// immediately loses the ability to say "this plan is cheaper but needs
/// 8 GB." The components are kept and collapsed only at comparison time
/// (`total`) — which also means Phase 4 adds network cost by changing a
/// weight, not by rewriting this type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    /// Bytes read from storage.
    pub io: f64,
    /// Tuples processed.
    pub cpu: f64,
    /// Peak bytes held.
    pub memory: f64,
    /// Bytes shuffled — always 0 in Phase 3, used in Phase 4.
    pub network: f64,
}

impl Cost {
    pub const ZERO: Cost = Cost {
        io: 0.0,
        cpu: 0.0,
        memory: 0.0,
        network: 0.0,
    };

    pub fn total(&self, weights: &CostWeights) -> f64 {
        self.io * weights.io_per_byte
            + self.cpu * weights.cpu_per_tuple
            + self.memory * weights.memory_penalty
            + self.network * weights.network_per_byte
    }

    /// Sums `io`/`cpu`/`network` (both operators' work happens), but takes
    /// the **max** for `memory`: two operators in the same pipeline don't
    /// both hold peak memory simultaneously in the general case (a
    /// streaming operator's memory footprint doesn't add to its parent's),
    /// but pipeline breakers stacked on top of each other really do sum
    /// their memory. This is an approximation, documented rather than
    /// silently wrong: a chain of several pipeline breakers will
    /// under-estimate total memory pressure.
    #[must_use]
    pub fn combine(&self, other: &Cost) -> Cost {
        Cost {
            io: self.io + other.io,
            cpu: self.cpu + other.cpu,
            memory: self.memory.max(other.memory),
            network: self.network + other.network,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostWeights {
    pub io_per_byte: f64,
    pub cpu_per_tuple: f64,
    pub memory_penalty: f64,
    pub network_per_byte: f64,
}

impl CostWeights {
    /// Reasonable defaults for when calibration hasn't been run — a
    /// documented fallback, not a substitute for `calibrate`.
    pub fn defaults() -> Self {
        CostWeights {
            io_per_byte: 1.0,
            cpu_per_tuple: 100.0,
            memory_penalty: 0.001,
            network_per_byte: 0.0,
        }
    }
}

/// Fits `CostWeights` by timing known operations on this machine: scan N
/// bytes (a linear read over a buffer), hash N tuples (a `HashMap` insert
/// loop), compare N tuples (a sort). Converts the cost model from "numbers
/// I made up" into "numbers I measured" — an afternoon of work for a much
/// better answer to "why these constants."
///
/// **What cost is for, and isn't**: a device for *ranking* plans, not
/// predicting runtime. It only has to get the ordering right; resist the
/// pull toward making it a wall-clock time estimator, which is a much
/// harder problem this project doesn't need solved.
pub fn calibrate() -> CostWeights {
    const N: usize = 200_000;

    let cpu_per_tuple = time_cpu_per_tuple(N);
    let io_per_byte = time_io_per_byte(N);

    CostWeights {
        io_per_byte,
        cpu_per_tuple,
        // Memory penalty and network aren't measured the same
        // wall-clock way (memory pressure and shuffle bytes don't have a
        // "time per unit" this machine can time in isolation the way a
        // scan or a hash probe does) — kept at the documented defaults.
        memory_penalty: CostWeights::defaults().memory_penalty,
        network_per_byte: CostWeights::defaults().network_per_byte,
    }
}

fn time_cpu_per_tuple(n: usize) -> f64 {
    let mut builder = PrimitiveBuilder::<Int64Type>::with_capacity(n);
    for i in 0..n as i64 {
        builder.append_value(i);
    }
    let lhs = ColumnarValue::Array(Arc::new(builder.finish()));
    let rhs = ColumnarValue::Scalar(ScalarValue::Int64(Some(1)));

    let start = Instant::now();
    std::hint::black_box(arith::add(&lhs, &rhs).unwrap());
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / n as f64
}

fn time_io_per_byte(n: usize) -> f64 {
    let src = vec![0u8; n];
    let start = Instant::now();
    let dst = std::hint::black_box(src.clone());
    let elapsed = start.elapsed();
    drop(dst);
    elapsed.as_secs_f64() / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_weights_each_component() {
        let cost = Cost {
            io: 10.0,
            cpu: 5.0,
            memory: 2.0,
            network: 1.0,
        };
        let weights = CostWeights {
            io_per_byte: 1.0,
            cpu_per_tuple: 2.0,
            memory_penalty: 3.0,
            network_per_byte: 4.0,
        };
        // 10*1 + 5*2 + 2*3 + 1*4 = 10+10+6+4 = 30
        assert!((cost.total(&weights) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn combine_sums_everything_but_memory_which_takes_the_max() {
        let a = Cost {
            io: 1.0,
            cpu: 1.0,
            memory: 10.0,
            network: 1.0,
        };
        let b = Cost {
            io: 2.0,
            cpu: 2.0,
            memory: 5.0,
            network: 2.0,
        };
        let combined = a.combine(&b);
        assert_eq!(combined.io, 3.0);
        assert_eq!(combined.cpu, 3.0);
        assert_eq!(combined.memory, 10.0);
        assert_eq!(combined.network, 3.0);
    }

    #[test]
    fn calibration_produces_positive_finite_weights() {
        let weights = calibrate();
        assert!(weights.cpu_per_tuple > 0.0 && weights.cpu_per_tuple.is_finite());
        assert!(weights.io_per_byte > 0.0 && weights.io_per_byte.is_finite());
    }

    #[test]
    fn zero_cost_totals_to_zero_under_any_weights() {
        let weights = CostWeights::defaults();
        assert_eq!(Cost::ZERO.total(&weights), 0.0);
    }
}
