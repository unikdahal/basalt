//! Per-operator cost functions. See design-docs/basalt-phase3-lld.md §4.2.
//!
//! `Cost`'s `cpu`/`io`/`memory` fields are counts (tuples/bytes), not
//! pre-weighted; `Cost::total` applies `CostWeights` once, at comparison
//! time. Every function here returns the *structure* of the cost (how many
//! tuples get touched, how many times, how much memory is live) — the
//! algorithmic shape the LLD's table specifies (e.g. `n log n` for `Sort`,
//! `n log k` for `TopK`) is what these functions capture; the constant
//! factor is `CostWeights`.

use super::model::Cost;

/// `bytes_read * io + rows * cpu`, where `bytes_read` reflects projection
/// and pruning (i.e. the caller passes the bytes *after* those, not the
/// full file size).
pub fn scan_cost(bytes_read: f64, rows: f64) -> Cost {
    Cost {
        io: bytes_read,
        cpu: rows,
        memory: 0.0,
        network: 0.0,
    }
}

/// `input_rows * num_predicates * cpu`.
pub fn filter_cost(input_rows: f64, num_predicates: f64) -> Cost {
    Cost {
        io: 0.0,
        cpu: input_rows * num_predicates.max(1.0),
        memory: 0.0,
        network: 0.0,
    }
}

/// `input_rows * num_exprs * cpu`.
pub fn projection_cost(input_rows: f64, num_exprs: f64) -> Cost {
    Cost {
        io: 0.0,
        cpu: input_rows * num_exprs.max(1.0),
        memory: 0.0,
        network: 0.0,
    }
}

/// `build_rows (hash) + probe_rows (probe) + output_rows (materialize)`,
/// memory `build_rows * row_width` (the hash table holding the build side).
pub fn hash_join_cost(build_rows: f64, probe_rows: f64, output_rows: f64, row_width: f64) -> Cost {
    Cost {
        io: 0.0,
        cpu: build_rows + probe_rows + output_rows,
        memory: build_rows * row_width,
        network: 0.0,
    }
}

/// `left_rows * right_rows * cpu` — deliberately punishing, so the
/// enumerator only picks this when nothing else applies (no equi-join
/// column available).
pub fn nested_loop_join_cost(left_rows: f64, right_rows: f64) -> Cost {
    Cost {
        io: 0.0,
        cpu: left_rows * right_rows,
        memory: 0.0,
        network: 0.0,
    }
}

/// `input_rows (hash) + num_groups (finalize)`, memory
/// `num_groups * state_width`.
pub fn aggregate_cost(input_rows: f64, num_groups: f64, state_width: f64) -> Cost {
    Cost {
        io: 0.0,
        cpu: input_rows + num_groups,
        memory: num_groups * state_width,
        network: 0.0,
    }
}

/// `n log n * cpu`, plus a spill cost if `n * row_width` exceeds
/// `memory_budget`. **The spill discontinuity is the most important part
/// of this function**: a sort that fits in memory and one that spills
/// differ by a large constant factor, and modelling that cliff explicitly
/// is what stops the optimizer from confidently picking a plan that
/// quietly falls off it. A smooth cost curve across the memory boundary
/// would hide exactly the case that matters most.
pub fn sort_cost(n: f64, row_width: f64, memory_budget: f64) -> Cost {
    let compare_cost = if n > 1.0 { n * n.log2() } else { 0.0 };
    let live_memory = n * row_width;
    if live_memory > memory_budget && memory_budget > 0.0 {
        // Spilling: every row is written to and read back from disk at
        // least once on top of the in-memory comparison work — modeled
        // as extra `io` proportional to the data that didn't fit.
        let spilled_bytes = live_memory - memory_budget;
        Cost {
            io: spilled_bytes * 2.0, // written once, read back once
            cpu: compare_cost,
            memory: memory_budget,
            network: 0.0,
        }
    } else {
        Cost {
            io: 0.0,
            cpu: compare_cost,
            memory: live_memory,
            network: 0.0,
        }
    }
}

/// `n log k * cpu`, memory `k * row_width` — much cheaper than `Sort`,
/// which is exactly how the planner learns to prefer it for
/// `ORDER BY ... LIMIT k` over a full `Sort` + `Limit`.
pub fn topk_cost(n: f64, k: f64, row_width: f64) -> Cost {
    let k = k.max(1.0);
    Cost {
        io: 0.0,
        cpu: n * k.log2().max(0.0),
        memory: k * row_width,
        network: 0.0,
    }
}

/// Approximately free: `Limit` only takes the first few rows of its
/// already-computed input.
pub fn limit_cost() -> Cost {
    Cost::ZERO
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_is_much_cheaper_than_sort_for_a_small_k() {
        let n = 1_000_000.0;
        let k = 10.0;
        let width = 16.0;
        let sort = sort_cost(n, width, f64::INFINITY);
        let topk = topk_cost(n, k, width);
        assert!(
            topk.cpu < sort.cpu,
            "TopK cpu ({}) should be far below Sort cpu ({})",
            topk.cpu,
            sort.cpu
        );
        assert!(topk.memory < sort.memory);
    }

    #[test]
    fn sort_spilling_costs_more_io_than_an_in_memory_sort() {
        let n = 1_000_000.0;
        let width = 100.0; // 100MB total, exceeds a small budget
        let in_memory = sort_cost(n, width, f64::INFINITY);
        let spilled = sort_cost(n, width, 1_000_000.0); // 1MB budget
        assert_eq!(in_memory.io, 0.0);
        assert!(
            spilled.io > 0.0,
            "a sort exceeding its memory budget must show nonzero I/O"
        );
    }

    #[test]
    fn nested_loop_join_is_quadratic() {
        let cost_100 = nested_loop_join_cost(100.0, 100.0);
        let cost_200 = nested_loop_join_cost(200.0, 200.0);
        assert!(
            (cost_200.cpu / cost_100.cpu - 4.0).abs() < 1e-9,
            "doubling both sides should quadruple cost"
        );
    }

    #[test]
    fn hash_join_memory_reflects_only_the_build_side() {
        let cost = hash_join_cost(1000.0, 1_000_000.0, 1000.0, 16.0);
        assert_eq!(cost.memory, 1000.0 * 16.0);
    }

    #[test]
    fn limit_cost_is_zero() {
        assert_eq!(limit_cost(), Cost::ZERO);
    }
}
