# Benchmarks

Measured, not claimed. Run via `cargo bench --bench row_vs_batch`.

## col + 1: row-at-a-time (Phase 1) vs batch-at-a-time (Phase 2)

Same computation (`Int64` column + literal `1`), same data, two execution models:

- `phase1_row_at_a_time`: `expr::eval::eval`, called once per row via a match on the boxed `Expr` tree.
- `phase2_batch_at_a_time`: `compute::arith::add`, one call over the whole `Int64Array`.

| N         | Phase 1 (row-at-a-time) | Phase 2 (batch-at-a-time) | Speedup |
|-----------|--------------------------|----------------------------|---------|
| 1,000     | 63.8 µs                  | 14.3 µs                    | 4.47x   |
| 100,000   | 6.27 ms                  | 1.50 ms                    | 4.17x   |
| 1,000,000 | 62.6 ms                  | 26.2 ms                    | 2.39x   |

(Median of 100 samples per point, criterion 0.8.2, release profile with `lto = true`,
`codegen-units = 1`. Machine-local numbers — treat as directional, not absolute.)

## A first version of this benchmark was wrong, and here's the fix

The first cut of `compute::arith::add` materialized a scalar operand into a full array via
`ColumnarValue::into_array` before doing the elementwise op — so `col + 1` first built an
entire `Int64Array` of `1`s (one `append_value` call per row) before adding it to `col`. That's
a second N-element allocation and fill pass hiding inside what was supposed to be a single-pass
kernel, and it dominated the timing: the earlier numbers showed a ~27 ns/element floor that
should have been close to 1 ns, plus a 1.8x cliff at 1M rows from the extra 8 MB allocation.
16 MB of real traffic (8 in + 8 out) in 48 ms is 333 MB/s — two orders of magnitude below DRAM
bandwidth, so "bandwidth-bound" was the wrong explanation; the actual bottleneck was unnecessary
work, not memory traffic.

The fix: `compute::arith::arith.rs` now has a dedicated array⊕scalar / scalar⊕array path
(`array_scalar_numeric`) that reads the scalar once and applies it against the array's values
directly, with no intermediate array ever built. Array⊕array is unaffected — it was already a
single pass. Speedups after the fix: 4.47x / 4.17x / 2.39x at 1K/100K/1M, up from 2.53x / 2.68x
/ 1.34x before it, and the earlier 1M-only degradation this write-up called out is gone.

## Reading this

The row-at-a-time interpreter pays a per-row cost that's roughly constant regardless of N:
tree-walk the `Expr`, match on `BinaryOp`, allocate/return an owned `Value`. The batch kernel
pays that dispatch cost once per column instead of once per row, so the speedup is largest at
smaller N (4.1-4.5x at 1K-100K) and narrows somewhat at 1M (2.39x) as both paths start paying
real per-element costs that don't amortize away: `checked_add`'s overflow branch per element
(a deliberate correctness choice — see this module's doc comment — that also blocks
auto-vectorization) and cache effects on 8 MB of live data.

## Caveats

- Single kernel (`add`), single type (`Int64`), no nulls in the hot path. Not representative of
  aggregation, joins, or string/variable-width kernels, which are not benchmarked here.
- The kernel uses checked arithmetic (traps on overflow, per this project's "no silently wrong
  answers" policy — see `compute::arith`'s doc comment) rather than wrapping. That per-element
  branch is real and not free; a wrapping/unchecked kernel would very likely be faster still but
  isn't what a query engine should ship for `SUM`-adjacent code paths.
- No SIMD-specific kernels exist in Phase 2 yet. The gains measured here come from eliminating
  redundant per-call work (dispatch amortization + not double-materializing), not from
  vectorization; a follow-up with explicit SIMD kernels would need its own benchmark.
- Single machine, single run per point. Not a substitute for a proper regression-tracking suite.
