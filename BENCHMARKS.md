# Benchmarks

Measured, not claimed. Run via `cargo bench --bench row_vs_batch`.

## col + 1: row-at-a-time (Phase 1) vs batch-at-a-time (Phase 2)

Same computation (`Int64` column + literal `1`), same data, two execution models:

- `phase1_row_at_a_time`: `expr::eval::eval`, called once per row via a match on the boxed `Expr` tree.
- `phase2_batch_at_a_time`: `compute::arith::add`, one call over the whole `Int64Array`.

| N         | Phase 1 (row-at-a-time) | Phase 2 (batch-at-a-time) | Speedup |
|-----------|--------------------------|----------------------------|---------|
| 1,000     | 67.9 µs                  | 26.8 µs                    | 2.53x   |
| 100,000   | 7.29 ms                  | 2.72 ms                    | 2.68x   |
| 1,000,000 | 64.5 ms                  | 48.1 ms                    | 1.34x   |

(Median of 100 samples per point, criterion 0.8.2, release profile with `lto = true`,
`codegen-units = 1`. Machine-local numbers — treat as directional, not absolute.)

## Reading this

The row-at-a-time interpreter pays a per-row cost that's roughly constant regardless of N:
tree-walk the `Expr`, match on `BinaryOp`, allocate/return an owned `Value`. The batch kernel
pays that dispatch cost once per column instead of once per row, so at 1K–100K rows the win is
a fairly steady ~2.5-2.7x.

At 1M rows the gap narrows to 1.34x. Both paths are now large enough to be bandwidth-bound on a
single `Int64` column (8 MB of data touched twice — read + write), so the fixed per-row dispatch
overhead that dominates at smaller N stops being the bottleneck. This is expected: the
architectural win of columnar batching is dispatch amortization and (with real kernels, not
shown here) SIMD-friendly access patterns — not a constant-factor speedup independent of scale.

## Caveats

- Single kernel (`add`), single type (`Int64`), no nulls. Not representative of aggregation,
  joins, or string/variable-width kernels, which are not benchmarked here.
- No SIMD-specific kernels have been written yet in Phase 2 — `compute::arith::add` is a plain
  loop over the buffer. The batching win measured above comes from dispatch amortization, not
  vectorization; a follow-up with explicit SIMD kernels would need its own benchmark.
- Single machine, single run per point. Not a substitute for a proper regression-tracking suite.
