# Benchmarks

Measured, not claimed. Run via `cargo bench --bench row_vs_batch`.
Diagnostic harness for the investigation below: `cargo run --release --example profile_add`.

## col + 1: row-at-a-time (Phase 1) vs batch-at-a-time (Phase 2)

Same computation (`Int64` column + literal `1`), same data, two execution models:

- `phase1_row_at_a_time`: `expr::eval::eval`, called once per row via a match on the boxed `Expr` tree.
- `phase2_batch_at_a_time`: `compute::arith::add`, one call over the whole `Int64Array`.

| N         | Phase 1 (row-at-a-time) | Phase 2 (batch-at-a-time) | Speedup |
|-----------|--------------------------|----------------------------|---------|
| 1,000     | 63.7 µs                  | 9.89 µs                    | 6.44x   |
| 100,000   | 6.31 ms                  | 1.07 ms                    | 5.88x   |
| 1,000,000 | 63.3 ms                  | 21.4 ms                    | 2.95x   |

(Median of 100 samples per point, criterion 0.8.2, release profile with `lto = true`,
`codegen-units = 1`. Machine-local numbers — treat as directional, not absolute.)

## Two rounds of "the kernel itself was the bottleneck"

**Round 1 — scalar materialization.** The first cut of `compute::arith::add` materialized a
scalar operand into a full array via `ColumnarValue::into_array` before doing the elementwise
op — `col + 1` first built an entire `Int64Array` of `1`s (one `append_value` call per row)
before adding it to `col`. A second N-element allocation and fill pass hiding inside what was
supposed to be a single-pass kernel. Fixed by giving array⊕scalar / scalar⊕array a dedicated
path (`array_scalar_numeric`) that reads the scalar once and applies it directly — no
intermediate array ever built. Speedup at 1M went from 1.34x to 2.39x; 1K/100K improved to
4.47x/4.17x.

**Round 2 — re-deriving the slice per element.** Still not at the ~10x a query-engine
microbenchmark like this should show. Wrote `examples/profile_add.rs` to isolate the floor: a
bare `for &v in &data { checked_add }` loop over a plain `&[i64]` landed at 3.85 ns/element —
call that the checked-arithmetic floor. The actual kernel's inner loop called `a.value(i)` per
element, and `value()` calls `values()`, which re-derives its slice from the `Arc`-backed
`Buffer` on every single call (`Buffer::as_slice` → `Arc` deref → `AlignedBytes::as_slice` →
offset slicing, from scratch, every element). Measured cost of that indirection alone: full
kernel dropped from 16.89 ns/element to 11.44 ns/element (a ~32% cut) just from hoisting
`let values = a.values();` once before the loop and indexing `values[i]` instead of calling
`a.value(i)` inside it. Applied to both the array⊕scalar and array⊕array paths.

Isolation numbers from `profile_add` (N = 1,000,000, release, before round 2 landed):

| Variant                                | ns/element |
|-----------------------------------------|-----------:|
| bare `checked_add` loop over `&[i64]`   | 3.85       |
| bare `wrapping_add` loop                | 4.00       |
| `iter().map().collect()` (autovectorized)| 1.42      |
| `PrimitiveBuilder` + per-element checked_add | 6.51   |
| `PrimitiveBuilder::append_slice` (bulk, precomputed) | 7.82 |
| full kernel, before hoisting `values()` out of the loop | 16.89 |
| full kernel, after (`compute::arith::add` today)       | 11.44 |

The `append_slice` row being *slower* than the fused per-element builder loop is itself a real
finding: `append_slice` does values in one bulk `extend_from_slice`, then a *separate* pass
writing the validity bitmap one bit at a time. Two passes over 8 MB beats one fused pass on
paper but loses in practice — consistent with the cache-effects explanation for the earlier
1M-row degradation, not "bandwidth-bound" in the naive sense (16 MB of real traffic in
single-digit milliseconds is nowhere near the 20-50 GB/s DRAM ceiling either way).

## Reading this

The row-at-a-time interpreter pays a per-row cost that's roughly constant regardless of N:
tree-walk the `Expr`, match on `BinaryOp`, allocate/return an owned `Value`. The batch kernel
pays that dispatch cost once per column instead of once per row, so the speedup is largest at
smaller N (5.9-6.4x at 1K-100K) and narrows at 1M (2.95x) as both paths start paying real
per-element costs that don't amortize away: `checked_add`'s overflow branch per element (a
deliberate correctness choice — see `compute::arith`'s module doc comment — that also blocks
auto-vectorization) and cache effects on 8 MB of live data.

## Remaining known gap, not yet chased

`PrimitiveBuilder::append_value` still writes one validity bit per element via `BitmapBuilder`
even when the array turns out to have zero nulls (the bitmap is only *dropped* at `finish()`,
not skipped during construction). A "does this batch have any nulls at all" fast path that
skips bitmap writes entirely in the common no-null case is a real, identified next lever — not
implemented here because it changes `PrimitiveBuilder`'s API/internals rather than this one
kernel, and deserves its own benchmark before/after rather than being folded into this one.

## Caveats

- Single kernel (`add`), single type (`Int64`), no nulls in the hot path. Not representative of
  aggregation, joins, or string/variable-width kernels, which are not benchmarked here.
- The kernel uses checked arithmetic (traps on overflow, per this project's "no silently wrong
  answers" policy) rather than wrapping. That per-element branch is real and not free (~3.85 ns
  vs ~1.4 ns for an autovectorized wrapping add) — a wrapping/unchecked kernel would likely be
  faster still but isn't what a query engine should ship for `SUM`-adjacent code paths.
- No SIMD-specific kernels exist in Phase 2 yet. The gains measured here come from eliminating
  redundant per-call work (dispatch amortization, not double-materializing the scalar, not
  re-deriving slices per element), not from vectorization; a follow-up with explicit SIMD
  kernels would need its own benchmark.
- Single machine, single run per point. Not a substitute for a proper regression-tracking suite.
