# Benchmarks

Measured, not claimed. Run via `cargo bench --bench row_vs_batch`.
Diagnostic harnesses: `cargo run --release --example profile_add`,
`cargo bench --bench kernel_only` (a minimal, single-benchmark criterion file
used to isolate harness-level effects from the code itself — see "Round 4").

## TPC-H Q6-shaped end-to-end query: filter + project

`cargo bench --bench tpch`. Per design-docs/basalt-phase2-lld.md §11 ("the same
TPC-H query, Phase 1 vs Phase 2, on the same data"): Phase 1's `exec/` has no
aggregation at all (no GROUP BY, no SUM/COUNT/AVG), so this runs the largest
fair subset of TPC-H Q6 both engines can actually execute — the multi-predicate
`WHERE` (a `l_shipdate` range, a `l_discount` range, a `l_quantity` bound,
`AND`-ed together) and the `l_extendedprice * l_discount` projection, with the
`SUM` aggregate dropped. The same `Expr` tree is built once and fed to both
engines (`LogicalPlan` is built directly on Phase 1's own `expr::expr::Expr` —
there's no separate Phase 2 expression type), so this measures execution, not
two independently-hand-written predicates that happen to look similar. Plan
construction (`LogicalPlanBuilder` + `PhysicalPlanner::create_physical_plan`)
runs once, outside the timed loop, matching how `predicate`/`projection` are
also built once for Phase 1's loop — timing planning here would compare
Phase 2's planner against Phase 1's bare interpreter, not the two execution
models against each other (an earlier draft of this benchmark made exactly
that mistake; see below).

| N         | Phase 1 (row-at-a-time) | Phase 2 (batch-at-a-time) | Speedup |
|-----------|--------------------------|----------------------------|---------|
| 1,000     | 199 µs                    | 120 µs                     | 1.66x   |
| 100,000   | 20.7 ms                  | 11.4 ms                    | 1.81x   |
| 1,000,000 | 226 ms                   | 139 ms                     | 1.62x   |

This is a much smaller multiple than `col + 1`'s 17-24x, and that's expected
and honest: a five-predicate filter plus a multiply has real per-row cost in
*both* engines (five comparisons, three `AND`s, a gather, a projection), so
there's much less fixed per-row dispatch overhead for batching to amortize
away relative to the actual work — unlike a single `+ 1`, where Phase 1's
interpreter overhead was nearly the entire cost. 1.6-1.8x on a genuinely
multi-predicate query is a more representative number for what to expect from
this project's current state than the single-kernel microbenchmark is.

**The first version of this benchmark showed Phase 2 *losing*** — 0.83x/0.87x/1.08x,
i.e., slower than Phase 1's row-at-a-time interpreter at every N. Two real bugs,
found by taking that result seriously instead of dismissing it:

1. **Plan construction was inside the timed closure for Phase 2 but not for
   Phase 1.** Phase 1's loop only re-evaluates `predicate`/`projection`
   (already-built `Expr` trees) per iteration; Phase 2's loop was rebuilding
   the entire `LogicalPlanBuilder`/`PhysicalPlanner::create_physical_plan`
   chain every iteration — comparing query planning against a bare
   interpreter loop, not execution against execution. Fixed by building the
   physical plan once, outside `b.iter()` (`ExecutionPlan::execute` takes
   `&self` and is safe to call repeatedly).
2. **`compute::comparison.rs` had the exact same two bugs `compute::arith`
   had before its own fast path**: every scalar comparison (`l_shipdate >=
   8400`, `l_discount <= 0.07`, etc. — this query has four of them)
   materialized the scalar into a full array via `into_array`, and
   `value(i)` re-derived its buffer slice on every element. Since Q6-style
   queries are comparison-heavy, this mattered far more here than in the
   `add`-only microbenchmark. Fixed with the same dedicated
   `Int64`/`Float64` array⊕scalar fast path `arith.rs` uses.

   Chasing that fix down further surfaced the **same re-derivation bug
   throughout the bit-packed `Bitmap`/`BooleanArray` path**: `Bitmap::get`
   calls `Buffer::as_slice()` (an `Arc` deref plus two nested offset/len
   computations) on every single bit read, and `filter.rs`'s predicate scan,
   `boolean.rs`'s `AND`/`OR`/`NOT` (this query ANDs three comparisons
   together), `take.rs`'s index and source-array gathers, and
   `sort.rs`'s per-comparison-call downcast-and-re-derive in its sort
   comparator (called `O(n log n)` times) all had it. Added
   `Bitmap::as_bytes()`/`bit_offset()` plus a free `buffer::bit_at` helper so
   each of those hot loops hoists the bitmap slice once instead of
   re-deriving it per bit; `sort.rs`'s comparator additionally now downcasts
   each column once before sorting instead of on every one of the `O(n log n)`
   comparison calls. `Utf8`'s own internal buffer re-derivation
   (`StringArray::value`) was left as-is — same class of bug, not fixed here,
   a real follow-up.

After all of the above: 0.83x/0.87x/1.08x → 1.66x/1.81x/1.62x.

## col + 1: row-at-a-time (Phase 1) vs batch-at-a-time (Phase 2)

Same computation (`Int64` column + literal `1`), same data, two execution models:

- `phase1_row_at_a_time`: `expr::eval::eval`, called once per row via a match on the boxed `Expr` tree.
- `phase2_batch_at_a_time`: `compute::arith::add`, one call over the whole `Int64Array`.

| N         | Phase 1 (row-at-a-time) | Phase 2 (batch-at-a-time) | Speedup |
|-----------|--------------------------|----------------------------|---------|
| 1,000     | 67.3 µs                  | 2.75 µs                    | 24.4x   |
| 100,000   | 7.12 ms                  | 357 µs                     | 20.0x   |
| 1,000,000 | 66.1 ms                  | 3.77 ms                    | 17.5x   |

(Median of 100 samples per point, criterion 0.8.2, release profile with `lto = true`,
`codegen-units = 1`. Machine-local numbers — treat as directional, not absolute.)

## Four rounds of "the kernel itself was the bottleneck"

**Round 1 — scalar materialization.** `compute::arith::add` originally materialized a scalar
operand into a full array via `ColumnarValue::into_array` before the elementwise op — `col + 1`
built an entire `Int64Array` of `1`s first. Fixed with a dedicated array⊕scalar / scalar⊕array
path that reads the scalar once.

**Round 2 — re-deriving the slice per element.** The inner loop called `a.value(i)` per element;
`value()` re-derives its slice from the `Arc`-backed `Buffer` from scratch every call. Fixed by
hoisting `let values = a.values();` once before the loop.

**Round 3 — the per-element `checked_add` branch, and an unconditional validity bitmap.**
`checked_add` compiles to a real per-element branch, and `PrimitiveBuilder` wrote one validity
bit per element via `BitmapBuilder::push` even when the array turned out to have zero nulls
(the bitmap was only *dropped* at `finish()`, never skipped during construction). Fixed for
`add`/`sub` with a dedicated branch-free path (`fast_int64_array_array` /
`fast_int64_array_scalar` in `compute::arith`) taken whenever every operand is `Int64` with no
nulls: signed-overflow detection for `+`/`-` has a well-known bitwise form
(`overflow = ((a ^ r) & (b ^ r)) < 0` for add, computed against `r = a.wrapping_add(b)`; similarly
for sub), so the loop ORs the overflow flag into one accumulator and checks it once, after the
loop, instead of branching per element. Result is written with `validity: None` directly,
skipping `BitmapBuilder` entirely. `mul`/`div`/`rem` don't have as cheap a branch-free form and
still use the general per-element path.

**Round 4 — two benchmark harnesses disagreeing by 2.8x forced a real bug hunt, not a
rationalization.** After round 3, `examples/profile_add.rs` reported 5.34 ns/element for the full
kernel at N=1,000,000, but the criterion bench (`cargo bench`) reported 14.8 ms — 14.8 ns/element.
Same kernel, same N, 2.8x apart. Chased in order:

1. *Is the fast path even being taken?* Added a test constructing an array the same way the
   benchmarks do and asserting `null_count() == 0` and `validity().is_none()` after `finish()`.
   Confirmed — both harnesses build arrays identically and both take the fast path. Ruled out.
2. *Sustained load / thermal throttling?* `profile_add` ran short (~50 iterations, ~270 ms
   total); criterion sustains load for 5-9 seconds per size. Extended `profile_add` to run
   400 iterations after a 3-second warmup, matching criterion's discipline. Numbers stayed at
   ~4-5.6 ns/element — did not reproduce the 14.8 ns figure. Ruled out.
3. *Cross-contamination from other benchmarks sharing the process (icache/branch-predictor
   pollution from Phase 1's interpreter code)?* Ran the criterion bench filtered to only the
   N=1,000,000 batch case — no Phase 1 code executes at all. Still 15.7 ns/element. Ruled out.
4. *Is criterion itself lying?* Wrote `benches/kernel_only.rs`, a minimal criterion file with
   nothing but this one kernel call — no Phase 1, no benchmark group. Reproduced 15.5 ns/element.
   Then wrote `examples/kernel_only_manual.rs`: the *exact* same array construction and a
   `Instant`-based timer with the exact same 3-second-warmup-then-400-iterations shape as
   criterion — as a **standalone binary**, no criterion at all. That reproduced 15.70 ns/element,
   matching criterion almost exactly, while `profile_add.rs` — running the identical operation —
   kept reporting ~5.3-5.6 ns/element in the same process.

   The real difference wasn't the timing tool; it was allocator state. `examples/profile_add.rs`
   runs several other large-allocation benchmarks (steps 1-5) *before* measuring the full kernel.
   glibc's malloc dynamically raises its `mmap` threshold based on the sizes of chunks it has
   recently freed, so by the time `profile_add` reaches the full-kernel measurement, an 8 MB
   allocation is served from the heap arena (already-resident pages, no OS involvement). A truly
   fresh process — `kernel_only_manual`, `kernel_only.rs`, and criterion's own bench process —
   pays a real `mmap`/page-fault cost for that same allocation every time. Confirmed directly:
   running `kernel_only_manual` with `MALLOC_MMAP_THRESHOLD_` set above 8 MB dropped its number
   from 15.70 to 5.55 ns/element — matching `profile_add`'s number exactly. **`profile_add`'s
   numbers were real, but artificially fast due to incidental allocator warm-up from its own
   earlier, unrelated steps — not because the kernel is actually 3x faster than criterion
   measured.** Criterion's number was the honest one the whole time; this reconciles the two
   harnesses without discarding either.

   This pointed at a second, *actually fixable* problem: why does an 8 MB allocation cost enough
   to matter at all? `MutableBuffer::freeze()` called `Buffer::from_vec`, which allocates a
   *second* buffer (`AlignedBytes::new`, itself a fresh `vec![0u8; len + 64]`) and `copy_from_slice`s
   into it purely to guarantee 64-byte alignment — a second full-size allocation and memcpy on
   top of the one that already held the computed result. Real traffic per kernel call was closer
   to 32 MB than 16 MB, plus two separate large-allocation page-fault sequences instead of one.
   Fixed by making `MutableBuffer` allocate its padding up front (mirroring `AlignedBytes`'s own
   layout) so `freeze()` can hand the same `Vec<u8>` to `Buffer` as a move in the common case —
   falling back to the old copying path only if the `Vec` reallocated to a different alignment
   mid-construction (verified, tested: `frozen_buffer_is_aligned_even_when_capacity_is_underestimated`).

   Effect, measured with `cargo bench --bench kernel_only` (fresh process, no priming, N=1M):
   **15.55 → 4.35 ns/element, a 72% drop.** Re-running `profile_add` afterward also converged
   its number down to ~4.0-4.5 ns/element — the two harnesses now agree within ~10%, not 2.8x.

## Where the numbers stand now

| Stage                                             | ns/element @ N=1M (fresh-process, criterion) |
|-----------------------------------------------------|----------------------------------------:|
| Before round 1 (scalar materialized every call)      | ~48                                     |
| After rounds 1-3 (still had the `freeze` double-copy) | ~15                                     |
| After round 4 (`freeze` copy eliminated)              | **~4.4**                                |

The earlier version of this document claimed N=1M was "approaching a genuinely memory-bound
regime" to explain why the speedup shrank at scale (21.1x/13.2x/2.95x → narrowing badly at 1M).
That explanation was wrong, and this round is why: the shrinking wasn't physics, it was the
`freeze` copy — a fixed ~2x tax that mattered more in absolute terms at large N, mimicking what a
bandwidth ceiling would look like. With it fixed, the speedup no longer collapses at scale
(24.4x/20.0x/17.5x above) — which is itself evidence the earlier diagnosis was mistaken, not
just a nicer number.

## Caveats

- Single kernel (`add`), single type (`Int64`), no nulls in the fast path (arrays/scalars with
  nulls fall back to the original per-element `is_null`-checked loop, unchanged and still
  covered by tests). Not representative of aggregation, joins, or string/variable-width kernels.
- `mul`/`div`/`rem` were not given a branch-free fast path and are slower per-element than
  `add`/`sub` — not benchmarked here, a real, documented follow-up.
- No SIMD intrinsics were written, and no disassembly (`cargo asm` / objdump) was inspected to
  check whether the branch-free loop actually auto-vectorizes. `iter().map().collect()`'s
  ~1.4-1.6 ns/element floor suggests there's still headroom above the ~4.4 ns/element the real
  kernel achieves; that gap has not been chased further and no vectorization claim is made either
  way.
- The allocator-priming artifact in Round 4 is itself worth remembering: any future ad hoc
  timing harness in this repo that runs multiple large-allocation benchmarks in one process
  should be treated with suspicion until cross-checked against a fresh-process measurement
  (criterion runs each `cargo bench` invocation as its own process, which is part of why it's the
  source of truth here, not `examples/profile_add.rs`).
- Single machine, single run per point. Not a substitute for a proper regression-tracking suite.
