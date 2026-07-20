# Basalt — Performance Engineering

**Status:** normative. Performance claims without numbers are rejected.

---

## 0. The prime directive

> **No optimization lands without a benchmark showing it helped.**

Not "it should be faster." Not "this avoids an allocation." A number, before and after, from
`cargo bench`, in the PR description. This rule exists because programmer intuition about
performance is reliably wrong, and because an unmeasured optimization is pure cost: it makes the
code harder to read and buys nothing you can prove.

The corollary: **you may not optimize code you haven't profiled.** Find the hotspot first. In a
query engine the hotspot is almost never where you expect — it's usually allocation, hashing, or
a bounds check inside a loop you didn't think was hot.

**The loop:**

```
1. Write the clear version.
2. Benchmark it.           → cargo bench
3. Profile it.             → cargo flamegraph
4. Optimize the hotspot.   → one change at a time
5. Benchmark again.        → keep it only if the number moved
6. Document why the code is now weird.
```

Step 6 is not optional. Optimized code that doesn't explain itself gets "simplified" by a future
you and the regression ships silently.

---

## 1. The performance model you're building toward

Know what actually makes columnar engines fast, so you optimize the right axis:

| Lever | Why it matters | Phase |
|---|---|---|
| **Columnar layout** | Read only the columns the query touches; perfect cache locality within a column | 1 |
| **Batch-at-a-time** | Amortize per-row dispatch over ~8K rows; enables everything below | 2 |
| **SIMD / auto-vectorization** | 4–8 lanes per instruction on contiguous typed data | 2 |
| **Zero-copy sharing** | `Arc`'d buffers; slicing is offset arithmetic, not memcpy | 2 |
| **Predicate/projection pushdown** | Don't read what you'll discard — the biggest win, by far | 3 |
| **Good plans** | A better join order beats every micro-optimization combined | 3 |
| **Parallelism** | Linear scaling across cores on partitioned data | 4 |

**Order of magnitude matters more than order of operations.** A 5% kernel improvement is noise
next to a join reordering that eliminates 99% of intermediate rows. Spend effort where the
exponents are.

---

## 2. Hot path rules

A "hot path" is anything executing per row or per value. In Phase 1 that's expression evaluation
and the comparison inside sort. From Phase 2 it's every compute kernel.

**P1 — Zero allocation in hot loops.** No `String`, no `Vec`, no `format!`, no `to_string()`,
no `collect()` inside a per-row loop. Allocate once outside, reuse the buffer.

**P2 — Preallocate when the size is known.** `Vec::with_capacity(n)` — you almost always know `n`
(row count, batch size, column count). Repeated reallocation is log₂(n) memcpys you didn't need.

**P3 — Prefer iterators to indexing.** Not just style: `for x in slice` lets LLVM eliminate the
bounds check, `for i in 0..len { slice[i] }` often doesn't. Iterator chains compile to the same
code as hand-written loops (they inline and fuse) *and* they're safer. This is the rare case where
the idiomatic version is also the fastest.

**P4 — Static dispatch in hot paths.** Generics monomorphize and inline; `dyn Trait` costs a
vtable indirection and blocks inlining. Use `dyn` for open, extensible sets (plan nodes); use
generics for closed, hot sets (type-specialized kernels). The per-call cost is small — the real
loss is the inlining opportunity.

**P5 — Hoist invariants out of loops.** Type checks, schema lookups, null-count checks: do them
once before the loop, not per row. This is the *entire reason* the binder resolves types statically
(see `docs/CODING_STANDARDS.md` §0 and the LLD's bound/unbound split).

**P6 — Branch on the common case first,** and mark error paths `#[cold]`. Null handling is the
canonical example: check `null_count == 0` once and take an entirely separate, branch-free loop.

**P7 — Avoid `Option<T>` and enums in the innermost loop** where a null-free fast path is
available. Materializing a `Value` per row (Phase 1 does this deliberately) is exactly what
Phase 2 removes.

---

## 3. Rust-specific traps

Ranked by how often they bite in this kind of code:

| Trap | Cost | Fix |
|---|---|---|
| `.clone()` on a `String`/`Vec` in a loop | Allocation + memcpy per iteration | Borrow, or clone once outside |
| `format!` / `to_string()` in a hot path | Allocation per call | Preformat, or write into a reused buffer |
| Unnecessary `.collect()` mid-chain | Materializes the whole intermediate | Keep the iterator lazy to the end |
| `Vec<Box<dyn Trait>>` in a hot loop | Vtable indirection + pointer chase per element | Enum dispatch or generics |
| `HashMap` with the default hasher | SipHash is DoS-resistant but slow | `rustc-hash`/`ahash` for internal maps (never for untrusted keys) |
| `String` keys in a hash map | Hash + compare walks the bytes | Intern to a `u32` id |
| Indexing instead of iterating | Bounds check per access | Iterators (P3) |
| `Vec<Option<T>>` for nullable data | 16 bytes for an `i64`, no niche | Separate validity (see the LLD §2.3) |
| Passing large structs by value | memcpy per call | Pass `&T`, or `Box` the struct |
| `#[inline]` everywhere | Binary bloat, worse i-cache | Only on small cross-crate functions |
| Debug builds for benchmarking | 10–100× slower, meaningless numbers | **Always `--release`** |

That last one is not a joke. Benchmarking a debug build is the single most common way people reach
false conclusions about Rust performance.

---

## 4. `#[inline]` policy

Do **not** sprinkle it. LLVM inlines aggressively within a crate already.

- `#[inline]` — small functions called across crate boundaries where you've measured a win
- `#[inline(always)]` — only with a benchmark proving it, and a comment citing that benchmark
- `#[cold]` — error paths and unlikely branches, so they're laid out away from the hot code
- Default — nothing. Trust the optimizer until it's proven wrong.

---

## 5. Benchmarking

**Framework:** `criterion` (statistically rigorous, detects regressions, produces plots).

**B1 — Every hot-path module has benchmarks.** Kernels, expression eval, sort, hash aggregation,
join, the parser.

**B2 — Benchmark realistic shapes,** not one row. Vary: row count (1K / 100K / 10M), null density
(0% / 10% / 90%), cardinality (few groups / many groups), and data distribution (uniform / skewed).
Skew is where real systems fall over and synthetic benchmarks lie.

**B3 — `black_box` your inputs and outputs** so the optimizer can't delete the work you're timing.

**B4 — Benchmarks live in `benches/` and compile in CI.** A benchmark that doesn't compile is a
benchmark that isn't run.

**B5 — Track results over time in `BENCHMARKS.md`** with dates and commit hashes. The curve is
the artifact — it's what you show people, and it's what catches slow regressions that no single
PR would fail.

**B6 — Regression gate:** a PR that regresses any tracked benchmark by >5% must either justify it
in the description or not merge.

---

## 6. Profiling toolchain

```bash
cargo install flamegraph samply cargo-criterion

cargo flamegraph --bench my_bench      # where is time going?
samply record ./target/release/basalt  # interactive alternative
perf stat -d ./target/release/basalt   # cache misses, branch mispredicts, IPC
cargo bench                            # criterion, with regression detection
cargo asm basalt::kernel::add_i64      # did it actually vectorize?
```

**Read flamegraphs top-down for *width*, not depth.** A wide box is where the time is; a deep
stack is just call structure. The instinct to optimize the deepest frame is wrong.

**When you think you have a vectorization win, verify it** — `cargo asm` and look for the SIMD
registers (`xmm`/`ymm`/`zmm`). "It should vectorize" is a hypothesis, not a result.

---

## 7. Memory

**M1 — Know your struct sizes.** `std::mem::size_of::<T>()` in a test, asserted, for every core
type. A `Value` that silently grows from 24 to 40 bytes because someone added a variant is a
real regression, and an assertion catches it at CI time:

```rust
#[test]
fn value_size_is_bounded() {
    assert_eq!(std::mem::size_of::<Value>(), 32);
}
```

**M2 — Enums cost `max(variants) + tag`.** One fat variant makes every instance fat. `Box` the
outlier.

**M3 — Prefer SoA to AoS.** Struct-of-arrays *is* the columnar model — you're already doing this
at the table level. Apply the same instinct inside data structures.

**M4 — From Phase 2, share don't copy.** `Arc<Buffer>` + offset/length. Slicing a million-row
column must be O(1). If a "slice" allocates, the design is wrong.

**M5 — Watch peak memory, not just throughput.** A hash aggregation that's 20% faster but uses 3×
the memory is a worse operator — it OOMs on the workload that matters.

---

## 8. When *not* to optimize

- Anything outside the profile's top 10 frames
- Parsing and planning (microseconds against a query that runs for seconds)
- Error paths
- Anything where the readable version hasn't been measured yet
- Anything that would require `unsafe` before you've exhausted safe options

**Explicitly permitted slowness:** Phase 1 is row-at-a-time and eagerly materializing, on purpose.
Don't optimize it. Its job is to be the baseline that makes Phase 2's speedup measurable — and a
measured 30× is a far better story than an unmeasured "it's fast."
