---
name: basalt-perf
description: Performance engineering discipline for the Basalt query engine — the measure-first loop, hot-path rules, Rust performance traps, benchmarking with criterion, and profiling with flamegraph. Use this skill whenever optimizing, benchmarking, or profiling Basalt code, whenever writing compute kernels or per-row/per-batch execution code, and whenever anyone claims something is "faster" or "more efficient." Trigger it even for small optimizations — this project rejects any performance change that doesn't come with a measured number.
---

# Basalt performance engineering

Full detail in `docs/PERFORMANCE.md`. This is the operational summary.

## The prime directive

> **No optimization lands without a benchmark showing it helped.**

Not "this avoids an allocation." Not "this should be faster." A before/after number from
`cargo bench`. Programmer intuition about performance is reliably wrong, and an unmeasured
optimization is pure cost — harder to read, no proven benefit.

**Corollary: never optimize code you haven't profiled.** In a query engine the hotspot is rarely
where you'd guess — it's usually allocation, hashing, or a bounds check in a loop you didn't think
was hot.

## The loop

```
1. Write the clear version
2. Benchmark it            cargo bench
3. Profile it              cargo flamegraph --bench <name>
4. Change ONE thing        the widest frame in the flamegraph
5. Benchmark again         keep it only if the number moved
6. Comment why it's weird  ← not optional
```

Step 6 prevents a future reader (usually the author, months later) from "simplifying" the
optimization away.

## Optimize the right axis

Order of magnitude beats order of operations. Ranked by leverage in a columnar engine:

1. **Better plans** (join order, pushdown) — 10–1000×
2. **Read less data** (projection/predicate pushdown, partition pruning) — 10–100×
3. **Batch-at-a-time instead of row-at-a-time** — 10–50×
4. **SIMD / auto-vectorization on contiguous typed data** — 2–8×
5. **Zero-copy sharing (`Arc` + offsets) instead of copying** — varies, often huge
6. **Micro-optimizing a kernel** — 1.05–1.5×

A 5% kernel win is noise next to a join reordering that removes 99% of intermediate rows. Spend
effort where the exponents are.

## Hot-path rules

A hot path is per-row or per-value code — expression evaluation, kernels, sort comparators, hash
probes.

- **Zero allocation.** No `String`, `Vec`, `format!`, `to_string()`, or `collect()` inside the
  loop. Allocate once outside and reuse.
- **`Vec::with_capacity(n)`** — you almost always know `n` (row count, batch size).
- **Iterate, don't index.** `for x in slice` lets LLVM drop the bounds check; `slice[i]` often
  doesn't. Iterator chains inline and fuse into the same code as a hand-written loop.
- **Static dispatch.** Generics monomorphize and inline; `dyn` costs an indirection *and* blocks
  inlining. Use `dyn` for open sets (plan nodes), generics for closed hot sets (typed kernels).
- **Hoist invariants.** Type checks and schema lookups happen once at bind time, never per row.
- **Branch on the common case first**; mark error paths `#[cold]`. Check `null_count == 0` once
  and take a separate branch-free loop.

## Rust traps, by frequency

| Trap | Fix |
|---|---|
| `.clone()` on `String`/`Vec` in a loop | Borrow, or clone once outside |
| `format!` / `to_string()` in a hot path | Preformat, or reuse a buffer |
| Unnecessary `.collect()` mid-chain | Stay lazy until the final consumer |
| `Vec<Box<dyn Trait>>` in a hot loop | Enum dispatch or generics |
| Default `HashMap` hasher | `rustc-hash`/`ahash` for internal maps (never untrusted keys) |
| `String` hash keys | Intern to a `u32` id |
| `Vec<Option<T>>` for nullable data | Separate validity structure |
| Large structs by value | Pass `&T` |
| `#[inline]` everywhere | Only where measured; trust LLVM otherwise |
| **Benchmarking a debug build** | **Always `--release`** — debug is 10–100× slower |

That last one causes more false conclusions about Rust performance than everything else combined.

## Benchmarking

`criterion`, in `benches/`. Benchmarks must compile in CI.

- **Realistic shapes:** vary row count (1K / 100K / 10M), null density (0% / 10% / 90%),
  cardinality (few groups / many), and distribution (uniform / skewed). Skew is where engines
  fall over and synthetic benchmarks lie.
- **`black_box` inputs and outputs** so the optimizer can't delete the work being timed.
- **Record results in `BENCHMARKS.md`** with date and commit. The curve is the artifact.
- **Regression gate:** >5% regression on a tracked benchmark must be justified or reverted.

## Profiling

```bash
cargo flamegraph --bench <name>     # where is the time?
samply record ./target/release/...  # interactive
perf stat -d ./target/release/...   # cache misses, branch misses, IPC
cargo asm <path::to::fn>            # did it actually vectorize? look for xmm/ymm/zmm
```

**Read flamegraphs for width, not depth.** A wide box is time spent; a deep stack is just call
structure. The instinct to optimize the deepest frame is wrong.

"It should vectorize" is a hypothesis. Check the assembly.

## Memory

- Assert core type sizes in tests (`size_of::<Value>()`) so a new enum variant can't silently
  fatten every instance.
- Enums cost `max(variant) + tag` — `Box` the outlier variant.
- Struct-of-arrays over array-of-structs (that's what columnar *is*).
- Watch **peak memory**, not just throughput: an aggregation 20% faster on 3× the memory is worse,
  because it OOMs on the workload that matters.

## When NOT to optimize

- Anything outside the profile's top 10 frames
- Parsing and planning (microseconds against a multi-second query)
- Error paths
- Code whose readable version hasn't been measured yet
- Anything requiring `unsafe` before safe options are exhausted

**Phase 1 is deliberately slow** — row-at-a-time, eagerly materialized. Do not optimize it. Its
job is to be the baseline that makes Phase 2's vectorization speedup measurable. A documented 30×
is worth far more than an undocumented "it's fast."
