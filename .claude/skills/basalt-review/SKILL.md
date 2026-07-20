---
name: basalt-review
description: Structured code review for the Basalt query engine — correctness, null semantics, ownership, API design, performance, and test adequacy, in that order. Use this skill whenever reviewing a diff, a PR, or a file in the Basalt repository, whenever the user asks "does this look right", "review this", or "what's wrong with this code", and before merging any change. Trigger it proactively after writing a non-trivial chunk of Basalt code, since this project's review bar is higher than ordinary Rust review.
---

# Basalt code review

Review in this order. **Stop and report at the first correctness problem** — style feedback on
code that's wrong wastes everyone's time.

## 1. Correctness

- Does it do what the doc comment claims?
- Are the type's documented invariants preserved by every path through the change?
- **Error paths**: is every `Err` reachable, and does it carry actionable context?
- **Boundaries**: empty input, single element, zero rows, zero columns, all-null, no-null,
  `i64::MIN`/`MAX`, `f64::NAN`, empty string, non-ASCII.
- **Integer overflow**: debug panics, release wraps. Is that the intended behavior, or should this
  be `checked_*` / `saturating_*`?
- **Off-by-one** in slicing, offsets, and row ranges — the dominant bug class in columnar code.

## 2. Query-engine-specific bug classes

These are the ones that produce *silently wrong answers*, which is the worst outcome in a query
engine. Check them explicitly:

- **Null semantics.** `NULL = NULL` must be `NULL`, not `true`. `false AND NULL` is `false`.
  `true OR NULL` is `true`. `NOT NULL` is `NULL`. `WHERE` rejects both `false` and `NULL`.
- **Reading a null slot's data.** Null slots hold garbage values. Validity must be checked before
  the value is read, always.
- **Type coercion.** Is widening applied consistently on both operands? Is it materialized as an
  explicit `Cast` node rather than done implicitly at eval time?
- **Float comparison.** `f64` is `PartialOrd`, not `Ord`. Is there a documented `NaN` policy?
  Is `==` being used on computed floats (almost always a bug)?
- **Sort stability.** Multi-key ordering requires a stable sort.
- **Null ordering.** `ORDER BY` must place nulls per the documented policy, consistently.
- **Length invariants.** Do all columns in the batch still have equal length after the operation?
- **Schema/data agreement.** Does column *i*'s type still match field *i*'s declared type?

## 3. Hard rules (lint-enforced — flag any violation)

- No `unwrap()` / `expect()` / `panic!` / `todo!()` in library code
- No `v[i]` indexing in library code — `.get()` and handle `None`
- No `unsafe` without an ADR and a `// SAFETY:` comment
- Every `#[allow]` outside tests has a justification comment
- Public items documented, `# Errors` present on fallible functions

## 4. Ownership and borrowing

- **Every `.clone()`**: is it a cheap `Arc` bump, off the hot path with clarity gained, or
  explained by a comment? Unexplained clones are the most common review finding in this repo.
- Could a parameter be borrowed instead of owned?
- Are parameters `&str` / `&[T]` rather than `&String` / `&Vec<T>`?
- Is `Arc::clone(&x)` used rather than `x.clone()` for shared data?
- Any `RefCell`? It needs an ADR.

## 5. API design

- Do name prefixes match behavior — `try_*` fallible, `as_*` cheap, `to_*` allocates,
  `into_*` consumes?
- Are invariants enforced at construction rather than checked at each use?
- Are fields private with accessors?
- Is `#[must_use]` present on pure functions?
- Does the module layering hold (`types → array → batch → expr → plan → exec`, `sql → types`)?
  **Any upward dependency is a design bug, not an import to add.**

## 6. Performance (only after 1–5 pass)

- Is this on a hot path (per-row / per-value)? If not, prefer clarity and move on.
- If it is: allocation inside the loop? `with_capacity` where the size is known? Indexing where
  an iterator would let LLVM drop the bounds check? `dyn` dispatch that could be static?
- **Is there a benchmark number for any performance claim?** If the diff claims a speedup with no
  measurement, that's a blocking finding.
- Conversely: is this a speculative micro-optimization that hurts readability with no measurement
  behind it? That's also blocking, in the other direction.

## 7. Tests

- Happy path, error path, and boundary cases all present?
- Does each new invariant have a test that proves it fires?
- Are test names sentences describing behavior?
- Do the assertions check the **specific** error variant, not just that an error occurred?
- If null semantics are touched, are the three-valued-logic cases covered?
- Any nondeterminism — wall clock, unseeded randomness, `HashMap` iteration order?

## 8. Documentation

- Doc comment first line is one sentence, ending in a period
- `# Errors` on fallible functions; `# Panics` if it can panic
- Doctests compile and demonstrate real usage
- Comments explain **why**, not what
- Non-obvious algorithms named and cited (e.g. "Pratt parsing — precedence climbing")

## Reporting

Group findings by severity and be specific — quote the line, state the problem, propose the fix:

- **Blocking** — correctness, hard-rule violations, missing tests, unmeasured perf claims
- **Should fix** — API design, unexplained clones, missing docs
- **Consider** — style, naming, structure

If the change is clean, say so plainly and name the one thing done well. Manufactured nitpicks
train people to ignore review comments.

**Note on this repo:** Basalt is a learning project. When flagging something, explain the
underlying principle so the author can generalize it, rather than just prescribing the fix.
