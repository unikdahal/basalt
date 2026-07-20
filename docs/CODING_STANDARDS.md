# Basalt — Coding Standards

**Status:** normative. Violations block merge.

> **The premise of this document:** guidelines that aren't enforced by tooling are decoration.
> Almost every rule below is backed by a lint in `docs/TOOLING.md` that turns the violation into a
> compile error. The prose here exists to explain *why* — the compiler does the enforcing.

---

## 0. The priority order

**Correct → Clear → Fast.** In that order, with one caveat that matters for this project:

In a query engine, *fast is a feature*, not a nice-to-have. So the real rule is: **never trade
correctness for speed, never trade clarity for speed you haven't measured.** An optimization
without a benchmark number attached is a guess, and guesses get reverted (see `docs/PERFORMANCE.md`).

A wrong answer delivered quickly is worthless. A query engine that returns incorrect results is
not a fast query engine — it's a broken one.

---

## 1. Hard rules (non-negotiable, lint-enforced)

| # | Rule | Enforced by |
|---|---|---|
| H1 | **No `unwrap()` or `expect()` in library code.** Tests may use them freely. | `clippy::unwrap_used`, `clippy::expect_used` |
| H2 | **No `panic!`, `todo!`, `unimplemented!` in library code.** | `clippy::panic`, `clippy::todo` |
| H3 | **No slice/array indexing (`v[i]`) in library code.** Use `.get()` and handle `None`. | `clippy::indexing_slicing` |
| H4 | **No `unsafe` in Phase 1.** From Phase 2, `unsafe` requires an approved ADR and a `// SAFETY:` comment. | `unsafe_code = "forbid"` |
| H5 | **Every public item has a doc comment.** | `missing_docs` |
| H6 | **Every fallible public function documents its `# Errors`.** | `clippy::missing_errors_doc` |
| H7 | **Zero warnings.** CI runs `clippy -- -D warnings`. | CI |
| H8 | **Formatted.** `cargo fmt --check` passes. | CI |

**On H1–H3:** these look draconian and they are, deliberately. The point is that reaching for
`unwrap()` should require *typing an allow attribute*, which forces you to justify it in writing:

```rust
// The schema was validated in `try_new`; index is guaranteed in range.
#[allow(clippy::indexing_slicing)]
let field = &self.fields[index];
```

That comment is the deliverable. An `#[allow]` without a justification comment is a review reject.

**Internal invariants** use `debug_assert!`, which compiles out in release. Use it liberally to
document and check assumptions that *cannot* be violated by user input:

```rust
debug_assert_eq!(values.len(), validity.len(), "validity must match data length");
```

---

## 2. Naming

| Kind | Convention | Example |
|---|---|---|
| Types, traits, enum variants | `UpperCamelCase` | `RecordBatch`, `ColumnBuilder` |
| Functions, methods, variables, modules | `snake_case` | `null_count`, `try_new` |
| Constants, statics | `SCREAMING_SNAKE_CASE` | `DEFAULT_BATCH_SIZE` |
| Lifetimes | short, lowercase | `'a`, `'src` |
| Generic type params | single capital, or descriptive | `T`, `K`, `V` |

**Method name prefixes carry meaning. Use them consistently:**

| Prefix | Contract |
|---|---|
| `new` | Infallible construction |
| `try_new` | Fallible construction, returns `Result` |
| `with_*` | Construction with a parameter (`with_capacity`) |
| `as_*` | Cheap reference conversion, no allocation (`as_slice`) |
| `to_*` | Expensive conversion, allocates (`to_string`) |
| `into_*` | Consuming conversion, takes `self` (`into_batch`) |
| `is_*` / `has_*` | Returns `bool`, no side effects |
| `iter` / `iter_mut` / `into_iter` | Borrow / mut-borrow / consume |

Getting `as_` vs `to_` vs `into_` right is a real signal of Rust fluency — the reader should be
able to infer allocation cost and ownership from the name alone.

**Domain naming:** use the query-engine term of art, not an invented one. `predicate` not
`filter_condition`. `projection` not `column_selection`. `cardinality` not `row_estimate`.
`selectivity`, `scan`, `bind`, `coerce`. This makes the codebase legible to anyone who has read a
database paper, and it makes *your* reading of DataFusion easier by keeping vocabulary aligned.

---

## 3. API design

**R1 — Take the borrowed view, return the owned type.**

```rust
fn index_of(&self, name: &str) -> Option<usize>       // ✓ &str, not &String
fn take(&self, indices: &[usize]) -> Result<Column>   // ✓ &[usize], not &Vec<usize>
```

`&str` accepts a borrow of a `String`, a literal, or a slice of either. `&Vec<T>` accepts exactly
one thing. Deref coercion makes the general version free for callers.

**R2 — Accept `impl IntoIterator` where it costs nothing**, but don't over-genericize. A
`&[T]` parameter is usually clearer than `impl IntoIterator<Item = &T>` and monomorphizes less.
Generics have a compile-time and binary-size cost; spend it where it buys something.

**R3 — Make invalid states unrepresentable.** Prefer an enum over a struct with mutually
exclusive optional fields. If two fields must agree, put them behind a constructor that enforces
it, and keep the fields private.

**R4 — Enforce invariants at construction, assume them everywhere else.** `try_new` validates;
every method after that may rely on the invariants. Document the invariants on the type:

```rust
/// A column of typed data with optional null tracking.
///
/// # Invariants
/// - `validity.is_none()` implies the column contains no nulls.
/// - If `validity` is `Some`, its length equals the data length.
pub struct Column { /* ... */ }
```

**R5 — Fields are private by default.** Expose accessors. A public field is a permanent API
commitment and it defeats R4.

**R6 — `#[must_use]` on anything pure.** If calling a function and discarding the result is
always a bug, say so. All the builder/transform methods qualify.

**R7 — Consuming builders take `self`, not `&mut self`,** so the type system prevents reuse after
`finish()`.

---

## 4. Ownership & borrowing

**O1 — Borrow by default; own when you must.** Take `&T` unless you need to store, mutate, or
consume.

**O2 — `.clone()` is a decision, not a reflex.** Every `.clone()` in library code needs one of:
(a) it's a cheap `Arc` bump, (b) it's outside a hot path and clarity wins, (c) a comment
explaining why the borrow doesn't work.

Learning-mode exception: while getting something to compile, clone freely. But **a PR is not
allowed to land with clones you haven't revisited.** Grep your diff for `.clone()` before opening
it.

**O3 — Prefer `Arc<T>` over deep clones for shared immutable data.** From Phase 2, buffers and
plan nodes are `Arc`'d. Write `Arc::clone(&x)`, not `x.clone()` — the explicit form signals "cheap
refcount bump" to the reader.

**O4 — Don't fight the borrow checker with `RefCell`.** Interior mutability is an escape hatch,
not a workaround for a design you haven't thought through. Every `RefCell` in library code needs
an ADR.

---

## 5. Error handling

**E1 — One crate-level error enum** (`BasaltError`) with a variant group per layer. Crate-wide
`pub type Result<T> = std::result::Result<T, BasaltError>;`.

**E2 — `thiserror` for definitions.** `#[error("...")]` messages are lowercase, no trailing
period, and describe *what went wrong* from the user's perspective.

**E3 — Errors carry context, not just a category.** `UnknownColumn { name }` beats
`Error::NotFound`. Anything the user needs to fix the problem goes in the variant's fields.

**E4 — Source positions on anything user-facing.** Syntax and binder errors carry a `Span`.
"syntax error" is a bug report; "syntax error at line 3, col 17: expected expression after WHERE"
is a fix.

**E5 — `?` for propagation, never manual `match`-and-return.** Add `#[from]` impls so `?` converts
at the boundary.

**E6 — Recoverable vs unrecoverable.** Could correct code hit this with valid-looking input?
→ `Result`. Only reachable if the program is wrong? → `debug_assert!` or
`BasaltError::Internal`. Malformed CSV is recoverable. A column-length mismatch after `try_new`
validated it is internal.

---

## 6. Modules & layering

**M1 — Dependencies flow one way.** `types → array → batch → expr → plan → exec`. `sql` depends
only on `types`. **No cycles, ever.** Wanting an upward import means the layering is wrong — fix
the design, don't add the import.

**M2 — One concept per file.** If a file exceeds ~500 lines, it's doing two things.

**M3 — `mod.rs` re-exports, and contains no logic.** It's a table of contents.

**M4 — `pub(crate)` by default for internals.** Reserve `pub` for the actual public API. The
`unreachable_pub` lint catches accidental over-exposure.

---

## 7. Documentation

**D1 — Every public item has a doc comment.** First line is a single sentence ending in a period;
it appears in the module index, so make it count.

**D2 — The standard section order:**

```rust
/// Produces a new column containing only the given row positions, in order.
///
/// This is the primitive underlying both filter and sort.
///
/// # Errors
/// Returns [`BasaltError::Internal`] if any index is out of bounds.
///
/// # Panics
/// Never. (Omit this section if the function cannot panic.)
///
/// # Examples
/// ```
/// # use basalt::array::Column;
/// let col = Column::from_i64(vec![10, 20, 30]);
/// let taken = col.take(&[2, 0])?;
/// assert_eq!(taken.len(), 2);
/// # Ok::<(), basalt::BasaltError>(())
/// ```
pub fn take(&self, indices: &[usize]) -> Result<Column>
```

**D3 — Examples are doctests and they run in CI.** An example that doesn't compile is worse than
no example. This also means every public API gets exercised at least once by its own docs.

**D4 — Comments explain *why*, never *what*.** The code says what.

```rust
// ✗ increment the count
count += 1;

// ✓ Null slots still occupy a data slot, so the data index advances even
//   when we skip the value — see the I3 invariant on Column.
data_index += 1;
```

**D5 — Every non-obvious algorithm gets a comment naming it and citing a source.** "Pratt parsing
(precedence climbing) — see Pratt 1973" saves the next reader (which is you, in four months) an
hour.

**D6 — Document invariants on the type, complexity on the method.** If a method is O(n log n) or
allocates, say so in the doc.

---

## 8. `unsafe` policy

**Phase 1: `#![forbid(unsafe_code)]`.** No exceptions.

**Phase 2 onward**, when zero-copy buffers may genuinely require it:

1. `unsafe` requires an **ADR** explaining why safe code is insufficient, with a benchmark
   showing the safe version's cost.
2. Every `unsafe` block carries a `// SAFETY:` comment stating the invariants that make it sound
   and *why they hold here*. No comment → no merge.
3. `unsafe` is confined to the smallest possible module, wrapped in a safe API. It never leaks
   into call sites.
4. Anything touching `unsafe` gets a Miri run in CI and a fuzz target.

The bar is deliberately high. Most "I need `unsafe` for performance" instincts are wrong, and
bounds-check elimination via iterators (see `docs/PERFORMANCE.md`) usually gets you there safely.

---

## 9. Dependencies

**Every new dependency requires justification in the PR description.** The default answer is no.

| Phase | Allowed |
|---|---|
| 1 | `thiserror` |
| 2 | `+ criterion` (dev), `parquet`/`arrow` only if a deliberate ADR says so |
| 3 | `+ proptest` (dev) |
| 4 | `+ tokio`, `serde`, `object_store`, `tonic` |
| 5 | `+ tracing`, `prometheus`, `sqllogictest` (dev), `cargo-fuzz` (dev) |

`cargo deny` runs in CI: license check, advisory check, duplicate-version check.

Hand-rolling is the point of Phase 1. Adding `sqlparser-rs` to skip the parser defeats the entire
project.

---

## 10. The pre-commit gauntlet

Run all of these before every commit. Wire them as a git hook or a `just check` recipe:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo doc --no-deps --document-private-items
```

**A commit that fails any of these does not get pushed.** CI runs the same set plus `cargo deny`,
MSRV, and benchmark compilation. See `docs/TOOLING.md` for the exact configuration.

---

## 11. Review checklist

Applied to every PR, including your own:

- [ ] All hard rules (§1) satisfied; every `#[allow]` has a justification comment
- [ ] Public items documented, with `# Errors` where fallible
- [ ] Invariants documented on the type **and** tested
- [ ] No `.clone()` you can't justify
- [ ] Borrowed params (`&str`, `&[T]`) where possible
- [ ] Errors carry actionable context; spans on user-facing errors
- [ ] Tests: unit + at least one end-to-end path
- [ ] Naming follows §2, including domain vocabulary
- [ ] No new dependency (or justified)
- [ ] Performance claims backed by a benchmark number
