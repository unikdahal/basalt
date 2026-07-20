# Basalt — Testing & Correctness

**Status:** normative. Untested code does not merge.

---

## 0. The premise

> **A query engine that is fast and wrong is worthless.**

Correctness bugs in a query engine are uniquely bad: they're silent. A crash gets fixed; a
`SUM` that's off by the rows where a predicate met a `NULL` produces a plausible number that
someone builds a decision on. Nobody files a bug, because nobody notices.

So the standard here is higher than "it has tests." The standard is: **for every claim the code
makes, there is a mechanism that would catch it being false.**

---

## 1. The test pyramid

| Layer | What it proves | Speed | Count |
|---|---|---|---|
| **Unit** | A function does what its doc says | ms | Most |
| **Invariant** | A documented invariant actually holds | ms | One per invariant |
| **Property** | An algebraic law holds for *all* inputs | s | Per law |
| **End-to-end** | SQL in → correct table out | s | Per feature |
| **Differential** | We agree with a reference implementation | s–min | Per query class |
| **Fuzz** | No input crashes or hangs the parser | continuous | Per entry point |

Bias toward end-to-end and differential tests. They're what actually catch the bugs that matter,
and they survive refactors — unit tests get deleted when you restructure a module, but
"this SQL produces this table" is true forever.

---

## 2. Rules

**T1 — Every documented invariant has a test that proves it fires.** If `Column` documents
"validity length equals data length," there is a test constructing the violation and asserting the
error. An invariant without a test is a comment.

**T2 — Every bug fix starts with a failing test.** Write the test that reproduces it, watch it
fail, then fix. This is the only way to know your fix addresses the actual cause, and it
permanently prevents the regression.

**T3 — Every public function is exercised by at least its doctest.** Doctests run in CI, so
examples cannot rot.

**T4 — Test the error paths, not just the happy path.** Malformed CSV, unknown column, type
mismatch, division by zero, empty input, out-of-range `LIMIT`. Assert on the *specific* error
variant, not just "it errored."

**T5 — Test the boundaries.** Zero rows. One row. Zero columns. All nulls. No nulls. Empty
string. `i64::MIN` / `MAX`. `f64::NAN` / `INFINITY`. Unicode in identifiers and string literals.
These are where the bugs live.

**T6 — No flaky tests.** A test that fails intermittently is deleted or fixed the same day.
Tolerating flakiness trains you to ignore red CI, which is how real failures ship.

**T7 — Tests are deterministic.** No wall-clock dependence, no unseeded randomness, no reliance
on `HashMap` iteration order (it's randomized per-map, by design).

---

## 3. Naming and structure

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_returns_rows_in_requested_order() { /* ... */ }

    #[test]
    fn take_errors_when_index_out_of_bounds() { /* ... */ }

    #[test]
    fn null_and_true_is_null() { /* ... */ }
}
```

**Name tests as sentences describing the behavior**, not `test_take_1`. The name is the spec, and
a failing test name should tell you what broke without opening the file.

`unwrap()` and `expect()` are **allowed and encouraged in tests** — a panic is the correct
failure mode there. The lint config allows them under `#[cfg(test)]`.

---

## 4. Property-based testing (`proptest`)

Some things are true for *all* inputs. Assert those directly rather than sampling by hand:

| Property | Law |
|---|---|
| Round-trip | `parse(print(expr)) == expr` |
| Filter | `filter(p).len() <= input.len()` |
| Filter idempotence | `filter(p).filter(p) == filter(p)` |
| Sort | output is ordered, and is a permutation of the input |
| Limit | `limit(n).len() == min(n, input.len())` |
| **Optimizer (Phase 3)** | **`execute(optimize(plan)) == execute(plan)`** |

That last one is the single most valuable test in the entire project. A rewrite rule that changes
results is the worst possible bug class, and property testing over generated plans is the only
practical way to catch it. Build the generator infrastructure in Phase 1 (generating random
expressions is easy) so it's ready when the optimizer arrives.

**Shrinking is the payoff:** when proptest finds a failure it minimizes the input automatically,
so you get the *smallest* expression that breaks your evaluator, not a 40-node monster.

---

## 5. End-to-end tests

Text in, text out, as data files rather than Rust code:

```
tests/
├── sql/
│   ├── select_basic.sql        → select_basic.expected
│   ├── where_null.sql          → where_null.expected
│   └── order_by_nulls.sql      → order_by_nulls.expected
└── harness.rs                  # runs every .sql/.expected pair in the directory
```

Adding a test becomes adding two text files, which means you'll actually add them. The harness
compares rendered output byte-for-byte.

**Design it as a miniature `sqllogictest` from day one** — same shape (statement, expected result),
so that in Phase 5 you swap in the real sqllogictest crate and inherit thousands of existing
correctness cases written for other engines. That's a very large amount of free correctness for
an hour of design foresight now.

---

## 6. Three-valued logic — a required suite

SQL `NULL` semantics get their own section because they're the most commonly-wrong part of any
hand-built engine, and the bugs are silent.

**Required coverage:**

- Every cell of the `AND` truth table (9 cases)
- Every cell of the `OR` truth table (9 cases)
- `NOT NULL` → `NULL`
- Null propagation through every arithmetic operator
- Null propagation through every comparison operator
- `NULL = NULL` → `NULL` (**not** `true`)
- `IS NULL` / `IS NOT NULL` never return null
- `WHERE` rejects both `false` **and** `NULL`
- `ORDER BY` null placement matches the documented policy

That's roughly 40 tiny tests, and it's the highest value-per-line test suite in Phase 1. Write it
as a table-driven test so adding a case is one line.

---

## 7. Differential testing (Phase 3+)

Run the same query against Basalt and a reference engine (DuckDB — embeddable, fast, correct) and
compare results.

- Generate random queries over a fixed schema, or curate a query corpus
- Normalize ordering before comparing when the query has no `ORDER BY`
- Any mismatch is a bug in Basalt until proven otherwise
- Document every intentional divergence (dialect differences are real; undocumented ones are bugs)

This is the highest-leverage correctness technique available, because it tests against tens of
thousands of person-hours of someone else's correctness work.

---

## 8. Fuzzing (Phase 4+)

`cargo-fuzz` targets on every parsing entry point:

- **Lexer/parser:** no input may panic, hang, or overflow the stack. Malformed SQL must produce
  a clean `Err`, always.
- **CSV reader:** no input may panic. Truncated files, wrong field counts, invalid UTF-8.
- **Anything with `unsafe`** (Phase 2+): fuzz plus Miri, non-negotiable.

Deeply nested expressions (`((((((...))))))`) are the classic stack-overflow finder. Add a
recursion depth limit *before* fuzzing tells you to.

---

## 9. CI gates

Every PR must pass:

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features                    # unit + integration + doctests
cargo test --release                         # catches debug-only assumptions
cargo bench --no-run                         # benchmarks must compile
cargo doc --no-deps                          # docs must build
cargo deny check                             # licenses, advisories
```

Plus, from the phase where each becomes relevant: property tests, differential tests, Miri (if
`unsafe` exists), and the benchmark regression check.

---

## 10. Coverage

Measure it (`cargo llvm-cov`), but don't target a number. **Coverage tells you what is *not*
tested; it says nothing about whether what *is* tested is tested well.** 100% coverage with
assertions that never fail is worthless.

Use it as a checklist: look at the uncovered lines and decide, one by one, whether each is
genuinely unreachable or is a missing test. Usually it's the error paths — which are exactly
what T4 says to test.
