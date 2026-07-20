# Contributing to Basalt

Read `docs/CODING_STANDARDS.md`, `docs/PERFORMANCE.md`, and `docs/TESTING.md` before your first
change. This file covers *process*: what "done" means, and how work moves.

---

## Definition of Done

A change is done when **every** box is checked. Not most.

- [ ] It compiles with zero warnings (`cargo clippy --all-targets -- -D warnings`)
- [ ] It is formatted (`cargo fmt --all --check`)
- [ ] All tests pass, in both debug and release
- [ ] New public items have doc comments, with `# Errors` where fallible
- [ ] New invariants are documented on the type **and** have a test that proves they fire
- [ ] New behavior has: a unit test, an error-path test, and a boundary test (empty / single / null)
- [ ] Every `#[allow]` outside tests has a justification comment
- [ ] Every `.clone()` in the diff has been revisited and justified
- [ ] Any performance claim has a benchmark number in the PR description
- [ ] Any new dependency is justified in the PR description
- [ ] Any architectural decision is recorded as an ADR

**"I'll add tests later" is not done.** The test is part of the change, because the test is what
encodes the intent.

---

## Workflow

```bash
git checkout -b <area>/<short-description>     # e.g. expr/three-valued-logic
# ... work ...
just check                                      # the full local gauntlet
git commit
```

**Small commits, each one green.** A commit that doesn't compile is a commit that can't be
bisected, and bisect is how you'll find the regression you introduce in Phase 3 and notice in
Phase 4.

### Commit messages

Conventional Commits, scoped by module:

```
feat(expr): add three-valued logic for AND/OR
fix(csv): handle CRLF line endings in type inference
perf(kernel): avoid per-row allocation in filter — 2.3x on 1M rows
refactor(array): hoist validity out of ColumnData variants
test(expr): cover every cell of the AND/OR truth tables
docs(adr): record why validity uses Vec<bool> in Phase 1
```

Body answers **why**, not what — the diff already says what. For `perf:` commits, **the number
goes in the subject line.** You will want that history when you write `BENCHMARKS.md`.

---

## Architecture Decision Records

Any decision that a future reader would ask "why on earth is it like this?" gets an ADR in
`docs/adr/NNNN-short-title.md`:

```markdown
# NNNN — Title

**Status:** proposed | accepted | superseded by ADR-MMMM
**Date:** YYYY-MM-DD

## Context
What forced a decision. Constraints, requirements, what we knew at the time.

## Options considered
1. Option A — pros, cons
2. Option B — pros, cons
3. Option C — pros, cons

## Decision
What we chose, and the deciding reason.

## Consequences
What this makes easy. What it makes hard. What we'll have to revisit, and when.
```

**ADRs are required for:** any use of `unsafe`, any new dependency, any change to the module
layering, any data-structure choice with a performance tradeoff, any deliberate deviation from
the standards.

Seed the log with the decisions already made in the Phase 1 LLD (validity representation,
bound/unbound expression split, coercion-as-explicit-casts, eager execution). Writing them up is
an hour and it turns "design I remember" into "design the project records."

---

## Pull requests

Even solo, open a PR and let CI run. The description template:

```markdown
## What
One paragraph.

## Why
The problem this solves.

## Design notes
Non-obvious choices and what else was considered. Link the ADR if there is one.

## Benchmarks          (required for anything touching a hot path)
| Benchmark | Before | After | Δ |
|---|---|---|---|
| filter_1m_rows | 12.4 ms | 5.3 ms | −57% |

## Checklist
- [ ] Definition of Done satisfied
```

Self-review the diff in the GitHub UI before merging. You will catch things there that you did not
catch in your editor — different surface, different attention.

---

## AI-assisted development policy

This project is built partly with AI assistance. That's fine, and it has one hard constraint:

> **You must be able to explain every line you merge, without assistance, months later.**

The reason is practical, not moral. This codebase is a portfolio artifact and a learning vehicle.
Code you can't explain fails at both jobs, and it fails loudly the first time someone asks you to
walk through your join-order enumeration.

**Use AI freely for:** explaining concepts, API discovery, debugging borrow-checker errors,
boilerplate (`Display` impls, test scaffolding, serde structs), code review, generating test cases
and edge cases, and critiquing a design you already sketched.

**Write yourself:** the parser, the optimizer, the join algorithms, the execution model, and
every architectural decision. These are the parts you'll be asked about, and the parts where the
learning actually lives.

**Three checks:**

1. **The explain-back test.** After any AI-assisted section, close the chat and explain the code
   aloud — or write it into the blog post. Can't? Delete it and write it yourself.
2. **Never merge code you couldn't modify.** If you couldn't confidently change its behavior, you
   don't own it yet.
3. **Design before you ask.** "How do I implement X" skips the learning. "Here's my design for X,
   what breaks?" is where the value is.

The `.claude/skills/` directory encodes this project's standards so AI assistance follows them by
default rather than being told each time.

---

## Getting started

```bash
cargo install just cargo-watch cargo-deny flamegraph
just check          # verify a clean baseline
just watch          # fast inner loop while working
```

Pick the next unstarted item from the Phase build order. Read the LLD section for it first — the
design is already made, and the contracts are already specified. Your job is the implementation
and the tests.
