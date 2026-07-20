# Basalt — Tooling & Enforcement

**This file is the one that makes the standards real.** Everything in `CODING_STANDARDS.md`,
`PERFORMANCE.md`, and `TESTING.md` is aspirational until a lint or a CI job enforces it. Copy
these configs into the repo and the rules stop being suggestions.

---

## 1. Lints — `Cargo.toml`

Rust 1.74+ supports a `[lints]` table in `Cargo.toml`, which is better than a `lib.rs` header
because it applies to all targets including benches and tests.

```toml
[lints.rust]
unsafe_code            = "forbid"   # Phase 1. Becomes "deny" + per-module allow in Phase 2.
missing_docs           = "warn"
unreachable_pub        = "warn"
missing_debug_implementations = "warn"
rust_2018_idioms       = { level = "warn", priority = -1 }

[lints.clippy]
# Baseline groups (priority -1 so specific lints below can override)
all       = { level = "deny",  priority = -1 }
pedantic  = { level = "warn",  priority = -1 }
nursery   = { level = "warn",  priority = -1 }
cargo     = { level = "warn",  priority = -1 }

# ---- Correctness: hard rules H1-H3 ----
unwrap_used            = "deny"
expect_used            = "deny"
panic                  = "deny"
todo                   = "deny"
unimplemented          = "deny"
indexing_slicing       = "deny"
integer_division       = "deny"
lossy_float_literal    = "deny"
mem_forget             = "deny"
exit                   = "deny"

# ---- Correctness: the subtle ones ----
float_cmp              = "deny"   # == on floats is almost always a bug
cast_possible_truncation = "deny"
cast_sign_loss         = "deny"
cast_precision_loss    = "warn"
as_conversions         = "warn"   # prefer TryFrom over `as`

# ---- Performance ----
redundant_clone        = "deny"
needless_collect       = "deny"
inefficient_to_string  = "deny"
large_stack_arrays     = "deny"
large_types_passed_by_value = "warn"
trivially_copy_pass_by_ref  = "warn"
needless_pass_by_value = "warn"
suboptimal_flops       = "warn"

# ---- Documentation ----
missing_errors_doc     = "warn"
missing_panics_doc     = "warn"
doc_markdown           = "warn"
missing_const_for_fn   = "warn"

# ---- Style / clarity ----
must_use_candidate     = "warn"
if_not_else            = "warn"
match_same_arms        = "warn"
semicolon_if_nothing_returned = "warn"
uninlined_format_args  = "warn"   # println!("{x}") over println!("{}", x)

# ---- Deliberately relaxed, with reasons ----
module_name_repetitions = "allow"  # array::ArrayBuilder reads fine
missing_inline_in_public_items = "allow"  # see PERFORMANCE.md §4 — don't sprinkle #[inline]
multiple_crate_versions = "allow"  # transitive deps; cargo-deny handles this properly
```

**Tests get an exemption** for the panic family — a panic is the correct failure mode in a test.
Put this at the top of every test module:

```rust
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    // ...
}
```

**The `#[allow]` rule:** any `#[allow]` outside a test module requires a comment on the line above
explaining why. No comment → review reject. That comment is the entire point of the strict
baseline: it converts "I reached for `unwrap`" into "I justified reaching for `unwrap`, in
writing, where a reviewer can check it."

---

## 2. `clippy.toml`

```toml
# Complexity ceilings — exceed these and the function needs splitting
too-many-arguments-threshold  = 6
too-many-lines-threshold      = 100
cognitive-complexity-threshold = 20
type-complexity-threshold     = 250

# Force explicit thresholds rather than magic numbers
enum-variant-name-threshold   = 3
trivial-copy-size-limit       = 16          # bytes; larger types pass by reference

msrv = "1.75.0"

# Ban footguns outright
disallowed-methods = [
    { path = "std::collections::HashMap::new", reason = "use with_capacity when the size is known, or HashMap::default with a faster hasher" },
]
```

Tune `disallowed-methods` as you find your own footguns — it's the mechanism for turning "we
learned not to do X" into something the compiler remembers for you.

---

## 3. `rustfmt.toml`

```toml
edition = "2021"
max_width = 100
newline_style = "Unix"
use_field_init_shorthand = true
use_try_shorthand = true
reorder_imports = true
reorder_modules = true

# --- nightly-only (run `cargo +nightly fmt`); harmless to leave on stable ---
imports_granularity = "Module"
group_imports = "StdExternalCrate"
format_code_in_doc_comments = true
normalize_comments = true
wrap_comments = true
comment_width = 100
```

`group_imports = "StdExternalCrate"` produces std / external / crate-local import blocks
automatically, which removes an entire category of review nitpick.

---

## 4. `deny.toml` (cargo-deny)

```toml
[advisories]
yanked = "deny"
ignore = []

[licenses]
allow = ["Apache-2.0", "MIT", "BSD-3-Clause", "ISC", "Unicode-DFS-2016"]
confidence-threshold = 0.9

[bans]
multiple-versions = "warn"
wildcards = "deny"          # no "*" version requirements, ever

[sources]
unknown-registry = "deny"
unknown-git = "deny"
```

---

## 5. `.cargo/config.toml`

```toml
[build]
rustflags = ["-C", "target-cpu=native"]   # local dev only — see note

[alias]
c  = "check --all-targets --all-features"
t  = "test --all-features"
l  = "clippy --all-targets --all-features -- -D warnings"
b  = "bench"
```

**Note on `target-cpu=native`:** it enables AVX-512 etc. on your machine, which is great for
local benchmarking and produces binaries that won't run elsewhere. Do **not** set it in CI or for
release artifacts — you'd be benchmarking a build nobody else can reproduce.

---

## 6. Release profile — `Cargo.toml`

```toml
[profile.release]
opt-level = 3
lto = "fat"           # link-time optimization across crates; big win, slower builds
codegen-units = 1     # better optimization, slower builds — worth it for benchmarks
panic = "abort"       # smaller/faster; revisit in Phase 4 when threads need unwinding

[profile.bench]
inherits = "release"
debug = true          # keep symbols so flamegraphs are readable

# Fast iteration: optimize dependencies even in debug builds
[profile.dev.package."*"]
opt-level = 3
```

That last block is a genuine quality-of-life win — your own crate stays fast to compile and
debuggable, while dependencies get optimized once and cached.

---

## 7. CI — `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-D warnings"

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { components: rustfmt, clippy }
      - uses: Swatinem/rust-cache@v2

      - name: Format
        run: cargo fmt --all --check

      - name: Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

      - name: Test (debug)
        run: cargo test --all-features

      - name: Test (release)
        run: cargo test --release --all-features

      - name: Doc
        run: cargo doc --no-deps --all-features
        env: { RUSTDOCFLAGS: "-D warnings" }

      - name: Benchmarks compile
        run: cargo bench --no-run

  deny:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: EmbarkStudios/cargo-deny-action@v2

  msrv:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@1.75.0
      - run: cargo check --all-features
```

**Why both debug and release tests:** `debug_assert!` only fires in debug, and integer overflow
only panics in debug. Release-only bugs are real and this catches them. It costs a few CI minutes
and it will save you a day.

Add from the relevant phase: **Miri** (once `unsafe` exists), **fuzz smoke runs**, and a
**benchmark regression check** against the previous commit.

---

## 8. `justfile` — the local gauntlet

```just
default: check

# Run before every commit — same gates as CI
check:
    cargo fmt --all
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test --all-features
    cargo doc --no-deps

# Fast inner loop while writing code
watch:
    cargo watch -x "check --all-targets"

bench:
    cargo bench

flame BENCH:
    cargo flamegraph --bench {{BENCH}}

# What CI will run, locally
ci: check
    cargo test --release --all-features
    cargo bench --no-run
    cargo deny check
```

Install once: `cargo install just cargo-watch cargo-deny flamegraph cargo-llvm-cov`.

---

## 9. Git hooks

`.git/hooks/pre-commit` (or use `cargo-husky` to version it):

```bash
#!/usr/bin/env bash
set -euo pipefail
cargo fmt --all --check || { echo "run: cargo fmt --all"; exit 1; }
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

The hook exists so red CI is *rare*, not so it's caught. Catching it locally takes 30 seconds;
catching it in CI takes a context switch.

---

## 10. Adoption order

Turning all of this on at once, mid-project, produces hundreds of warnings and you'll disable it.
Turn it on in this order instead:

1. **Day 1:** `rustfmt.toml`, the `justfile`, CI with fmt + clippy(default) + test
2. **Day 1:** hard rules H1–H3 (`unwrap_used`, `expect_used`, `indexing_slicing`) — these are the
   habit-forming ones and they're cheap when the codebase is small
3. **End of module 1.2:** `pedantic` + `nursery` as warnings; fix them as you go
4. **End of Phase 1:** `missing_docs`, doctests, `cargo-deny`
5. **Phase 2:** benchmark gates, release-mode CI, size assertions
6. **Phase 2 (if `unsafe` appears):** Miri, fuzzing

Strict from the start on a small codebase is easy. Strict retrofitted onto 20K lines is a
weekend of misery, and that's how projects end up with `#![allow(clippy::all)]` at the top.
