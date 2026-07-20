//! Bound expression tree: typing and row-at-a-time evaluation.
//! See design-docs/basalt-phase1-lld.md §5.

// The LLD names this file `expr/expr.rs` for the bound `Expr` tree specifically
// (mirrored by `expr/typing.rs`, `expr/eval.rs`); clippy reads it as a name clash
// with the parent module, but the naming is intentional.
pub mod eval;
#[allow(clippy::module_inception)]
pub mod expr;
pub mod typing;
