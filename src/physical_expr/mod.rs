//! Physical expressions: the tree evaluated once per batch. See
//! design-docs/basalt-phase2-lld.md §5.2.

pub mod binary;
pub mod cast;
pub mod column;
pub mod expr;
pub mod is_null;
pub mod literal;
pub mod unary;

pub use expr::{PhysicalExpr, PhysicalExprRef};
