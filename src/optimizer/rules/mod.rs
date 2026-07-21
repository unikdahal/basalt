//! The rule catalogue. See design-docs/basalt-phase3-lld.md §5.3-§6.

mod common;

pub mod common_subexpr;
pub mod constant_folding;
pub mod eliminate_cross_join;
pub mod eliminate_filter;
pub mod limit_pushdown;
pub mod merge_filters;
pub mod predicate_pushdown;
pub mod projection_pushdown;
pub mod simplify_expressions;
