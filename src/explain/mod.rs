//! `EXPLAIN` / `EXPLAIN ANALYZE`. See design-docs/basalt-phase3-lld.md §8.
//!
//! Sits on top of everything (`statistics -> optimizer -> {logical_plan,
//! physical_plan} -> explain`).

pub mod analyze;
pub mod display;

pub use analyze::{explain_analyze, q_error, AnalyzeNode, OperatorMetrics};
pub use display::{explain, explain_physical};
