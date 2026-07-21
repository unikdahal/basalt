//! The physical execution plan: batch-at-a-time operators pulled Volcano-
//! style from their children. See design-docs/basalt-phase2-lld.md §5.3–5.4.

pub mod aggregate;
pub mod filter;
pub mod limit;
pub mod plan;
pub mod planner;
pub mod projection;
pub mod scan;

pub use plan::{BatchStream, ExecutionPlan, ExecutionPlanRef, Metrics, Partitioning};
pub use planner::PhysicalPlanner;
