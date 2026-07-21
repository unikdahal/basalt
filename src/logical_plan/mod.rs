//! The logical query plan. See design-docs/basalt-phase2-lld.md §5.1.

pub mod builder;
pub mod display;
pub mod plan;

pub use builder::LogicalPlanBuilder;
pub use plan::{
    AggregateFunction, AggregateKind, JoinType, LogicalPlan, SortExpr, SortOptions, TableSource,
};
