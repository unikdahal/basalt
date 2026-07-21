//! Statistics for cost-based optimization. See
//! design-docs/basalt-phase3-lld.md §2.
//!
//! Sits below `optimizer` (`statistics -> optimizer -> {logical_plan,
//! physical_plan}`). Nothing here depends on the optimizer.

pub mod collect;
pub mod histogram;
pub mod hll;
pub mod mcv;
pub mod precision;
pub mod stats;

pub use collect::{analyze, statistics_from_parquet, StatisticsProvider};
pub use histogram::{Bucket, Histogram};
pub use hll::HyperLogLog;
pub use mcv::MostCommonValues;
pub use precision::Precision;
pub use stats::{ColumnStatistics, TableStatistics};
