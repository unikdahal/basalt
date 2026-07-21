//! The cost-based optimizer. See design-docs/basalt-phase3-lld.md.
//!
//! Sits above `statistics`, below nothing (`statistics -> optimizer ->
//! {logical_plan, physical_plan}`). Bottom-up dynamic programming for join
//! ordering plus a rule engine for everything else — what Postgres and
//! DataFusion do, and the right scope for this project (see §7.4: a full
//! Cascades/Volcano top-down transformational optimizer is explicitly out
//! of scope).

pub mod cardinality;
pub mod cost;
pub mod join_order;
// LLD names this file `optimizer/optimizer.rs` for the pass manager
// specifically (mirroring `buffer/buffer.rs` from Phase 2); clippy reads it
// as a name clash with the parent module, but it's intentional.
#[allow(clippy::module_inception)]
pub mod optimizer;
pub mod rule;
pub mod rules;
pub mod tree_node;

pub use optimizer::default_optimizer;
pub use rule::{ApplyOrder, NoStatistics, Optimizer, OptimizerContext, OptimizerRule};
pub use tree_node::{Transformed, TreeNode, VisitRecursion};
