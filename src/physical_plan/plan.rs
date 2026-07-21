//! `ExecutionPlan` and the batch-stream abstraction. See
//! design-docs/basalt-phase2-lld.md §5.3.
//!
//! Pull-based (Volcano), batch-granular: each operator's stream, polled,
//! pulls from its children until it can produce an output batch. Push-based
//! / morsel-driven execution has real advantages and is a legitimate
//! benchmarked experiment later — implement pull-based first and
//! completely, per the LLD.

use std::any::Any;
use std::sync::Arc;

use crate::batch::ColumnarBatch;
use crate::error::Result;
use crate::types::schema::SchemaRef;

/// Phase 2: a synchronous iterator of batches. Phase 4 turns this alias into
/// a `Stream` and `execute` into an `async fn` without touching call sites —
/// the trait shape below is deliberately already compatible with that swap.
pub type BatchStream = Box<dyn Iterator<Item = Result<ColumnarBatch>> + Send>;

/// How an operator's output rows are distributed across partitions. Phase 2
/// is always single-partition; the type exists now so Phase 4's shuffle
/// doesn't need a trait change, only new variants and real users of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Partitioning {
    UnknownPartitioning(usize),
}

impl Partitioning {
    pub fn partition_count(&self) -> usize {
        match self {
            Partitioning::UnknownPartitioning(n) => *n,
        }
    }
}

/// Per-operator execution metrics. Stubbed in Phase 2; Phase 5's
/// `EXPLAIN ANALYZE` is what actually populates and reads this.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub rows_produced: usize,
}

pub trait ExecutionPlan: std::fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn schema(&self) -> SchemaRef;
    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>>;

    /// # Errors
    /// Errors if `children`'s length doesn't match this operator's arity.
    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>>;

    /// Produce the batches for one output partition.
    ///
    /// # Errors
    /// Errors if `partition` is out of range for `output_partitioning()`, or
    /// if the operator can't be started (e.g. a child fails to execute).
    fn execute(&self, partition: usize) -> Result<BatchStream>;

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn metrics(&self) -> Option<Metrics> {
        None
    }
}

pub type ExecutionPlanRef = Arc<dyn ExecutionPlan>;
