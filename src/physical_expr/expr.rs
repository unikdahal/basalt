//! `PhysicalExpr` — the hot path of the engine. See
//! design-docs/basalt-phase2-lld.md §5.2.
//!
//! **`dyn`, not an enum, unlike `LogicalPlan`.** Physical expressions are an
//! open set — Phase 5 adds UDFs, users add custom expressions, and none of
//! them can be enum variants in this crate. The vtable cost is one virtual
//! call per *batch*, amortized over `DEFAULT_BATCH_SIZE` rows — the same
//! economics that make the `Array` trait's downcast-once-per-batch pattern
//! (module 2.1) pay for itself.
//!
//! Contrast with Phase 1's `eval(expr, batch, row)`: that was one recursive
//! tree walk *per row*. This is one tree walk per batch, with each node
//! doing `batch.num_rows()` of work in a tight loop — the tree-walk overhead
//! that dominated Phase 1's profile divides by the batch size.

use std::any::Any;
use std::sync::Arc;

use crate::batch::ColumnarBatch;
use crate::compute::ColumnarValue;
use crate::error::Result;
use crate::types::data_type::DataType;
use crate::types::schema::Schema;

pub trait PhysicalExpr: std::fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn data_type(&self, input_schema: &Schema) -> Result<DataType>;
    fn nullable(&self, input_schema: &Schema) -> Result<bool>;

    /// Evaluate over an entire batch. THE hot path of the engine.
    fn evaluate(&self, batch: &ColumnarBatch) -> Result<ColumnarValue>;

    fn children(&self) -> Vec<PhysicalExprRef>;
}

pub type PhysicalExprRef = Arc<dyn PhysicalExpr>;
