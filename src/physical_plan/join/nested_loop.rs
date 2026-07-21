//! `NestedLoopJoinExec` — the fallback for non-equi join predicates
//! (`ON a.x < b.y`), where hashing doesn't apply. See
//! design-docs/basalt-phase2-lld.md §7.2.
//!
//! `O(build_rows × probe_rows)` and therefore a last resort, not a first
//! choice — but the planner needs *some* plan it can always produce for an
//! arbitrary join predicate. **Inner join only** in this implementation;
//! outer nested-loop variants are a scope cut for this pass, same spirit as
//! `HashJoinExec`'s missing residual-filter support.
//!
//! Batched per build row rather than per pair: for build row `b`, broadcast
//! it across the *whole* probe batch (`take` with `b` repeated) and evaluate
//! the predicate once, vectorized over every probe row — not one predicate
//! evaluation per `(b, p)` pair, which would be evaluation-per-scalar all
//! over again.

use std::any::Any;
use std::sync::Arc;

use super::super::plan::{BatchStream, ExecutionPlan};
use crate::array::array::{as_boolean, Array, ArrayRef};
use crate::batch::ColumnarBatch;
use crate::compute::index::UInt32Builder;
use crate::compute::take::take;
use crate::error::{BasaltError, Result};
use crate::physical_expr::PhysicalExprRef;
use crate::types::schema::SchemaRef;

#[derive(Debug)]
pub struct NestedLoopJoinExec {
    build: Arc<dyn ExecutionPlan>,
    probe: Arc<dyn ExecutionPlan>,
    /// Evaluated over a batch combining build (broadcast) columns followed
    /// by probe columns — i.e. against `self.schema()`'s column layout.
    predicate: PhysicalExprRef,
    schema: SchemaRef,
}

impl NestedLoopJoinExec {
    pub fn new(
        build: Arc<dyn ExecutionPlan>,
        probe: Arc<dyn ExecutionPlan>,
        predicate: PhysicalExprRef,
        schema: SchemaRef,
    ) -> Self {
        NestedLoopJoinExec {
            build,
            probe,
            predicate,
            schema,
        }
    }
}

use super::hash_join::materialize;

impl ExecutionPlan for NestedLoopJoinExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![Arc::clone(&self.build), Arc::clone(&self.probe)]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.as_slice() {
            [build, probe] => Ok(Arc::new(NestedLoopJoinExec::new(
                Arc::clone(build),
                Arc::clone(probe),
                Arc::clone(&self.predicate),
                self.schema.clone(),
            ))),
            other => Err(BasaltError::Internal(format!(
                "NestedLoopJoinExec takes exactly 2 children, got {}",
                other.len()
            ))),
        }
    }

    fn execute(&self, partition: usize) -> Result<BatchStream> {
        let build_batch = materialize(self.build.as_ref(), partition)?;
        let probe_batch = materialize(self.probe.as_ref(), partition)?;
        let build_rows = build_batch.num_rows();
        let probe_rows = probe_batch.num_rows();

        let mut pair_build = UInt32Builder::with_capacity(0);
        let mut pair_probe = UInt32Builder::with_capacity(0);

        if build_rows > 0 && probe_rows > 0 {
            for b in 0..build_rows {
                let mut idx = UInt32Builder::with_capacity(probe_rows);
                for _ in 0..probe_rows {
                    idx.append_value(b as u32);
                }
                let idx = idx.finish();

                let mut combined_columns: Vec<ArrayRef> =
                    Vec::with_capacity(build_batch.num_columns() + probe_batch.num_columns());
                for i in 0..build_batch.num_columns() {
                    combined_columns.push(take(build_batch.column(i).unwrap().as_ref(), &idx)?);
                }
                for i in 0..probe_batch.num_columns() {
                    combined_columns.push(Arc::clone(probe_batch.column(i).unwrap()));
                }
                let combined = ColumnarBatch::try_new(self.schema.clone(), combined_columns)?;

                let matched = self.predicate.evaluate(&combined)?.into_array(probe_rows)?;
                let matched = as_boolean(matched.as_ref())?;
                for p in 0..probe_rows {
                    if matched.is_valid(p) && matched.value(p) {
                        pair_build.append_value(b as u32);
                        pair_probe.append_value(p as u32);
                    }
                }
            }
        }

        let build_indices = pair_build.finish();
        let probe_indices = pair_probe.finish();
        let mut columns = Vec::with_capacity(self.schema.len());
        for i in 0..build_batch.num_columns() {
            columns.push(take(
                build_batch.column(i).unwrap().as_ref(),
                &build_indices,
            )?);
        }
        for i in 0..probe_batch.num_columns() {
            columns.push(take(
                probe_batch.column(i).unwrap().as_ref(),
                &probe_indices,
            )?);
        }
        let out = ColumnarBatch::try_new(self.schema.clone(), columns)?;
        Ok(Box::new(std::iter::once(Ok(out))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_primitive;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::physical_expr::binary::BinaryExpr;
    use crate::physical_expr::column::ColumnExpr;
    use crate::physical_plan::scan::MemoryScanExec;
    use crate::types::coercion::BinaryOp;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema};

    fn schema(name: &str) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]).unwrap())
    }

    fn batch(schema: SchemaRef, values: &[i64]) -> ColumnarBatch {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            b.append_value(v);
        }
        ColumnarBatch::try_new(schema, vec![Arc::new(b.finish())]).unwrap()
    }

    #[test]
    fn joins_on_a_non_equi_predicate() {
        let build = Arc::new(MemoryScanExec::new(
            schema("b"),
            vec![batch(schema("b"), &[1, 2, 3])],
        ));
        let probe = Arc::new(MemoryScanExec::new(
            schema("p"),
            vec![batch(schema("p"), &[2])],
        ));
        let output_schema = Arc::new(
            Schema::new(vec![
                Field::new("b", DataType::Int64, false),
                Field::new("p", DataType::Int64, false),
            ])
            .unwrap(),
        );
        // ON b.b < p.p : only build row 1 (value 1) satisfies 1 < 2
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(ColumnExpr::new(0)),
            BinaryOp::Lt,
            Arc::new(ColumnExpr::new(1)),
        ));
        let exec = NestedLoopJoinExec::new(build, probe, predicate, output_schema);
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 1);
        let bcol = as_primitive::<Int64Type>(batches[0].column(0).unwrap().as_ref()).unwrap();
        assert_eq!(bcol.value(0), 1);
    }

    #[test]
    fn empty_probe_side_yields_no_rows() {
        let build = Arc::new(MemoryScanExec::new(
            schema("b"),
            vec![batch(schema("b"), &[1, 2])],
        ));
        let probe = Arc::new(MemoryScanExec::new(schema("p"), vec![]));
        let output_schema = Arc::new(
            Schema::new(vec![
                Field::new("b", DataType::Int64, false),
                Field::new("p", DataType::Int64, false),
            ])
            .unwrap(),
        );
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(ColumnExpr::new(0)),
            BinaryOp::Lt,
            Arc::new(ColumnExpr::new(1)),
        ));
        let exec = NestedLoopJoinExec::new(build, probe, predicate, output_schema);
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 0);
    }
}
