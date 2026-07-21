//! `FilterExec` — evaluates a predicate per batch, then applies `compute::filter`.
//!
//! Deliberately never emits zero-row batches: a selective predicate over
//! many input batches would otherwise flood downstream operators with empty
//! work, exactly the case the LLD calls out. This loops internally over the
//! child stream until it has a non-empty result or the child is exhausted.

use std::any::Any;
use std::sync::Arc;

use super::plan::{BatchStream, ExecutionPlan};
use crate::array::array::{as_boolean, Array};
use crate::batch::ColumnarBatch;
use crate::compute::{filter, ColumnarValue};
use crate::error::{BasaltError, Result};
use crate::physical_expr::PhysicalExprRef;
use crate::types::schema::SchemaRef;

#[derive(Debug)]
pub struct FilterExec {
    input: Arc<dyn ExecutionPlan>,
    predicate: PhysicalExprRef,
}

impl FilterExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, predicate: PhysicalExprRef) -> Self {
        FilterExec { input, predicate }
    }
}

impl ExecutionPlan for FilterExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![Arc::clone(&self.input)]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.as_slice() {
            [child] => Ok(Arc::new(FilterExec::new(
                Arc::clone(child),
                Arc::clone(&self.predicate),
            ))),
            other => Err(BasaltError::Internal(format!(
                "FilterExec takes exactly 1 child, got {}",
                other.len()
            ))),
        }
    }

    fn execute(&self, partition: usize) -> Result<BatchStream> {
        let input = self.input.execute(partition)?;
        let predicate = Arc::clone(&self.predicate);
        let iter = input.filter_map(move |batch_result| {
            let batch = match batch_result {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };
            match apply_filter(&predicate, &batch) {
                Ok(Some(filtered)) => Some(Ok(filtered)),
                Ok(None) => None, // empty result: skip, don't emit a zero-row batch
                Err(e) => Some(Err(e)),
            }
        });
        Ok(Box::new(iter))
    }
}

fn apply_filter(
    predicate: &PhysicalExprRef,
    batch: &ColumnarBatch,
) -> Result<Option<ColumnarBatch>> {
    let evaluated = predicate.evaluate(batch)?;
    let predicate_array = match evaluated {
        ColumnarValue::Array(a) => a,
        ColumnarValue::Scalar(s) => {
            // A scalar predicate (e.g. `WHERE TRUE`) applies to every row alike.
            let matched = matches!(s, crate::scalar::ScalarValue::Boolean(Some(true)));
            return Ok(if matched && batch.num_rows() > 0 {
                Some(batch.clone())
            } else {
                None
            });
        }
    };
    let predicate_array = as_boolean(predicate_array.as_ref())?;
    if predicate_array.len() != batch.num_rows() {
        return Err(BasaltError::Internal(format!(
            "predicate length {} does not match batch length {}",
            predicate_array.len(),
            batch.num_rows()
        )));
    }

    let columns = (0..batch.num_columns())
        .map(|i| {
            let col = batch
                .column(i)
                .ok_or_else(|| BasaltError::Internal(format!("column index {i} out of bounds")))?;
            filter::filter(col.as_ref(), predicate_array)
        })
        .collect::<Result<Vec<_>>>()?;

    let num_rows = columns.first().map_or(0, |c| c.len());
    if num_rows == 0 {
        return Ok(None);
    }
    Ok(Some(ColumnarBatch::try_new(
        batch.schema().clone(),
        columns,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::physical_expr::binary::BinaryExpr;
    use crate::physical_expr::column::ColumnExpr;
    use crate::physical_expr::literal::LiteralExpr;
    use crate::physical_plan::scan::MemoryScanExec;
    use crate::scalar::ScalarValue;
    use crate::types::coercion::BinaryOp;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap())
    }

    fn batch(values: &[i64]) -> ColumnarBatch {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            b.append_value(v);
        }
        ColumnarBatch::try_new(schema(), vec![Arc::new(b.finish())]).unwrap()
    }

    #[test]
    fn filters_each_batch_and_skips_empty_results() {
        let scan = Arc::new(MemoryScanExec::new(
            schema(),
            vec![batch(&[1, 2, 3]), batch(&[10, 20])],
        ));
        // a > 15: batch 1 matches nothing, batch 2 matches [20].
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(ColumnExpr::new(0)),
            BinaryOp::Gt,
            Arc::new(LiteralExpr::new(ScalarValue::Int64(Some(15)))),
        ));
        let exec = FilterExec::new(scan, predicate);
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        // The empty-result batch must be skipped entirely, not emitted as
        // a zero-row batch.
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
    }

    #[test]
    fn scalar_true_predicate_passes_every_row() {
        let scan = Arc::new(MemoryScanExec::new(schema(), vec![batch(&[1, 2])]));
        let predicate = Arc::new(LiteralExpr::new(ScalarValue::Boolean(Some(true))));
        let exec = FilterExec::new(scan, predicate);
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2);
    }

    #[test]
    fn scalar_false_predicate_yields_no_batches() {
        let scan = Arc::new(MemoryScanExec::new(schema(), vec![batch(&[1, 2])]));
        let predicate = Arc::new(LiteralExpr::new(ScalarValue::Boolean(Some(false))));
        let exec = FilterExec::new(scan, predicate);
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(batches.is_empty());
    }
}
