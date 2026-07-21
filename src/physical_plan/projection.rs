//! `ProjectionExec` — evaluates each output expression over the batch and
//! assembles a new batch from the results.

use std::any::Any;
use std::sync::Arc;

use super::plan::{BatchStream, ExecutionPlan};
use crate::batch::ColumnarBatch;
use crate::error::{BasaltError, Result};
use crate::physical_expr::PhysicalExprRef;
use crate::types::schema::SchemaRef;

#[derive(Debug)]
pub struct ProjectionExec {
    input: Arc<dyn ExecutionPlan>,
    exprs: Vec<PhysicalExprRef>,
    schema: SchemaRef,
}

impl ProjectionExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        exprs: Vec<PhysicalExprRef>,
        schema: SchemaRef,
    ) -> Self {
        ProjectionExec {
            input,
            exprs,
            schema,
        }
    }
}

impl ExecutionPlan for ProjectionExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![Arc::clone(&self.input)]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.as_slice() {
            [child] => Ok(Arc::new(ProjectionExec::new(
                Arc::clone(child),
                self.exprs.clone(),
                self.schema.clone(),
            ))),
            other => Err(BasaltError::Internal(format!(
                "ProjectionExec takes exactly 1 child, got {}",
                other.len()
            ))),
        }
    }

    fn execute(&self, partition: usize) -> Result<BatchStream> {
        let input = self.input.execute(partition)?;
        let exprs = self.exprs.clone();
        let schema = self.schema.clone();
        let iter = input.map(move |batch_result| {
            let batch = batch_result?;
            let num_rows = batch.num_rows();
            let columns = exprs
                .iter()
                .map(|e| e.evaluate(&batch)?.into_array(num_rows))
                .collect::<Result<Vec<_>>>()?;
            ColumnarBatch::try_new(schema.clone(), columns)
        });
        Ok(Box::new(iter))
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
    use crate::physical_expr::literal::LiteralExpr;
    use crate::physical_plan::scan::MemoryScanExec;
    use crate::scalar::ScalarValue;
    use crate::types::coercion::BinaryOp;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema};

    fn input_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap())
    }

    fn batch(values: &[i64]) -> ColumnarBatch {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            b.append_value(v);
        }
        ColumnarBatch::try_new(input_schema(), vec![Arc::new(b.finish())]).unwrap()
    }

    #[test]
    fn projects_a_computed_expression() {
        let scan = Arc::new(MemoryScanExec::new(input_schema(), vec![batch(&[1, 2, 3])]));
        let output_schema =
            Arc::new(Schema::new(vec![Field::new("a_plus_1", DataType::Int64, false)]).unwrap());
        let expr: PhysicalExprRef = Arc::new(BinaryExpr::new(
            Arc::new(ColumnExpr::new(0)),
            BinaryOp::Add,
            Arc::new(LiteralExpr::new(ScalarValue::Int64(Some(1)))),
        ));
        let exec = ProjectionExec::new(scan, vec![expr], output_schema);
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        let col = as_primitive::<Int64Type>(batches[0].column(0).unwrap().as_ref()).unwrap();
        assert_eq!(col.value(0), 2);
        assert_eq!(col.value(2), 4);
    }

    #[test]
    fn schema_matches_the_declared_projection_schema() {
        let scan = Arc::new(MemoryScanExec::new(input_schema(), vec![]));
        let output_schema =
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]).unwrap());
        let exec = ProjectionExec::new(
            scan,
            vec![Arc::new(ColumnExpr::new(0))],
            output_schema.clone(),
        );
        assert_eq!(exec.schema(), output_schema);
    }
}
