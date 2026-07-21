//! `MemoryScanExec` — the source operator for pre-loaded, in-memory
//! batches. CSV/Parquet-backed scans live in their own modules (`io::csv`'s
//! Phase 2 path, `io::parquet`) and produce the same `ExecutionPlan`.

use std::any::Any;
use std::sync::Arc;

use super::plan::{BatchStream, ExecutionPlan};
use crate::batch::ColumnarBatch;
use crate::error::{BasaltError, Result};
use crate::logical_plan::TableSource;
use crate::types::schema::SchemaRef;

/// A `TableSource` backed by batches already sitting in memory — the only
/// source Phase 2's core plumbing needs to exercise scan → filter →
/// projection → limit end to end without a real file format.
#[derive(Debug)]
pub struct MemoryTableSource {
    schema: SchemaRef,
    batches: Vec<ColumnarBatch>,
}

impl MemoryTableSource {
    pub fn new(schema: SchemaRef, batches: Vec<ColumnarBatch>) -> Self {
        MemoryTableSource { schema, batches }
    }

    pub fn batches(&self) -> &[ColumnarBatch] {
        &self.batches
    }
}

impl TableSource for MemoryTableSource {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[derive(Debug)]
pub struct MemoryScanExec {
    schema: SchemaRef,
    batches: Vec<ColumnarBatch>,
}

impl MemoryScanExec {
    pub fn new(schema: SchemaRef, batches: Vec<ColumnarBatch>) -> Self {
        MemoryScanExec { schema, batches }
    }
}

impl ExecutionPlan for MemoryScanExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(BasaltError::Internal(format!(
                "MemoryScanExec takes 0 children, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(MemoryScanExec::new(
            self.schema.clone(),
            self.batches.clone(),
        )))
    }

    fn execute(&self, partition: usize) -> Result<BatchStream> {
        if partition != 0 {
            return Err(BasaltError::Internal(format!(
                "MemoryScanExec has 1 partition, got request for partition {partition}"
            )));
        }
        // Cloning a batch is Arc bumps only (see ColumnarBatch's doc comment
        // on that point) — this iterator doesn't copy any column data.
        let batches: Vec<Result<ColumnarBatch>> = self.batches.iter().cloned().map(Ok).collect();
        Ok(Box::new(batches.into_iter()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
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
    fn execute_yields_every_batch_in_order() {
        let exec = MemoryScanExec::new(schema(), vec![batch(&[1, 2]), batch(&[3])]);
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 2);
        assert_eq!(batches[1].num_rows(), 1);
    }

    #[test]
    fn execute_on_invalid_partition_errors() {
        let exec = MemoryScanExec::new(schema(), vec![]);
        assert!(exec.execute(1).is_err());
    }

    #[test]
    fn scanning_zero_batches_yields_an_empty_stream() {
        let exec = MemoryScanExec::new(schema(), vec![]);
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(batches.is_empty());
    }
}
