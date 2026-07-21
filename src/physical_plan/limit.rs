//! `LimitExec` — `OFFSET`/`LIMIT`. Slices the batch that crosses either
//! boundary and stops pulling from its child as soon as `fetch` rows have
//! been produced, rather than draining the whole input first.

use std::any::Any;
use std::sync::Arc;

use super::plan::{BatchStream, ExecutionPlan};
use crate::batch::ColumnarBatch;
use crate::error::{BasaltError, Result};
use crate::types::schema::SchemaRef;

#[derive(Debug)]
pub struct LimitExec {
    input: Arc<dyn ExecutionPlan>,
    skip: usize,
    fetch: Option<usize>,
}

impl LimitExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, skip: usize, fetch: Option<usize>) -> Self {
        LimitExec { input, skip, fetch }
    }
}

impl ExecutionPlan for LimitExec {
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
            [child] => Ok(Arc::new(LimitExec::new(
                Arc::clone(child),
                self.skip,
                self.fetch,
            ))),
            other => Err(BasaltError::Internal(format!(
                "LimitExec takes exactly 1 child, got {}",
                other.len()
            ))),
        }
    }

    fn execute(&self, partition: usize) -> Result<BatchStream> {
        Ok(Box::new(LimitStream {
            input: self.input.execute(partition)?,
            remaining_skip: self.skip,
            remaining_fetch: self.fetch,
            done: false,
        }))
    }
}

struct LimitStream {
    input: BatchStream,
    remaining_skip: usize,
    remaining_fetch: Option<usize>,
    done: bool,
}

impl Iterator for LimitStream {
    type Item = Result<ColumnarBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.remaining_fetch == Some(0) {
            return None;
        }
        loop {
            let batch = match self.input.next()? {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };
            let mut num_rows = batch.num_rows();

            // Consume the skip out of this batch first; a batch entirely
            // within the skip window contributes nothing and we pull again.
            let batch = if self.remaining_skip > 0 {
                if self.remaining_skip >= num_rows {
                    self.remaining_skip -= num_rows;
                    continue;
                }
                let b = batch.slice(self.remaining_skip, num_rows - self.remaining_skip);
                self.remaining_skip = 0;
                b
            } else {
                batch
            };
            num_rows = batch.num_rows();
            if num_rows == 0 {
                continue;
            }

            let batch = match self.remaining_fetch {
                Some(remaining) if remaining < num_rows => {
                    self.done = true;
                    batch.slice(0, remaining)
                }
                Some(remaining) => {
                    self.remaining_fetch = Some(remaining - num_rows);
                    batch
                }
                None => batch,
            };
            return Some(Ok(batch));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_primitive;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::physical_plan::scan::MemoryScanExec;
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

    fn values_of(b: &ColumnarBatch) -> Vec<i64> {
        let col = as_primitive::<Int64Type>(b.column(0).unwrap().as_ref()).unwrap();
        (0..b.num_rows()).map(|i| col.value(i)).collect()
    }

    #[test]
    fn fetch_slices_the_crossing_batch() {
        let scan = Arc::new(MemoryScanExec::new(
            schema(),
            vec![batch(&[1, 2, 3]), batch(&[4, 5])],
        ));
        let exec = LimitExec::new(scan, 0, Some(4));
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let all: Vec<i64> = batches.iter().flat_map(values_of).collect();
        assert_eq!(all, vec![1, 2, 3, 4]);
    }

    #[test]
    fn skip_consumes_whole_batches_and_slices_the_crossing_one() {
        let scan = Arc::new(MemoryScanExec::new(
            schema(),
            vec![batch(&[1, 2]), batch(&[3, 4, 5])],
        ));
        let exec = LimitExec::new(scan, 3, None);
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let all: Vec<i64> = batches.iter().flat_map(values_of).collect();
        assert_eq!(all, vec![4, 5]);
    }

    #[test]
    fn skip_and_fetch_combine() {
        let scan = Arc::new(MemoryScanExec::new(schema(), vec![batch(&[1, 2, 3, 4, 5])]));
        let exec = LimitExec::new(scan, 1, Some(2));
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let all: Vec<i64> = batches.iter().flat_map(values_of).collect();
        assert_eq!(all, vec![2, 3]);
    }

    #[test]
    fn fetch_zero_yields_no_rows() {
        let scan = Arc::new(MemoryScanExec::new(schema(), vec![batch(&[1, 2, 3])]));
        let exec = LimitExec::new(scan, 0, Some(0));
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(batches.is_empty());
    }

    #[test]
    fn fetch_beyond_input_returns_everything() {
        let scan = Arc::new(MemoryScanExec::new(schema(), vec![batch(&[1, 2])]));
        let exec = LimitExec::new(scan, 0, Some(100));
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let all: Vec<i64> = batches.iter().flat_map(values_of).collect();
        assert_eq!(all, vec![1, 2]);
    }
}
