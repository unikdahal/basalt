//! `SortExec`/`TopKExec` — full sort and (for now) sort-then-truncate
//! `ORDER BY ... LIMIT k`. See design-docs/basalt-phase2-lld.md §8.1–8.2.
//!
//! Both are **pipeline breakers**: sorting requires every row that might
//! belong anywhere in the output to have been seen first.
//!
//! **`TopKExec` here is not yet the LLD's bounded-heap `O(n log k)` /
//! `O(k)`-memory algorithm** — it's a full sort followed by a truncate,
//! which is correct but pays `O(n log n)` time and buffers the whole input
//! just like `SortExec` does. For `k` small relative to `n` (the case this
//! operator exists for) the bounded heap is a real, worthwhile win; it's a
//! documented follow-up rather than a silently-assumed optimization, per
//! this project's "measure before claiming a speedup" discipline. Detecting
//! the `Sort` + `Limit` pattern and choosing which `ExecutionPlan` to build
//! is itself deferred to the physical planner picking one over the other
//! once that heap exists — today the planner always builds `SortExec` (or,
//! for `Sort` beneath a `Limit`, still just `SortExec` composed with the
//! existing `LimitExec`), so `TopKExec` is not yet wired into `planner.rs`.

use std::any::Any;
use std::sync::Arc;

use super::plan::{BatchStream, ExecutionPlan};
use crate::array::array::ArrayRef;
use crate::batch::ColumnarBatch;
use crate::compute::sort::{lexsort_to_indices, SortColumn, SortOptions};
use crate::compute::{concat, take};
use crate::error::{BasaltError, Result};
use crate::physical_expr::PhysicalExprRef;
use crate::types::schema::SchemaRef;

#[derive(Clone)]
pub struct PhysicalSortExpr {
    pub expr: PhysicalExprRef,
    pub options: SortOptions,
}

#[derive(Debug)]
pub struct SortExec {
    input: Arc<dyn ExecutionPlan>,
    exprs: Vec<PhysicalSortExpr>,
}

impl std::fmt::Debug for PhysicalSortExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalSortExpr")
            .field("options", &self.options)
            .finish()
    }
}

impl SortExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, exprs: Vec<PhysicalSortExpr>) -> Self {
        SortExec { input, exprs }
    }

    /// Consumes the whole child stream and returns the single sorted batch
    /// (or `None` for zero input rows) — shared by `SortExec::execute` and
    /// `TopKExec`'s sort-then-truncate implementation.
    fn collect_sorted(&self, partition: usize) -> Result<Option<ColumnarBatch>> {
        let schema = self.input.schema();
        let all_batches: Vec<ColumnarBatch> =
            self.input.execute(partition)?.collect::<Result<Vec<_>>>()?;
        if all_batches.is_empty() {
            return Ok(None);
        }

        let num_columns = schema.len();
        let mut concatenated = Vec::with_capacity(num_columns);
        for col_idx in 0..num_columns {
            let parts: Vec<ArrayRef> = all_batches
                .iter()
                .map(|b| Arc::clone(b.column(col_idx).unwrap()))
                .collect();
            concatenated.push(concat::concat(&parts)?);
        }
        let whole_batch = ColumnarBatch::try_new(schema.clone(), concatenated)?;
        let num_rows = whole_batch.num_rows();
        if num_rows == 0 {
            return Ok(None);
        }

        let sort_columns = self
            .exprs
            .iter()
            .map(|se| {
                let values = se.expr.evaluate(&whole_batch)?.into_array(num_rows)?;
                Ok(SortColumn {
                    values,
                    options: se.options,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let indices = lexsort_to_indices(&sort_columns)?;

        let sorted_columns = (0..num_columns)
            .map(|i| take::take(whole_batch.column(i).unwrap().as_ref(), &indices))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(ColumnarBatch::try_new(
            schema.clone(),
            sorted_columns,
        )?))
    }
}

impl ExecutionPlan for SortExec {
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
            [child] => Ok(Arc::new(SortExec::new(
                Arc::clone(child),
                self.exprs.clone(),
            ))),
            other => Err(BasaltError::Internal(format!(
                "SortExec takes exactly 1 child, got {}",
                other.len()
            ))),
        }
    }

    fn execute(&self, partition: usize) -> Result<BatchStream> {
        match self.collect_sorted(partition)? {
            Some(batch) => Ok(Box::new(std::iter::once(Ok(batch)))),
            None => Ok(Box::new(std::iter::empty())),
        }
    }
}

/// `ORDER BY ... LIMIT k`. See the module doc comment: this is currently a
/// full sort truncated to `k`, not yet the bounded-heap `O(n log k)`
/// algorithm the LLD describes.
#[derive(Debug)]
pub struct TopKExec {
    sort: SortExec,
    k: usize,
}

impl TopKExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, exprs: Vec<PhysicalSortExpr>, k: usize) -> Self {
        TopKExec {
            sort: SortExec::new(input, exprs),
            k,
        }
    }
}

impl ExecutionPlan for TopKExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.sort.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        self.sort.children()
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.as_slice() {
            [child] => Ok(Arc::new(TopKExec::new(
                Arc::clone(child),
                self.sort.exprs.clone(),
                self.k,
            ))),
            other => Err(BasaltError::Internal(format!(
                "TopKExec takes exactly 1 child, got {}",
                other.len()
            ))),
        }
    }

    fn execute(&self, partition: usize) -> Result<BatchStream> {
        match self.sort.collect_sorted(partition)? {
            Some(batch) => {
                let k = self.k.min(batch.num_rows());
                Ok(Box::new(std::iter::once(Ok(batch.slice(0, k)))))
            }
            None => Ok(Box::new(std::iter::empty())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_primitive;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::physical_expr::column::ColumnExpr;
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
    fn sorts_across_multiple_input_batches() {
        let scan = Arc::new(MemoryScanExec::new(
            schema(),
            vec![batch(&[3, 1]), batch(&[2, 5, 4])],
        ));
        let exec = SortExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }],
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(values_of(&batches[0]), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn sort_on_empty_input_yields_no_batches() {
        let scan = Arc::new(MemoryScanExec::new(schema(), vec![]));
        let exec = SortExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }],
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(batches.is_empty());
    }

    #[test]
    fn topk_truncates_the_sorted_output() {
        let scan = Arc::new(MemoryScanExec::new(schema(), vec![batch(&[5, 3, 1, 4, 2])]));
        let exec = TopKExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }],
            3,
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(values_of(&batches[0]), vec![1, 2, 3]);
    }

    #[test]
    fn topk_with_k_larger_than_input_returns_everything() {
        let scan = Arc::new(MemoryScanExec::new(schema(), vec![batch(&[2, 1])]));
        let exec = TopKExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }],
            100,
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(values_of(&batches[0]), vec![1, 2]);
    }
}
