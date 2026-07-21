//! `SortExec`/`TopKExec` — full sort and bounded-heap `ORDER BY ... LIMIT k`.
//! See design-docs/basalt-phase2-lld.md §8.1–8.2.
//!
//! Both are **pipeline breakers** in the sense that a row's final position
//! can't be fixed until everything it might be compared against has been
//! seen. `SortExec` buffers its whole input (documented, same as
//! `AggregateExec`/`HashJoinExec`); `TopKExec` does not — see its own doc
//! comment.
//!
//! The comparator is column-wise, not the LLD's normalized order-preserving
//! row-key encoding (2–5× faster per the LLD) — correctness first at a
//! bug-dense area of this phase, consistent with the same call made for
//! `Bitmap`/Kleene logic elsewhere in this codebase. The row-key path can
//! reuse `aggregate::group_keys::GroupKeyEncoder` almost as-is, which is
//! why that encoder was built order-preserving in the first place.

use std::any::Any;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use super::plan::{BatchStream, ExecutionPlan};
use crate::array::array::{as_boolean, as_primitive, as_string, Array, ArrayRef};
use crate::array::boolean::BooleanBuilder;
use crate::array::primitive::PrimitiveBuilder;
use crate::array::string::StringBuilder;
use crate::array::types::{Float64Type, Int64Type};
use crate::batch::ColumnarBatch;
use crate::compute::sort::{lexsort_to_indices, SortColumn, SortOptions};
use crate::compute::{concat, take};
use crate::error::{BasaltError, Result};
use crate::physical_expr::PhysicalExprRef;
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;
use crate::types::schema::SchemaRef;

#[derive(Clone)]
pub struct PhysicalSortExpr {
    pub expr: PhysicalExprRef,
    pub options: SortOptions,
}

impl std::fmt::Debug for PhysicalSortExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalSortExpr")
            .field("options", &self.options)
            .finish()
    }
}

#[derive(Debug)]
pub struct SortExec {
    input: Arc<dyn ExecutionPlan>,
    exprs: Vec<PhysicalSortExpr>,
}

impl SortExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, exprs: Vec<PhysicalSortExpr>) -> Self {
        SortExec { input, exprs }
    }

    /// Consumes the whole child stream and returns the single sorted batch
    /// (or `None` for zero input rows).
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

/// A single value extracted from a column at a row, carrying enough type
/// information to compare and to rebuild an output array later. `Null`
/// sorts per `SortOptions::nulls_first`, exactly like `compute::sort`'s
/// array-level comparator — this is the same policy re-expressed at the
/// scalar level so `TopKExec` and `SortExec` agree on ordering.
fn scalar_at(array: &dyn Array, row: usize) -> Result<ScalarValue> {
    if array.is_null(row) {
        return Ok(match array.data_type() {
            DataType::Int64 => ScalarValue::Int64(None),
            DataType::Float64 => ScalarValue::Float64(None),
            DataType::Utf8 => ScalarValue::Utf8(None),
            DataType::Boolean => ScalarValue::Boolean(None),
        });
    }
    Ok(match array.data_type() {
        DataType::Int64 => ScalarValue::Int64(Some(as_primitive::<Int64Type>(array)?.value(row))),
        DataType::Float64 => {
            ScalarValue::Float64(Some(as_primitive::<Float64Type>(array)?.value(row)))
        }
        DataType::Utf8 => ScalarValue::Utf8(Some(as_string(array)?.value(row).to_string())),
        DataType::Boolean => ScalarValue::Boolean(Some(as_boolean(array)?.value(row))),
    })
}

/// Compares two same-typed scalars under one sort key's options — the
/// scalar-level twin of `compute::sort`'s array-index comparator, used here
/// because `TopKExec`'s heap holds already-extracted values, not array
/// positions.
fn compare_scalars(a: &ScalarValue, b: &ScalarValue, options: SortOptions) -> Ordering {
    let (a_null, b_null) = (a.is_null(), b.is_null());
    if a_null || b_null {
        return match (a_null, b_null) {
            (true, true) => Ordering::Equal,
            (true, false) => {
                if options.nulls_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            (false, true) => {
                if options.nulls_first {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            (false, false) => unreachable!(),
        };
    }
    let ord = match (a, b) {
        (ScalarValue::Int64(Some(x)), ScalarValue::Int64(Some(y))) => x.cmp(y),
        (ScalarValue::Float64(Some(x)), ScalarValue::Float64(Some(y))) => {
            match (x.is_nan(), y.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
            }
        }
        (ScalarValue::Utf8(Some(x)), ScalarValue::Utf8(Some(y))) => x.cmp(y),
        (ScalarValue::Boolean(Some(x)), ScalarValue::Boolean(Some(y))) => x.cmp(y),
        _ => Ordering::Equal, // mismatched types: shouldn't happen post-binder; treat as tied.
    };
    if options.descending {
        ord.reverse()
    } else {
        ord
    }
}

/// One retained candidate row in `TopKExec`'s heap: its sort keys (for
/// comparison) and its full output row (to rebuild the result at the end).
struct HeapEntry {
    keys: Vec<ScalarValue>,
    row: Vec<ScalarValue>,
    options: Vec<SortOptions>,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in 0..self.keys.len() {
            let ord = compare_scalars(&self.keys[i], &other.keys[i], self.options[i]);
            if ord != Ordering::Equal {
                return ord;
            }
        }
        Ordering::Equal
    }
}

fn scalars_to_array(values: &[ScalarValue], data_type: DataType) -> Result<ArrayRef> {
    Ok(match data_type {
        DataType::Int64 => {
            let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
            for v in values {
                match v {
                    ScalarValue::Int64(Some(x)) => b.append_value(*x),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Float64 => {
            let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(values.len());
            for v in values {
                match v {
                    ScalarValue::Float64(Some(x)) => b.append_value(*x),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Utf8 => {
            let mut b = StringBuilder::with_capacity(values.len(), 0);
            for v in values {
                match v {
                    ScalarValue::Utf8(Some(x)) => b.append_value(x)?,
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Boolean => {
            let mut b = BooleanBuilder::with_capacity(values.len());
            for v in values {
                match v {
                    ScalarValue::Boolean(Some(x)) => b.append_value(*x),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
    })
}

/// `ORDER BY ... LIMIT k`. A bounded max-heap of size `k`: every row is
/// pushed, and once the heap exceeds `k` entries the worst one (by the same
/// comparator the final output uses) is popped and discarded. Memory is
/// `O(k)` and time is `O(n log k)`, versus `SortExec`'s `O(n)`/`O(n log n)` —
/// for `k` small relative to `n` (the case this operator exists for) this
/// is not a marginal win: it's the difference between buffering the whole
/// input and holding a constant-size heap.
///
/// Streams its child directly rather than going through `SortExec` — unlike
/// the version this replaced, `TopKExec` never buffers more than `k` rows
/// plus the current input batch at a time.
#[derive(Debug)]
pub struct TopKExec {
    input: Arc<dyn ExecutionPlan>,
    exprs: Vec<PhysicalSortExpr>,
    k: usize,
}

impl TopKExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, exprs: Vec<PhysicalSortExpr>, k: usize) -> Self {
        TopKExec { input, exprs, k }
    }
}

impl ExecutionPlan for TopKExec {
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
            [child] => Ok(Arc::new(TopKExec::new(
                Arc::clone(child),
                self.exprs.clone(),
                self.k,
            ))),
            other => Err(BasaltError::Internal(format!(
                "TopKExec takes exactly 1 child, got {}",
                other.len()
            ))),
        }
    }

    fn execute(&self, partition: usize) -> Result<BatchStream> {
        let schema = self.input.schema();
        let num_columns = schema.len();
        let options: Vec<SortOptions> = self.exprs.iter().map(|se| se.options).collect();

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(self.k + 1);

        if self.k > 0 {
            for batch_result in self.input.execute(partition)? {
                let batch = batch_result?;
                let num_rows = batch.num_rows();
                if num_rows == 0 {
                    continue;
                }
                let key_arrays: Vec<ArrayRef> = self
                    .exprs
                    .iter()
                    .map(|se| se.expr.evaluate(&batch)?.into_array(num_rows))
                    .collect::<Result<Vec<_>>>()?;

                for row in 0..num_rows {
                    let keys = key_arrays
                        .iter()
                        .map(|a| scalar_at(a.as_ref(), row))
                        .collect::<Result<Vec<_>>>()?;
                    let full_row = (0..num_columns)
                        .map(|c| scalar_at(batch.column(c).unwrap().as_ref(), row))
                        .collect::<Result<Vec<_>>>()?;
                    heap.push(HeapEntry {
                        keys,
                        row: full_row,
                        options: options.clone(),
                    });
                    if heap.len() > self.k {
                        heap.pop(); // discard the current worst-of-the-best-k-so-far
                    }
                }
            }
        }

        if heap.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        // The heap yields its max (worst-under-the-final-order) first;
        // popping all of them and reversing recovers the correct final order.
        let mut rows: Vec<Vec<ScalarValue>> = Vec::with_capacity(heap.len());
        while let Some(entry) = heap.pop() {
            rows.push(entry.row);
        }
        rows.reverse();

        let mut columns = Vec::with_capacity(num_columns);
        for col_idx in 0..num_columns {
            let data_type = schema.field(col_idx).unwrap().data_type;
            let values: Vec<ScalarValue> = rows.iter().map(|r| r[col_idx].clone()).collect();
            columns.push(scalars_to_array(&values, data_type)?);
        }
        Ok(Box::new(std::iter::once(ColumnarBatch::try_new(
            schema.clone(),
            columns,
        ))))
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
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]).unwrap())
    }

    fn batch(values: &[Option<i64>]) -> ColumnarBatch {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            match v {
                Some(v) => b.append_value(v),
                None => b.append_null(),
            }
        }
        ColumnarBatch::try_new(schema(), vec![Arc::new(b.finish())]).unwrap()
    }

    fn values_of(b: &ColumnarBatch) -> Vec<i64> {
        let col = as_primitive::<Int64Type>(b.column(0).unwrap().as_ref()).unwrap();
        (0..b.num_rows()).map(|i| col.value(i)).collect()
    }

    fn asc() -> SortOptions {
        SortOptions {
            descending: false,
            nulls_first: true,
        }
    }

    #[test]
    fn sorts_across_multiple_input_batches() {
        let scan = Arc::new(MemoryScanExec::new(
            schema(),
            vec![
                batch(&[Some(3), Some(1)]),
                batch(&[Some(2), Some(5), Some(4)]),
            ],
        ));
        let exec = SortExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: asc(),
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
                options: asc(),
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
    fn topk_keeps_the_k_smallest_values_in_order() {
        let scan = Arc::new(MemoryScanExec::new(
            schema(),
            vec![batch(&[Some(5), Some(3), Some(1), Some(4), Some(2)])],
        ));
        let exec = TopKExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: asc(),
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
    fn topk_across_multiple_batches_still_finds_the_global_top_k() {
        let scan = Arc::new(MemoryScanExec::new(
            schema(),
            vec![
                batch(&[Some(10), Some(1)]),
                batch(&[Some(2), Some(9)]),
                batch(&[Some(3)]),
            ],
        ));
        let exec = TopKExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: asc(),
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
    fn topk_with_k_larger_than_input_returns_everything_sorted() {
        let scan = Arc::new(MemoryScanExec::new(
            schema(),
            vec![batch(&[Some(2), Some(1)])],
        ));
        let exec = TopKExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: asc(),
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

    #[test]
    fn topk_descending_keeps_the_k_largest() {
        let scan = Arc::new(MemoryScanExec::new(
            schema(),
            vec![batch(&[Some(1), Some(5), Some(3), Some(2), Some(4)])],
        ));
        let desc = SortOptions {
            descending: true,
            nulls_first: true,
        };
        let exec = TopKExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: desc,
            }],
            2,
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(values_of(&batches[0]), vec![5, 4]);
    }

    #[test]
    fn topk_respects_null_ordering_policy() {
        let scan = Arc::new(MemoryScanExec::new(
            schema(),
            vec![batch(&[Some(5), None, Some(1)])],
        ));
        let exec = TopKExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: asc(),
            }],
            2,
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        // nulls_first: the null is the smallest value and must be retained
        // ahead of 5 when k=2 keeps [NULL, 1].
        assert!(batches[0].column(0).unwrap().is_null(0));
    }

    #[test]
    fn topk_with_k_zero_yields_no_rows() {
        let scan = Arc::new(MemoryScanExec::new(
            schema(),
            vec![batch(&[Some(1), Some(2)])],
        ));
        let exec = TopKExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: asc(),
            }],
            0,
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(batches.is_empty());
    }

    #[test]
    fn topk_on_empty_input_yields_no_rows() {
        let scan = Arc::new(MemoryScanExec::new(schema(), vec![]));
        let exec = TopKExec::new(
            scan,
            vec![PhysicalSortExpr {
                expr: Arc::new(ColumnExpr::new(0)),
                options: asc(),
            }],
            5,
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(batches.is_empty());
    }
}
