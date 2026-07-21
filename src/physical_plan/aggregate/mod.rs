//! `AggregateExec` — the aggregation operator. See
//! design-docs/basalt-phase2-lld.md §6.
//!
//! A **pipeline breaker**: unlike `FilterExec`/`ProjectionExec`/`LimitExec`,
//! this must consume its entire input before it can emit anything (a group
//! can't be finalized until every row that might belong to it has been
//! seen). `execute` therefore drains the child stream eagerly and returns a
//! stream of exactly one batch.
//!
//! Implements the no-group case (`SELECT SUM(x) FROM t`, no `GROUP BY`)
//! separately from the hashed path, per the LLD: a single accumulator set
//! with no hashing at all, since it's both trivial and very common.

pub mod accumulator;
pub mod group_keys;
pub mod hash_table;

use std::any::Any;
use std::sync::Arc;

use self::accumulator::{
    Accumulator, AvgAccumulator, CountAccumulator, MinMaxAccumulator, SumAccumulator,
};
use self::hash_table::GroupedHashAggregator;
use super::plan::{BatchStream, ExecutionPlan};
use crate::array::array::ArrayRef;
use crate::array::boolean::BooleanBuilder;
use crate::array::primitive::PrimitiveBuilder;
use crate::array::string::StringBuilder;
use crate::array::types::{Float64Type, Int64Type};
use crate::batch::ColumnarBatch;
use crate::error::{BasaltError, Result};
use crate::logical_plan::AggregateKind;
use crate::physical_expr::PhysicalExprRef;
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;
use crate::types::schema::SchemaRef;

#[derive(Debug)]
pub struct AggregateExec {
    input: Arc<dyn ExecutionPlan>,
    group_exprs: Vec<PhysicalExprRef>,
    group_types: Vec<DataType>,
    /// `None` for `COUNT(*)`, which counts rows rather than a column.
    agg_arg_exprs: Vec<Option<PhysicalExprRef>>,
    agg_kinds: Vec<AggregateKind>,
    /// The type of each aggregate's *argument* — needed to build a
    /// `SumAccumulator`/`MinMaxAccumulator` (their state is typed); `Count`
    /// and `Avg` ignore this.
    agg_arg_types: Vec<DataType>,
    schema: SchemaRef,
}

impl AggregateExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        group_exprs: Vec<PhysicalExprRef>,
        group_types: Vec<DataType>,
        agg_arg_exprs: Vec<Option<PhysicalExprRef>>,
        agg_kinds: Vec<AggregateKind>,
        agg_arg_types: Vec<DataType>,
        schema: SchemaRef,
    ) -> Self {
        AggregateExec {
            input,
            group_exprs,
            group_types,
            agg_arg_exprs,
            agg_kinds,
            agg_arg_types,
            schema,
        }
    }

    fn build_factories(&self) -> Vec<Box<dyn Fn() -> Box<dyn Accumulator> + Send + Sync>> {
        self.agg_kinds
            .iter()
            .zip(&self.agg_arg_types)
            .zip(&self.agg_arg_exprs)
            .map(|((&kind, &data_type), arg)| make_factory(kind, data_type, arg.is_none()))
            .collect()
    }
}

fn make_factory(
    kind: AggregateKind,
    data_type: DataType,
    is_star: bool,
) -> Box<dyn Fn() -> Box<dyn Accumulator> + Send + Sync> {
    match kind {
        AggregateKind::Sum => {
            Box::new(move || Box::new(SumAccumulator::new(data_type)) as Box<dyn Accumulator>)
        }
        AggregateKind::Count => {
            if is_star {
                Box::new(|| Box::new(CountAccumulator::star()) as Box<dyn Accumulator>)
            } else {
                Box::new(|| Box::new(CountAccumulator::expr()) as Box<dyn Accumulator>)
            }
        }
        AggregateKind::Min => {
            Box::new(move || Box::new(MinMaxAccumulator::min(data_type)) as Box<dyn Accumulator>)
        }
        AggregateKind::Max => {
            Box::new(move || Box::new(MinMaxAccumulator::max(data_type)) as Box<dyn Accumulator>)
        }
        AggregateKind::Avg => Box::new(|| Box::new(AvgAccumulator::new()) as Box<dyn Accumulator>),
    }
}

/// A stand-in input array for `COUNT(*)`: any existing column supplies the
/// right row count without caring about its contents; a schema with zero
/// columns falls back to a fresh all-null `Int64` array of the batch's
/// length so the aggregate still has *something* to read a length from.
fn row_count_carrier(batch: &ColumnarBatch) -> ArrayRef {
    if let Some(col) = batch.column(0) {
        return Arc::clone(col);
    }
    let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(batch.num_rows());
    for _ in 0..batch.num_rows() {
        b.append_null();
    }
    Arc::new(b.finish())
}

fn scalars_to_array(values: &[ScalarValue], data_type: DataType) -> Result<ArrayRef> {
    Ok(match data_type {
        DataType::Int64 => {
            let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
            for v in values {
                match v {
                    ScalarValue::Int64(Some(x)) => b.append_value(*x),
                    ScalarValue::Int64(None) => b.append_null(),
                    other => {
                        return Err(BasaltError::Internal(format!(
                            "expected Int64 aggregate output, got {other:?}"
                        )))
                    }
                }
            }
            Arc::new(b.finish())
        }
        DataType::Float64 => {
            let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(values.len());
            for v in values {
                match v {
                    ScalarValue::Float64(Some(x)) => b.append_value(*x),
                    ScalarValue::Float64(None) => b.append_null(),
                    other => {
                        return Err(BasaltError::Internal(format!(
                            "expected Float64 aggregate output, got {other:?}"
                        )))
                    }
                }
            }
            Arc::new(b.finish())
        }
        DataType::Utf8 => {
            let mut b = StringBuilder::with_capacity(values.len(), 0);
            for v in values {
                match v {
                    ScalarValue::Utf8(Some(x)) => b.append_value(x)?,
                    ScalarValue::Utf8(None) => b.append_null(),
                    other => {
                        return Err(BasaltError::Internal(format!(
                            "expected Utf8 aggregate output, got {other:?}"
                        )))
                    }
                }
            }
            Arc::new(b.finish())
        }
        DataType::Boolean => {
            let mut b = BooleanBuilder::with_capacity(values.len());
            for v in values {
                match v {
                    ScalarValue::Boolean(Some(x)) => b.append_value(*x),
                    ScalarValue::Boolean(None) => b.append_null(),
                    other => {
                        return Err(BasaltError::Internal(format!(
                            "expected Boolean aggregate output, got {other:?}"
                        )))
                    }
                }
            }
            Arc::new(b.finish())
        }
    })
}

impl ExecutionPlan for AggregateExec {
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
            [child] => Ok(Arc::new(AggregateExec::new(
                Arc::clone(child),
                self.group_exprs.clone(),
                self.group_types.clone(),
                self.agg_arg_exprs.clone(),
                self.agg_kinds.clone(),
                self.agg_arg_types.clone(),
                self.schema.clone(),
            ))),
            other => Err(BasaltError::Internal(format!(
                "AggregateExec takes exactly 1 child, got {}",
                other.len()
            ))),
        }
    }

    fn execute(&self, partition: usize) -> Result<BatchStream> {
        let no_group = self.group_exprs.is_empty();
        let mut hashed = if no_group {
            None
        } else {
            Some(GroupedHashAggregator::new(
                self.group_types.clone(),
                self.build_factories(),
            ))
        };
        let mut flat: Option<Vec<Box<dyn Accumulator>>> = if no_group {
            Some(self.build_factories().iter().map(|f| f()).collect())
        } else {
            None
        };

        for batch_result in self.input.execute(partition)? {
            let batch = batch_result?;
            let num_rows = batch.num_rows();
            if num_rows == 0 {
                continue;
            }

            let agg_arrays = self
                .agg_arg_exprs
                .iter()
                .map(|arg| match arg {
                    Some(e) => e.evaluate(&batch)?.into_array(num_rows),
                    None => Ok(row_count_carrier(&batch)),
                })
                .collect::<Result<Vec<_>>>()?;

            if let Some(accs) = &mut flat {
                for (acc, arr) in accs.iter_mut().zip(&agg_arrays) {
                    acc.update_batch(std::slice::from_ref(arr))?;
                }
            } else {
                let group_arrays = self
                    .group_exprs
                    .iter()
                    .map(|e| e.evaluate(&batch)?.into_array(num_rows))
                    .collect::<Result<Vec<_>>>()?;
                hashed
                    .as_mut()
                    .unwrap()
                    .update_batch(&group_arrays, &agg_arrays)?;
            }
        }

        let batch = if let Some(accs) = flat {
            let mut columns = Vec::with_capacity(accs.len());
            for (acc, &data_type) in accs
                .into_iter()
                .zip(self.schema.fields().iter().map(|f| &f.data_type))
            {
                columns.push(scalars_to_array(&[acc.evaluate()?], data_type)?);
            }
            ColumnarBatch::try_new(self.schema.clone(), columns)?
        } else {
            let (group_rows, agg_outputs) = hashed.unwrap().finish()?;
            let num_groups = group_rows.len();
            let mut columns = Vec::with_capacity(self.group_types.len() + agg_outputs.len());
            for (col_idx, &data_type) in self.group_types.iter().enumerate() {
                let values: Vec<ScalarValue> = (0..num_groups)
                    .map(|r| group_rows[r][col_idx].clone())
                    .collect();
                columns.push(scalars_to_array(&values, data_type)?);
            }
            for (values, field) in agg_outputs
                .into_iter()
                .zip(&self.schema.fields()[self.group_types.len()..])
            {
                columns.push(scalars_to_array(&values, field.data_type)?);
            }
            ColumnarBatch::try_new(self.schema.clone(), columns)?
        };

        Ok(Box::new(std::iter::once(Ok(batch))))
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
    use crate::types::schema::{Field, Schema};

    fn input_schema() -> SchemaRef {
        Arc::new(
            Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v", DataType::Int64, false),
            ])
            .unwrap(),
        )
    }

    fn batch(keys: &[i64], values: &[i64]) -> ColumnarBatch {
        let mut kb = PrimitiveBuilder::<Int64Type>::with_capacity(keys.len());
        for &k in keys {
            kb.append_value(k);
        }
        let mut vb = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            vb.append_value(v);
        }
        ColumnarBatch::try_new(
            input_schema(),
            vec![Arc::new(kb.finish()), Arc::new(vb.finish())],
        )
        .unwrap()
    }

    #[test]
    fn no_group_by_sum_over_multiple_batches() {
        let scan = Arc::new(MemoryScanExec::new(
            input_schema(),
            vec![batch(&[1, 1], &[10, 20]), batch(&[2], &[30])],
        ));
        let output_schema =
            Arc::new(Schema::new(vec![Field::new("total", DataType::Int64, true)]).unwrap());
        let exec = AggregateExec::new(
            scan,
            vec![],
            vec![],
            vec![Some(Arc::new(ColumnExpr::new(1)))],
            vec![AggregateKind::Sum],
            vec![DataType::Int64],
            output_schema,
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        let col = as_primitive::<Int64Type>(batches[0].column(0).unwrap().as_ref()).unwrap();
        assert_eq!(col.value(0), 60);
    }

    #[test]
    fn group_by_sum_across_multiple_batches() {
        let scan = Arc::new(MemoryScanExec::new(
            input_schema(),
            vec![batch(&[1, 2], &[10, 20]), batch(&[1], &[5])],
        ));
        let output_schema = Arc::new(
            Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("total", DataType::Int64, true),
            ])
            .unwrap(),
        );
        let exec = AggregateExec::new(
            scan,
            vec![Arc::new(ColumnExpr::new(0))],
            vec![DataType::Int64],
            vec![Some(Arc::new(ColumnExpr::new(1)))],
            vec![AggregateKind::Sum],
            vec![DataType::Int64],
            output_schema,
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 2);

        let keys = as_primitive::<Int64Type>(batches[0].column(0).unwrap().as_ref()).unwrap();
        let sums = as_primitive::<Int64Type>(batches[0].column(1).unwrap().as_ref()).unwrap();
        let mut by_key = std::collections::HashMap::new();
        for i in 0..2 {
            by_key.insert(keys.value(i), sums.value(i));
        }
        assert_eq!(by_key.get(&1), Some(&15));
        assert_eq!(by_key.get(&2), Some(&20));
    }

    #[test]
    fn count_star_ignores_the_arg_and_counts_rows() {
        let scan = Arc::new(MemoryScanExec::new(
            input_schema(),
            vec![batch(&[1, 1, 1], &[1, 2, 3])],
        ));
        let output_schema =
            Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]).unwrap());
        let exec = AggregateExec::new(
            scan,
            vec![],
            vec![],
            vec![None],
            vec![AggregateKind::Count],
            vec![DataType::Int64],
            output_schema,
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let col = as_primitive::<Int64Type>(batches[0].column(0).unwrap().as_ref()).unwrap();
        assert_eq!(col.value(0), 3);
    }

    #[test]
    fn empty_input_with_no_group_by_still_emits_one_row() {
        let scan = Arc::new(MemoryScanExec::new(input_schema(), vec![]));
        let output_schema =
            Arc::new(Schema::new(vec![Field::new("total", DataType::Int64, true)]).unwrap());
        let exec = AggregateExec::new(
            scan,
            vec![],
            vec![],
            vec![Some(Arc::new(ColumnExpr::new(1)))],
            vec![AggregateKind::Sum],
            vec![DataType::Int64],
            output_schema,
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        assert!(batches[0].column(0).unwrap().is_null(0));
    }

    #[test]
    fn empty_input_with_group_by_emits_zero_groups() {
        let scan = Arc::new(MemoryScanExec::new(input_schema(), vec![]));
        let output_schema = Arc::new(
            Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("total", DataType::Int64, true),
            ])
            .unwrap(),
        );
        let exec = AggregateExec::new(
            scan,
            vec![Arc::new(ColumnExpr::new(0))],
            vec![DataType::Int64],
            vec![Some(Arc::new(ColumnExpr::new(1)))],
            vec![AggregateKind::Sum],
            vec![DataType::Int64],
            output_schema,
        );
        let batches: Vec<_> = exec
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 0);
    }
}
