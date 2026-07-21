//! `HashJoinExec`. See design-docs/basalt-phase2-lld.md §7.1.
//!
//! **Two explicit, documented scope decisions relative to the LLD:**
//!
//! 1. **Both sides are fully materialized**, not just the build side. The
//!    LLD's probe side streams; this implementation concatenates it first
//!    (the same `compute::concat` pattern `SortExec`/`AggregateExec` already
//!    use for their own pipeline-breaking). Real streaming-probe with
//!    bounded memory is a legitimate follow-up; buffering both sides is
//!    simplest to get *correct* first, and this operator was always going
//!    to buffer the build side regardless.
//! 2. **The residual `filter` (non-equi predicate) is not implemented.**
//!    Equi-joins — the overwhelming majority of real queries — are fully
//!    supported; a query needing `ON a.x = b.y AND a.z < b.w` isn't yet.
//!
//! **Join-type convention, spelled out because "left"/"right" is genuinely
//! ambiguous between SQL-table-position and build/probe-role:** this
//! operator names its children `build` and `probe`, not `left`/`right`, and
//! `JoinType` is defined purely in terms of those roles, matching the LLD's
//! own emit-rule table read literally:
//! - `Left`/`LeftSemi`/`LeftAnti` are about the **probe** side (its rows are
//!   preserved/tested), matching the LLD's "Left: unmatched *probe* rows...".
//! - `Right`/`RightSemi`/`RightAnti` are about the **build** side, matching
//!   "Right: unmatched *build* rows...". `RightSemi`/`RightAnti` are this
//!   operator's build-side analogue of `LeftSemi`/`LeftAnti`.
//!
//! The planner is responsible for deciding which physical input becomes
//! `build` vs `probe` (the LLD defers real cost-based build-side selection
//! to Phase 3 statistics); today it always builds on the plan's `build`
//! argument as given, matching the LLD's Phase 2 default of "build on the
//! left/first input."

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use super::super::plan::{BatchStream, ExecutionPlan};
use crate::array::array::ArrayRef;
use crate::batch::ColumnarBatch;
use crate::compute::index::UInt32Builder;
use crate::compute::take::take;
use crate::compute::{concat, take as take_mod};
use crate::error::{BasaltError, Result};
use crate::logical_plan::JoinType;
use crate::physical_expr::PhysicalExprRef;
use crate::physical_plan::aggregate::group_keys::GroupKeyEncoder;
use crate::types::schema::SchemaRef;

#[derive(Debug)]
pub struct HashJoinExec {
    build: Arc<dyn ExecutionPlan>,
    probe: Arc<dyn ExecutionPlan>,
    /// `(build_key_expr, probe_key_expr)` pairs — an equi-join condition per pair.
    on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
    join_type: JoinType,
    schema: SchemaRef,
}

impl HashJoinExec {
    pub fn new(
        build: Arc<dyn ExecutionPlan>,
        probe: Arc<dyn ExecutionPlan>,
        on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
        join_type: JoinType,
        schema: SchemaRef,
    ) -> Self {
        HashJoinExec {
            build,
            probe,
            on,
            join_type,
            schema,
        }
    }
}

fn empty_array_for(data_type: crate::types::data_type::DataType) -> ArrayRef {
    use crate::array::boolean::BooleanBuilder;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::string::StringBuilder;
    use crate::array::types::{Float64Type, Int64Type};
    use crate::types::data_type::DataType;
    match data_type {
        DataType::Int64 => Arc::new(PrimitiveBuilder::<Int64Type>::with_capacity(0).finish()),
        DataType::Float64 => Arc::new(PrimitiveBuilder::<Float64Type>::with_capacity(0).finish()),
        DataType::Utf8 => Arc::new(StringBuilder::with_capacity(0, 0).finish()),
        DataType::Boolean => Arc::new(BooleanBuilder::with_capacity(0).finish()),
    }
}

pub(super) fn materialize(plan: &dyn ExecutionPlan, partition: usize) -> Result<ColumnarBatch> {
    let schema = plan.schema();
    let batches: Vec<ColumnarBatch> = plan.execute(partition)?.collect::<Result<Vec<_>>>()?;
    if batches.is_empty() {
        let columns = schema
            .fields()
            .iter()
            .map(|f| empty_array_for(f.data_type))
            .collect();
        return ColumnarBatch::try_new(schema.clone(), columns);
    }
    let num_columns = schema.len();
    let mut columns = Vec::with_capacity(num_columns);
    for col_idx in 0..num_columns {
        let parts: Vec<ArrayRef> = batches
            .iter()
            .map(|b| Arc::clone(b.column(col_idx).unwrap()))
            .collect();
        columns.push(concat::concat(&parts)?);
    }
    ColumnarBatch::try_new(schema.clone(), columns)
}

fn is_null_key(key_arrays: &[ArrayRef], row: usize) -> bool {
    key_arrays.iter().any(|a| a.is_null(row))
}

impl ExecutionPlan for HashJoinExec {
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
            [build, probe] => Ok(Arc::new(HashJoinExec::new(
                Arc::clone(build),
                Arc::clone(probe),
                self.on.clone(),
                self.join_type,
                self.schema.clone(),
            ))),
            other => Err(BasaltError::Internal(format!(
                "HashJoinExec takes exactly 2 children, got {}",
                other.len()
            ))),
        }
    }

    fn execute(&self, partition: usize) -> Result<BatchStream> {
        let build_batch = materialize(self.build.as_ref(), partition)?;
        let probe_batch = materialize(self.probe.as_ref(), partition)?;
        let build_rows = build_batch.num_rows();
        let probe_rows = probe_batch.num_rows();

        let build_key_arrays: Vec<ArrayRef> = self
            .on
            .iter()
            .map(|(be, _)| be.evaluate(&build_batch)?.into_array(build_rows))
            .collect::<Result<Vec<_>>>()?;
        let probe_key_arrays: Vec<ArrayRef> = self
            .on
            .iter()
            .map(|(_, pe)| pe.evaluate(&probe_batch)?.into_array(probe_rows))
            .collect::<Result<Vec<_>>>()?;

        let key_types: Vec<_> = build_key_arrays.iter().map(|a| a.data_type()).collect();
        let encoder = GroupKeyEncoder::new(key_types);

        let mut build_encoded = Vec::new();
        let mut build_offsets = Vec::new();
        encoder.encode(&build_key_arrays, &mut build_encoded, &mut build_offsets)?;

        // Null keys never match, in any join type (Phase 1's Rule 1: `NULL =
        // NULL` is `NULL`, not true) — so they're simply never inserted into
        // the build side's lookup map. They still participate as ordinary
        // (always-unmatched) rows for Right/Full/RightAnti's bookkeeping.
        let mut build_map: HashMap<&[u8], Vec<u32>> = HashMap::new();
        for row in 0..build_rows {
            if is_null_key(&build_key_arrays, row) {
                continue;
            }
            let key = &build_encoded[build_offsets[row] as usize..build_offsets[row + 1] as usize];
            build_map.entry(key).or_default().push(row as u32);
        }

        let mut probe_encoded = Vec::new();
        let mut probe_offsets = Vec::new();
        encoder.encode(&probe_key_arrays, &mut probe_encoded, &mut probe_offsets)?;

        let mut build_matched = vec![false; build_rows];
        let mut probe_matched = vec![false; probe_rows];
        let mut pair_build = UInt32Builder::with_capacity(probe_rows);
        let mut pair_probe = UInt32Builder::with_capacity(probe_rows);

        for row in 0..probe_rows {
            if is_null_key(&probe_key_arrays, row) {
                continue;
            }
            let key = &probe_encoded[probe_offsets[row] as usize..probe_offsets[row + 1] as usize];
            if let Some(build_positions) = build_map.get(key) {
                for &b in build_positions {
                    pair_build.append_value(b);
                    pair_probe.append_value(row as u32);
                    build_matched[b as usize] = true;
                }
                if !build_positions.is_empty() {
                    probe_matched[row] = true;
                }
            }
        }

        match self.join_type {
            JoinType::Inner => {}
            JoinType::Left | JoinType::Full => {
                for (row, &was_matched) in probe_matched.iter().enumerate() {
                    if !was_matched {
                        pair_build.append_null();
                        pair_probe.append_value(row as u32);
                    }
                }
                if self.join_type == JoinType::Full {
                    for (b, &was_matched) in build_matched.iter().enumerate() {
                        if !was_matched {
                            pair_build.append_value(b as u32);
                            pair_probe.append_null();
                        }
                    }
                }
            }
            JoinType::Right => {
                for (b, &was_matched) in build_matched.iter().enumerate() {
                    if !was_matched {
                        pair_build.append_value(b as u32);
                        pair_probe.append_null();
                    }
                }
            }
            JoinType::LeftSemi | JoinType::LeftAnti | JoinType::RightSemi | JoinType::RightAnti => {
                // These four don't use the pair builders at all — handled below.
            }
        }

        match self.join_type {
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
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
            JoinType::LeftSemi | JoinType::LeftAnti => {
                let mut idx = UInt32Builder::with_capacity(probe_rows);
                for (row, &was_matched) in probe_matched.iter().enumerate() {
                    let keep = if self.join_type == JoinType::LeftSemi {
                        was_matched
                    } else {
                        !was_matched
                    };
                    if keep {
                        idx.append_value(row as u32);
                    }
                }
                let indices = idx.finish();
                let columns = (0..probe_batch.num_columns())
                    .map(|i| take_mod::take(probe_batch.column(i).unwrap().as_ref(), &indices))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Box::new(std::iter::once(Ok(ColumnarBatch::try_new(
                    self.schema.clone(),
                    columns,
                )?))))
            }
            JoinType::RightSemi | JoinType::RightAnti => {
                let mut idx = UInt32Builder::with_capacity(build_rows);
                for (b, &was_matched) in build_matched.iter().enumerate() {
                    let keep = if self.join_type == JoinType::RightSemi {
                        was_matched
                    } else {
                        !was_matched
                    };
                    if keep {
                        idx.append_value(b as u32);
                    }
                }
                let indices = idx.finish();
                let columns = (0..build_batch.num_columns())
                    .map(|i| take_mod::take(build_batch.column(i).unwrap().as_ref(), &indices))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Box::new(std::iter::once(Ok(ColumnarBatch::try_new(
                    self.schema.clone(),
                    columns,
                )?))))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::{as_primitive, Array};
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::physical_expr::column::ColumnExpr;
    use crate::physical_plan::scan::MemoryScanExec;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema};

    fn side_schema(col_name: &str) -> SchemaRef {
        Arc::new(
            Schema::new(vec![
                Field::new("key", DataType::Int64, true),
                Field::new(col_name, DataType::Int64, false),
            ])
            .unwrap(),
        )
    }

    fn side_batch(schema: SchemaRef, keys: &[Option<i64>], vals: &[i64]) -> ColumnarBatch {
        let mut kb = PrimitiveBuilder::<Int64Type>::with_capacity(keys.len());
        for &k in keys {
            match k {
                Some(k) => kb.append_value(k),
                None => kb.append_null(),
            }
        }
        let mut vb = PrimitiveBuilder::<Int64Type>::with_capacity(vals.len());
        for &v in vals {
            vb.append_value(v);
        }
        ColumnarBatch::try_new(schema, vec![Arc::new(kb.finish()), Arc::new(vb.finish())]).unwrap()
    }

    fn output_schema() -> SchemaRef {
        Arc::new(
            Schema::new(vec![
                Field::new("bkey", DataType::Int64, true),
                Field::new("bval", DataType::Int64, false),
                Field::new("pkey", DataType::Int64, true),
                Field::new("pval", DataType::Int64, false),
            ])
            .unwrap(),
        )
    }

    fn make_join(
        join_type: JoinType,
        build_rows: ColumnarBatch,
        probe_rows: ColumnarBatch,
    ) -> HashJoinExec {
        let build_schema = build_rows.schema().clone();
        let probe_schema = probe_rows.schema().clone();
        let build = Arc::new(MemoryScanExec::new(build_schema.clone(), vec![build_rows]));
        let probe = Arc::new(MemoryScanExec::new(probe_schema.clone(), vec![probe_rows]));
        // Semi/anti joins keep only one side's columns; every other type
        // keeps both (build fields followed by probe fields).
        let schema = match join_type {
            JoinType::LeftSemi | JoinType::LeftAnti => probe_schema,
            JoinType::RightSemi | JoinType::RightAnti => build_schema,
            _ => output_schema(),
        };
        HashJoinExec::new(
            build,
            probe,
            vec![(Arc::new(ColumnExpr::new(0)), Arc::new(ColumnExpr::new(0)))],
            join_type,
            schema,
        )
    }

    fn int_col(b: &ColumnarBatch, i: usize) -> Vec<Option<i64>> {
        let arr = as_primitive::<Int64Type>(b.column(i).unwrap().as_ref()).unwrap();
        (0..b.num_rows())
            .map(|r| {
                if arr.is_null(r) {
                    None
                } else {
                    Some(arr.value(r))
                }
            })
            .collect()
    }

    #[test]
    fn inner_join_emits_only_matched_pairs() {
        let build = side_batch(side_schema("bval"), &[Some(1), Some(2)], &[10, 20]);
        let probe = side_batch(side_schema("pval"), &[Some(2), Some(3)], &[200, 300]);
        let join = make_join(JoinType::Inner, build, probe);
        let batches: Vec<_> = join
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(int_col(&batches[0], 0), vec![Some(2)]);
        assert_eq!(int_col(&batches[0], 3), vec![Some(200)]);
    }

    #[test]
    fn left_join_keeps_unmatched_probe_rows_with_null_build_side() {
        let build = side_batch(side_schema("bval"), &[Some(1)], &[10]);
        let probe = side_batch(side_schema("pval"), &[Some(1), Some(99)], &[100, 900]);
        let join = make_join(JoinType::Left, build, probe);
        let batches: Vec<_> = join
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 2);
        // The unmatched probe row (key 99) must appear with nulls on the build side.
        let bkeys = int_col(&batches[0], 0);
        let pkeys = int_col(&batches[0], 2);
        assert!(bkeys.contains(&None));
        assert!(pkeys.contains(&Some(99)));
    }

    #[test]
    fn right_join_keeps_unmatched_build_rows_with_null_probe_side() {
        let build = side_batch(side_schema("bval"), &[Some(1), Some(2)], &[10, 20]);
        let probe = side_batch(side_schema("pval"), &[Some(1)], &[100]);
        let join = make_join(JoinType::Right, build, probe);
        let batches: Vec<_> = join
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 2);
        let pkeys = int_col(&batches[0], 2);
        assert!(pkeys.contains(&None));
    }

    #[test]
    fn full_join_keeps_both_sides_unmatched_rows() {
        let build = side_batch(side_schema("bval"), &[Some(1), Some(2)], &[10, 20]);
        let probe = side_batch(side_schema("pval"), &[Some(2), Some(3)], &[200, 300]);
        let join = make_join(JoinType::Full, build, probe);
        let batches: Vec<_> = join
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        // matched: (2,2); unmatched build: 1; unmatched probe: 3 -> 3 total rows
        assert_eq!(batches[0].num_rows(), 3);
    }

    #[test]
    fn left_semi_emits_each_matching_probe_row_once() {
        let build = side_batch(side_schema("bval"), &[Some(1), Some(1)], &[10, 11]); // two build rows share key 1
        let probe = side_batch(side_schema("pval"), &[Some(1), Some(2)], &[100, 200]);
        let join = make_join(JoinType::LeftSemi, build, probe);
        let batches: Vec<_> = join
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        // Only the probe schema (2 fields), and the matching probe row exactly once
        // despite matching two build rows.
        assert_eq!(batches[0].num_columns(), 2);
        assert_eq!(batches[0].num_rows(), 1);
    }

    #[test]
    fn left_anti_emits_unmatched_probe_rows_only() {
        let build = side_batch(side_schema("bval"), &[Some(1)], &[10]);
        let probe = side_batch(side_schema("pval"), &[Some(1), Some(2)], &[100, 200]);
        let join = make_join(JoinType::LeftAnti, build, probe);
        let batches: Vec<_> = join
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 1);
    }

    #[test]
    fn right_semi_and_anti_operate_on_build_rows() {
        let build = side_batch(side_schema("bval"), &[Some(1), Some(2)], &[10, 20]);
        let probe = side_batch(side_schema("pval"), &[Some(1)], &[100]);

        let semi = make_join(JoinType::RightSemi, build.clone(), probe.clone());
        let semi_batches: Vec<_> = semi
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(semi_batches[0].num_rows(), 1); // build key 1 matched

        let anti = make_join(JoinType::RightAnti, build, probe);
        let anti_batches: Vec<_> = anti
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(anti_batches[0].num_rows(), 1); // build key 2 unmatched
    }

    #[test]
    fn null_keys_never_match_but_still_appear_as_unmatched() {
        let build = side_batch(side_schema("bval"), &[None, Some(1)], &[10, 11]);
        let probe = side_batch(side_schema("pval"), &[None, Some(1)], &[100, 111]);
        let join = make_join(JoinType::Full, build, probe);
        let batches: Vec<_> = join
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        // (1,1) matches; null-keyed build row and null-keyed probe row both
        // appear as unmatched (never matching each other, per NULL = NULL = NULL).
        assert_eq!(batches[0].num_rows(), 3);
    }

    #[test]
    fn one_to_many_match_produces_a_row_per_pair() {
        let build = side_batch(
            side_schema("bval"),
            &[Some(1), Some(1), Some(1)],
            &[10, 11, 12],
        );
        let probe = side_batch(side_schema("pval"), &[Some(1)], &[100]);
        let join = make_join(JoinType::Inner, build, probe);
        let batches: Vec<_> = join
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 3);
    }

    #[test]
    fn empty_build_side_with_inner_join_yields_no_rows() {
        let build = side_batch(side_schema("bval"), &[], &[]);
        let probe = side_batch(side_schema("pval"), &[Some(1)], &[100]);
        let join = make_join(JoinType::Inner, build, probe);
        let batches: Vec<_> = join
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 0);
    }
}
