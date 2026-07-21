//! `LogicalPlan` → `ExecutionPlan`, and the bound `expr::expr::Expr` →
//! `PhysicalExpr` conversion it needs along the way. See
//! design-docs/basalt-phase2-lld.md §5.4.
//!
//! In Phase 2 the mapping is 1:1 and mechanical — one `create_*_exec` method
//! per `LogicalPlan` variant. Phase 3 is where this starts *choosing* (hash
//! vs merge join, full sort vs top-N) based on statistics; the one-method-
//! per-node structure already gives each choice an obvious home.

use std::sync::Arc;

use super::aggregate::AggregateExec;
use super::filter::FilterExec;
use super::join::{HashJoinExec, NestedLoopJoinExec};
use super::limit::LimitExec;
use super::plan::ExecutionPlanRef;
use super::projection::ProjectionExec;
use super::scan::MemoryTableSource;
use super::sort::{PhysicalSortExpr, SortExec};
use crate::error::{BasaltError, Result};
use crate::expr::expr::{Expr, UnaryOp};
use crate::logical_plan::{JoinType, LogicalPlan};
use crate::physical_expr::binary::BinaryExpr;
use crate::physical_expr::cast::CastExpr;
use crate::physical_expr::column::ColumnExpr;
use crate::physical_expr::is_null::IsNullExpr;
use crate::physical_expr::literal::LiteralExpr;
use crate::physical_expr::unary::{NegExpr, NotExpr};
use crate::physical_expr::PhysicalExprRef;
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;
use crate::types::value::Value;

pub struct PhysicalPlanner;

impl PhysicalPlanner {
    /// # Errors
    /// Errors if a `LogicalPlan` node references a `TableSource` this
    /// planner doesn't know how to scan, or if an expression fails to
    /// convert (e.g. an untyped `NULL` literal with no `Cast` around it —
    /// the same case Phase 1's binder already can't type).
    pub fn create_physical_plan(&self, logical: &LogicalPlan) -> Result<ExecutionPlanRef> {
        match logical {
            LogicalPlan::TableScan { source, schema, .. } => {
                let mem_source = source
                    .as_any()
                    .downcast_ref::<MemoryTableSource>()
                    .ok_or_else(|| {
                        BasaltError::Internal(
                            "PhysicalPlanner only supports MemoryTableSource in Phase 2 core; \
                         CSV/Parquet scans are planned via their own io modules"
                                .to_string(),
                        )
                    })?;
                Ok(Arc::new(super::scan::MemoryScanExec::new(
                    schema.clone(),
                    mem_source.batches().to_vec(),
                )))
            }
            LogicalPlan::Filter { input, predicate } => {
                let child = self.create_physical_plan(input)?;
                let expr = self.create_physical_expr(predicate)?;
                Ok(Arc::new(FilterExec::new(child, expr)))
            }
            LogicalPlan::Projection {
                input,
                exprs,
                schema,
            } => {
                let child = self.create_physical_plan(input)?;
                let physical_exprs = exprs
                    .iter()
                    .map(|e| self.create_physical_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(ProjectionExec::new(
                    child,
                    physical_exprs,
                    schema.clone(),
                )))
            }
            // `ORDER BY ... LIMIT k` with no OFFSET: detect the Sort-under-
            // Limit shape and build the bounded-heap TopKExec instead of a
            // full SortExec followed by a LimitExec. In Phase 2 this is a
            // planner-level special case rather than a real optimizer rule
            // (the LLD's own framing: "in Phase 3 this becomes a proper
            // optimizer rule; in Phase 2, special-case it in the planner").
            LogicalPlan::Limit {
                input,
                skip: 0,
                fetch: Some(k),
            } if matches!(input.as_ref(), LogicalPlan::Sort { .. }) => {
                let LogicalPlan::Sort {
                    input: sort_input,
                    exprs,
                } = input.as_ref()
                else {
                    unreachable!("matched above")
                };
                let child = self.create_physical_plan(sort_input)?;
                let physical_exprs = self.physical_sort_exprs(exprs)?;
                Ok(Arc::new(super::sort::TopKExec::new(
                    child,
                    physical_exprs,
                    *k,
                )))
            }
            LogicalPlan::Limit { input, skip, fetch } => {
                let child = self.create_physical_plan(input)?;
                Ok(Arc::new(LimitExec::new(child, *skip, *fetch)))
            }
            LogicalPlan::Sort { input, exprs } => {
                let child = self.create_physical_plan(input)?;
                let physical_exprs = self.physical_sort_exprs(exprs)?;
                Ok(Arc::new(SortExec::new(child, physical_exprs)))
            }
            LogicalPlan::Aggregate {
                input,
                group_expr,
                aggr_expr,
                schema,
            } => {
                let child = self.create_physical_plan(input)?;
                let group_exprs = group_expr
                    .iter()
                    .map(|e| self.create_physical_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                let group_types = group_expr
                    .iter()
                    .map(Expr::data_type)
                    .collect::<Result<Vec<_>>>()?;

                let mut agg_arg_exprs = Vec::with_capacity(aggr_expr.len());
                let mut agg_kinds = Vec::with_capacity(aggr_expr.len());
                let mut agg_arg_types = Vec::with_capacity(aggr_expr.len());
                for agg in aggr_expr {
                    agg_kinds.push(agg.kind);
                    match &agg.arg {
                        Some(e) => {
                            agg_arg_exprs.push(Some(self.create_physical_expr(e)?));
                            agg_arg_types.push(e.data_type()?);
                        }
                        None => {
                            agg_arg_exprs.push(None);
                            agg_arg_types.push(DataType::Int64); // unused for COUNT(*)
                        }
                    }
                }

                Ok(Arc::new(AggregateExec::new(
                    child,
                    group_exprs,
                    group_types,
                    agg_arg_exprs,
                    agg_kinds,
                    agg_arg_types,
                    schema.clone(),
                )))
            }
            LogicalPlan::Join {
                left,
                right,
                on,
                filter,
                join_type,
                schema,
            } => {
                let build = self.create_physical_plan(left)?;
                let probe = self.create_physical_plan(right)?;
                if !on.is_empty() {
                    let physical_on = on
                        .iter()
                        .map(|(l, r)| {
                            Ok((self.create_physical_expr(l)?, self.create_physical_expr(r)?))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let physical_filter = filter
                        .as_ref()
                        .map(|f| self.create_physical_expr(f))
                        .transpose()?;
                    Ok(Arc::new(HashJoinExec::with_filter(
                        build,
                        probe,
                        physical_on,
                        physical_filter,
                        *join_type,
                        schema.clone(),
                    )))
                } else if let Some(f) = filter {
                    if *join_type != JoinType::Inner {
                        return Err(BasaltError::Internal(format!(
                            "NestedLoopJoinExec only supports Inner joins, got {join_type:?}"
                        )));
                    }
                    let predicate = self.create_physical_expr(f)?;
                    Ok(Arc::new(NestedLoopJoinExec::new(
                        build,
                        probe,
                        predicate,
                        schema.clone(),
                    )))
                } else {
                    Err(BasaltError::Internal(
                        "Join requires either equi-join keys or a filter predicate".to_string(),
                    ))
                }
            }
        }
    }

    fn physical_sort_exprs(
        &self,
        exprs: &[crate::logical_plan::SortExpr],
    ) -> Result<Vec<PhysicalSortExpr>> {
        exprs
            .iter()
            .map(|se| {
                Ok(PhysicalSortExpr {
                    expr: self.create_physical_expr(&se.expr)?,
                    options: crate::compute::sort::SortOptions {
                        descending: se.options.descending,
                        nulls_first: se.options.nulls_first,
                    },
                })
            })
            .collect()
    }

    /// Takes no schema parameter: Phase 1's bound `Expr` already carries
    /// every column's resolved type and ordinal inline (that's what
    /// "bound" means), so unlike the logical-plan builder this conversion
    /// never needs to look anything up against a `Schema`.
    fn create_physical_expr(&self, expr: &Expr) -> Result<PhysicalExprRef> {
        match expr {
            Expr::Column { index, .. } => Ok(Arc::new(ColumnExpr::new(*index))),
            Expr::Literal(value) => {
                // An untyped Value::Null with no static type context errors
                // here exactly as it does in Phase 1's expr::typing — this
                // isn't a new restriction, just where it now surfaces.
                let data_type = expr.data_type()?;
                Ok(Arc::new(LiteralExpr::new(value_to_scalar(
                    value, data_type,
                ))))
            }
            Expr::Binary { left, op, right } => {
                let l = self.create_physical_expr(left)?;
                let r = self.create_physical_expr(right)?;
                Ok(Arc::new(BinaryExpr::new(l, *op, r)))
            }
            Expr::Unary { op, expr } => {
                let inner = self.create_physical_expr(expr)?;
                Ok(match op {
                    UnaryOp::Neg => Arc::new(NegExpr::new(inner)),
                    UnaryOp::Not => Arc::new(NotExpr::new(inner)),
                })
            }
            Expr::Cast { expr, to } => {
                let inner = self.create_physical_expr(expr)?;
                Ok(Arc::new(CastExpr::new(inner, *to)))
            }
            Expr::IsNull(inner) => Ok(Arc::new(IsNullExpr::new(
                self.create_physical_expr(inner)?,
                false,
            ))),
            Expr::IsNotNull(inner) => Ok(Arc::new(IsNullExpr::new(
                self.create_physical_expr(inner)?,
                true,
            ))),
        }
    }
}

fn value_to_scalar(value: &Value, data_type: DataType) -> ScalarValue {
    match value {
        Value::Null => match data_type {
            DataType::Int64 => ScalarValue::Int64(None),
            DataType::Float64 => ScalarValue::Float64(None),
            DataType::Utf8 => ScalarValue::Utf8(None),
            DataType::Boolean => ScalarValue::Boolean(None),
        },
        Value::Int64(v) => ScalarValue::Int64(Some(*v)),
        Value::Float64(v) => ScalarValue::Float64(Some(*v)),
        Value::Utf8(v) => ScalarValue::Utf8(Some(v.clone())),
        Value::Boolean(v) => ScalarValue::Boolean(Some(*v)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_primitive;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::batch::ColumnarBatch;
    use crate::logical_plan::LogicalPlanBuilder;
    use crate::types::coercion::BinaryOp;
    use crate::types::schema::{Field, Schema, SchemaRef};
    use std::sync::Arc;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap())
    }

    fn source_batch(values: &[i64]) -> ColumnarBatch {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            b.append_value(v);
        }
        ColumnarBatch::try_new(schema(), vec![Arc::new(b.finish())]).unwrap()
    }

    /// The first end-to-end vectorized query: scan -> filter -> project ->
    /// limit, exactly the pipeline the LLD calls the emotional milestone of
    /// build order step 11.
    #[test]
    fn end_to_end_scan_filter_project_limit() {
        let source = Arc::new(MemoryTableSource::new(
            schema(),
            vec![source_batch(&[1, 2, 3, 4, 5])],
        ));
        let filter_expr = Expr::Binary {
            left: Box::new(Expr::Column {
                index: 0,
                data_type: DataType::Int64,
                nullable: false,
            }),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal(Value::Int64(2))),
        };
        let project_expr = Expr::Binary {
            left: Box::new(Expr::Column {
                index: 0,
                data_type: DataType::Int64,
                nullable: false,
            }),
            op: BinaryOp::Mul,
            right: Box::new(Expr::Literal(Value::Int64(10))),
        };

        let logical = LogicalPlanBuilder::scan("t", source)
            .filter(filter_expr)
            .project(vec![project_expr], vec!["a_times_10".to_string()])
            .unwrap()
            .limit(0, Some(2))
            .build();

        let physical = PhysicalPlanner.create_physical_plan(&logical).unwrap();
        let batches: Vec<_> = physical
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let values: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                let col = as_primitive::<Int64Type>(b.column(0).unwrap().as_ref()).unwrap();
                (0..b.num_rows()).map(|i| col.value(i)).collect::<Vec<_>>()
            })
            .collect();
        // a > 2 -> [3,4,5]; * 10 -> [30,40,50]; limit 2 -> [30,40]
        assert_eq!(values, vec![30, 40]);
    }

    #[test]
    fn end_to_end_scan_then_aggregate() {
        let source = Arc::new(MemoryTableSource::new(
            schema(),
            vec![source_batch(&[1, 2, 3, 4, 5])],
        ));
        let logical = LogicalPlanBuilder::scan("t", source)
            .aggregate(
                vec![],
                vec![crate::logical_plan::AggregateFunction {
                    output_name: "total".to_string(),
                    kind: crate::logical_plan::AggregateKind::Sum,
                    arg: Some(Expr::Column {
                        index: 0,
                        data_type: DataType::Int64,
                        nullable: false,
                    }),
                }],
            )
            .unwrap()
            .build();

        let physical = PhysicalPlanner.create_physical_plan(&logical).unwrap();
        let batches: Vec<_> = physical
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        let col = as_primitive::<Int64Type>(batches[0].column(0).unwrap().as_ref()).unwrap();
        assert_eq!(col.value(0), 15);
    }

    #[test]
    fn untyped_null_literal_errors_like_phase_one_does() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let logical = LogicalPlanBuilder::scan("t", source)
            .filter(Expr::Literal(Value::Null))
            .build();
        assert!(PhysicalPlanner.create_physical_plan(&logical).is_err());
    }

    #[test]
    fn unsupported_plan_node_errors_cleanly_not_panics() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let right =
            LogicalPlanBuilder::scan("t2", Arc::new(MemoryTableSource::new(schema(), vec![])));
        let logical = LogicalPlanBuilder::scan("t", source)
            .join(
                right.build(),
                vec![],
                None,
                crate::logical_plan::JoinType::Inner,
            )
            .unwrap()
            .build();
        assert!(PhysicalPlanner.create_physical_plan(&logical).is_err());
    }

    #[test]
    fn end_to_end_scan_then_sort() {
        let source = Arc::new(MemoryTableSource::new(
            schema(),
            vec![source_batch(&[3, 1, 2])],
        ));
        let logical = LogicalPlanBuilder::scan("t", source)
            .sort(vec![crate::logical_plan::SortExpr {
                expr: Expr::Column {
                    index: 0,
                    data_type: DataType::Int64,
                    nullable: false,
                },
                options: crate::logical_plan::SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }])
            .build();
        let physical = PhysicalPlanner.create_physical_plan(&logical).unwrap();
        let batches: Vec<_> = physical
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let col = as_primitive::<Int64Type>(batches[0].column(0).unwrap().as_ref()).unwrap();
        assert_eq!(
            (0..3).map(|i| col.value(i)).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    /// `ORDER BY ... LIMIT k` (no OFFSET) must plan to a `TopKExec`, not a
    /// `SortExec` composed with `LimitExec` — the planner-level special case
    /// documented in create_physical_plan's Limit-over-Sort arm.
    #[test]
    fn order_by_limit_plans_to_topk_exec() {
        let source = Arc::new(MemoryTableSource::new(
            schema(),
            vec![source_batch(&[5, 3, 1, 4, 2])],
        ));
        let logical = LogicalPlanBuilder::scan("t", source)
            .sort(vec![crate::logical_plan::SortExpr {
                expr: Expr::Column {
                    index: 0,
                    data_type: DataType::Int64,
                    nullable: false,
                },
                options: crate::logical_plan::SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }])
            .limit(0, Some(3))
            .build();
        let physical = PhysicalPlanner.create_physical_plan(&logical).unwrap();
        assert!(physical
            .as_any()
            .downcast_ref::<crate::physical_plan::sort::TopKExec>()
            .is_some());

        let batches: Vec<_> = physical
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let col = as_primitive::<Int64Type>(batches[0].column(0).unwrap().as_ref()).unwrap();
        assert_eq!(
            (0..3).map(|i| col.value(i)).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn end_to_end_inner_join() {
        let left_source = Arc::new(MemoryTableSource::new(
            schema(),
            vec![source_batch(&[1, 2, 3])],
        ));
        let right_source = Arc::new(MemoryTableSource::new(
            schema(),
            vec![source_batch(&[2, 3, 4])],
        ));
        let left = LogicalPlanBuilder::scan("l", left_source);
        let right = LogicalPlanBuilder::scan("r", right_source).build();

        let key = Expr::Column {
            index: 0,
            data_type: DataType::Int64,
            nullable: false,
        };
        let logical = left
            .join(
                right,
                vec![(key.clone(), key)],
                None,
                crate::logical_plan::JoinType::Inner,
            )
            .unwrap()
            .build();

        let physical = PhysicalPlanner.create_physical_plan(&logical).unwrap();
        let batches: Vec<_> = physical
            .execute(0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 2); // 2 and 3 match on both sides
    }
}
