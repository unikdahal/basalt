//! `MergeFilters` — `Filter(Filter(x, p1), p2) -> Filter(x, p1 AND p2)`.
//! See design-docs/basalt-phase3-lld.md §5.3.
//!
//! Cheap, and it makes pushdown simpler by giving it one conjunct list to
//! work with instead of two nested filter nodes.

use crate::error::Result;
use crate::expr::expr::Expr;
use crate::logical_plan::plan::LogicalPlan;
use crate::optimizer::rule::{ApplyOrder, OptimizerContext, OptimizerRule};
use crate::optimizer::tree_node::Transformed;
use crate::types::coercion::BinaryOp;

#[derive(Debug)]
pub struct MergeFilters;

impl OptimizerRule for MergeFilters {
    fn name(&self) -> &str {
        "merge_filters"
    }

    fn apply_order(&self) -> ApplyOrder {
        ApplyOrder::BottomUp
    }

    fn apply(&self, plan: LogicalPlan, _ctx: &dyn OptimizerContext) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Filter { input, predicate: outer } = plan else {
            return Ok(Transformed::No(plan));
        };
        let LogicalPlan::Filter { input: inner_input, predicate: inner } = input.as_ref() else {
            return Ok(Transformed::No(LogicalPlan::Filter { input, predicate: outer }));
        };
        Ok(Transformed::Yes(LogicalPlan::Filter {
            input: inner_input.clone(),
            predicate: Expr::Binary {
                left: Box::new(inner.clone()),
                op: BinaryOp::And,
                right: Box::new(outer),
            },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::optimizer::rule::NoStatistics;
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema};
    use crate::types::value::Value;
    use std::sync::Arc;

    fn schema() -> crate::types::schema::SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap())
    }

    #[test]
    fn merges_nested_filters_into_one_conjunction() {
        let scan = LogicalPlanBuilder::scan("t", Arc::new(MemoryTableSource::new(schema(), vec![]))).build();
        let p1 = Expr::Literal(Value::Boolean(true));
        let p2 = Expr::Literal(Value::Boolean(false));
        let plan = LogicalPlan::Filter {
            input: Arc::new(LogicalPlan::Filter {
                input: scan.clone(),
                predicate: p1.clone(),
            }),
            predicate: p2.clone(),
        };
        let result = MergeFilters.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Filter { input, predicate } => {
                assert!(matches!(input.as_ref(), LogicalPlan::TableScan { .. }));
                assert_eq!(
                    predicate,
                    Expr::Binary {
                        left: Box::new(p1),
                        op: BinaryOp::And,
                        right: Box::new(p2),
                    }
                );
            }
            other => panic!("expected Filter, got {other:?}"),
        }
    }

    #[test]
    fn single_filter_is_unchanged() {
        let scan = LogicalPlanBuilder::scan("t", Arc::new(MemoryTableSource::new(schema(), vec![]))).build();
        let plan = LogicalPlan::Filter {
            input: scan,
            predicate: Expr::Literal(Value::Boolean(true)),
        };
        let result = MergeFilters.apply(plan, &NoStatistics).unwrap();
        assert!(!result.is_yes());
    }
}
