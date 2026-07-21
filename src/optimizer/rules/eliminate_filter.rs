//! `EliminateFilter` — `WHERE true` drops the node; `WHERE false` replaces
//! the subtree with an `EmptyRelation` carrying the right schema. See
//! design-docs/basalt-phase3-lld.md §5.3.

use crate::error::Result;
use crate::expr::expr::Expr;
use crate::logical_plan::plan::LogicalPlan;
use crate::optimizer::rule::{ApplyOrder, OptimizerContext, OptimizerRule};
use crate::optimizer::tree_node::Transformed;
use crate::types::value::Value;

#[derive(Debug)]
pub struct EliminateFilter;

impl OptimizerRule for EliminateFilter {
    fn name(&self) -> &str {
        "eliminate_filter"
    }

    fn apply_order(&self) -> ApplyOrder {
        ApplyOrder::BottomUp
    }

    fn apply(&self, plan: LogicalPlan, _ctx: &dyn OptimizerContext) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Filter { input, predicate } = plan else {
            return Ok(Transformed::No(plan));
        };
        match predicate {
            Expr::Literal(Value::Boolean(true)) => Ok(Transformed::Yes(input.as_ref().clone())),
            Expr::Literal(Value::Boolean(false)) => Ok(Transformed::Yes(LogicalPlan::EmptyRelation {
                schema: input.schema().clone(),
            })),
            other => Ok(Transformed::No(LogicalPlan::Filter { input, predicate: other })),
        }
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
    use std::sync::Arc;

    fn schema() -> crate::types::schema::SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap())
    }

    fn scan() -> Arc<LogicalPlan> {
        LogicalPlanBuilder::scan("t", Arc::new(MemoryTableSource::new(schema(), vec![]))).build()
    }

    #[test]
    fn where_true_drops_the_filter_node() {
        let plan = LogicalPlan::Filter {
            input: scan(),
            predicate: Expr::Literal(Value::Boolean(true)),
        };
        let result = EliminateFilter.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        assert!(matches!(result.into_inner(), LogicalPlan::TableScan { .. }));
    }

    #[test]
    fn where_false_becomes_empty_relation_with_the_right_schema() {
        let plan = LogicalPlan::Filter {
            input: scan(),
            predicate: Expr::Literal(Value::Boolean(false)),
        };
        let result = EliminateFilter.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::EmptyRelation { schema: s } => assert_eq!(s, schema()),
            other => panic!("expected EmptyRelation, got {other:?}"),
        }
    }

    #[test]
    fn non_literal_predicate_is_unchanged() {
        let predicate = Expr::Column {
            index: 0,
            data_type: DataType::Boolean,
            nullable: false,
        };
        let plan = LogicalPlan::Filter {
            input: scan(),
            predicate: predicate.clone(),
        };
        let result = EliminateFilter.apply(plan, &NoStatistics).unwrap();
        assert!(!result.is_yes());
    }
}
