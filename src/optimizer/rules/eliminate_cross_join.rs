//! `EliminateCrossJoin` — a cross join followed by a filter referencing
//! both sides becomes an inner join with that condition promoted into the
//! join's `on`/`filter`. See design-docs/basalt-phase3-lld.md §5.3.
//!
//! Essential because SQL written as `FROM a, b WHERE a.id = b.id` parses as
//! exactly this shape (`Filter` over a join with no `on`/`filter` of its
//! own) — without this rule the join order enumerator (§3.6) never sees a
//! join edge for it at all, since it only looks at `Join.on`.
//!
//! Conservative by construction: only equality conjuncts between a
//! left-only column and a right-only column get promoted to `on`; anything
//! else (a non-equality comparison, or a predicate that only touches one
//! side) is kept as a residual `Join.filter` condition, evaluated against
//! the joined row exactly as a post-join `Filter` would have been — no
//! semantic change, just a different node holding the same condition.

use crate::error::Result;
use crate::logical_plan::plan::{JoinType, LogicalPlan};
use crate::optimizer::rule::{ApplyOrder, OptimizerContext, OptimizerRule};
use crate::optimizer::tree_node::Transformed;

use super::common::{flatten_conjuncts, rebuild_conjuncts, split_equi_join_conjunct};

#[derive(Debug)]
pub struct EliminateCrossJoin;

impl OptimizerRule for EliminateCrossJoin {
    fn name(&self) -> &str {
        "eliminate_cross_join"
    }

    fn apply_order(&self) -> ApplyOrder {
        ApplyOrder::BottomUp
    }

    fn apply(
        &self,
        plan: LogicalPlan,
        _ctx: &dyn OptimizerContext,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Filter { input, predicate } = plan else {
            return Ok(Transformed::No(plan));
        };
        let LogicalPlan::Join {
            left,
            right,
            on,
            filter: existing_filter,
            join_type: JoinType::Inner,
            schema,
        } = input.as_ref()
        else {
            return Ok(Transformed::No(LogicalPlan::Filter { input, predicate }));
        };
        if !on.is_empty() || existing_filter.is_some() {
            // Already has a join condition — not the bare-cross-join shape
            // this rule targets; leave it for PredicatePushdown instead.
            return Ok(Transformed::No(LogicalPlan::Filter { input, predicate }));
        }

        let left_width = left.schema().fields().len();
        let (left, right, schema) = (left.clone(), right.clone(), schema.clone());
        let conjuncts = flatten_conjuncts(predicate.clone());
        let mut promoted = Vec::new();
        let mut leftover = Vec::new();
        for conjunct in conjuncts {
            match split_equi_join_conjunct(&conjunct, left_width) {
                Some(pair) => promoted.push(pair),
                None => leftover.push(conjunct),
            }
        }

        if promoted.is_empty() {
            // No equi-join edge found — a genuine cross product with only
            // non-equality/single-side predicates. Nothing to rewrite.
            return Ok(Transformed::No(LogicalPlan::Filter { input, predicate }));
        }

        let residual_filter = if leftover.is_empty() {
            None
        } else {
            Some(rebuild_conjuncts(leftover))
        };

        Ok(Transformed::Yes(LogicalPlan::Join {
            left,
            right,
            on: promoted,
            filter: residual_filter,
            join_type: JoinType::Inner,
            schema,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::expr::Expr;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::optimizer::rule::NoStatistics;
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::types::coercion::BinaryOp;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema, SchemaRef};
    use std::sync::Arc;

    fn schema(name: &str) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]).unwrap())
    }

    fn joined_schema() -> SchemaRef {
        Arc::new(Schema::new_allow_duplicate_names(vec![
            Field::new("l", DataType::Int64, false),
            Field::new("r", DataType::Int64, false),
        ]))
    }

    fn col(i: usize) -> Expr {
        Expr::Column {
            index: i,
            data_type: DataType::Int64,
            nullable: false,
        }
    }

    fn cross_join() -> LogicalPlan {
        let left =
            LogicalPlanBuilder::scan("l", Arc::new(MemoryTableSource::new(schema("l"), vec![])))
                .build();
        let right =
            LogicalPlanBuilder::scan("r", Arc::new(MemoryTableSource::new(schema("r"), vec![])))
                .build();
        LogicalPlan::Join {
            left,
            right,
            on: vec![],
            filter: None,
            join_type: JoinType::Inner,
            schema: joined_schema(),
        }
    }

    #[test]
    fn promotes_an_equality_conjunct_to_on() {
        let plan = LogicalPlan::Filter {
            input: Arc::new(cross_join()),
            predicate: Expr::Binary {
                left: Box::new(col(0)),
                op: BinaryOp::Eq,
                right: Box::new(col(1)),
            },
        };
        let result = EliminateCrossJoin.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Join { on, filter, .. } => {
                assert_eq!(on.len(), 1);
                assert!(filter.is_none());
                assert_eq!(on[0], (col(0), col(0))); // right column rebased to index 0
            }
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn keeps_non_equi_conjuncts_as_a_residual_join_filter() {
        let equi = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Eq,
            right: Box::new(col(1)),
        };
        let non_equi = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Lt,
            right: Box::new(col(1)),
        };
        let plan = LogicalPlan::Filter {
            input: Arc::new(cross_join()),
            predicate: Expr::Binary {
                left: Box::new(equi),
                op: BinaryOp::And,
                right: Box::new(non_equi.clone()),
            },
        };
        let result = EliminateCrossJoin.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Join { on, filter, .. } => {
                assert_eq!(on.len(), 1);
                assert_eq!(filter, Some(non_equi));
            }
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn no_equi_conjunct_leaves_plan_unchanged_in_shape() {
        let non_equi = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Lt,
            right: Box::new(col(1)),
        };
        let plan = LogicalPlan::Filter {
            input: Arc::new(cross_join()),
            predicate: non_equi,
        };
        let result = EliminateCrossJoin.apply(plan, &NoStatistics).unwrap();
        assert!(!result.is_yes());
    }

    #[test]
    fn already_conditioned_join_is_left_alone() {
        let mut join = cross_join();
        if let LogicalPlan::Join { on, .. } = &mut join {
            *on = vec![(col(0), col(0))];
        }
        let plan = LogicalPlan::Filter {
            input: Arc::new(join),
            predicate: Expr::Literal(crate::types::value::Value::Boolean(true)),
        };
        let result = EliminateCrossJoin.apply(plan, &NoStatistics).unwrap();
        assert!(!result.is_yes());
    }
}
