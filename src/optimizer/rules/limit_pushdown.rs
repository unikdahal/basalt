//! `LimitPushdown` — push `LIMIT` through `Projection` (safe: row count
//! unchanged) and no further. See design-docs/basalt-phase3-lld.md §6.3.
//!
//! **Not** through `Filter` (the filter changes how many rows you need to
//! read) or `Join`/`Aggregate` in general.
//!
//! **`Sort` + `Limit` -> `TopK` is already handled**, just not here:
//! `physical_plan::planner::PhysicalPlanner::create_physical_plan` already
//! special-cases the `Limit { fetch: Some(k), .. }` over `Sort` shape and
//! builds a `TopKExec` directly (Phase 2's own framing for that code: "in
//! Phase 3 this becomes a proper optimizer rule; in Phase 2, special-case
//! it in the planner"). Since `LogicalPlan` has no separate `TopK` node to
//! introduce, there's no logical-plan rewrite to add — the shape the
//! planner detects (`Limit` directly over `Sort`) already exists naturally
//! from the query as written, with nothing for a logical rule to change.
//!
//! **Pushing `LIMIT` into `TableScan` (`into scans that can stop early`) is
//! a documented deferral, not implemented here**: it would need a
//! `limit: Option<usize>` field on `TableScan` plus early-stopping support
//! in the scan executors (`MemoryScanExec` reads everything unconditionally
//! today), which is real new plumbing beyond a plan rewrite — a real,
//! identified follow-up rather than a silent omission.

use std::sync::Arc;

use crate::error::Result;
use crate::logical_plan::plan::LogicalPlan;
use crate::optimizer::rule::{ApplyOrder, OptimizerContext, OptimizerRule};
use crate::optimizer::tree_node::Transformed;

#[derive(Debug)]
pub struct LimitPushdown;

impl OptimizerRule for LimitPushdown {
    fn name(&self) -> &str {
        "limit_pushdown"
    }

    fn apply_order(&self) -> ApplyOrder {
        ApplyOrder::BottomUp
    }

    fn apply(
        &self,
        plan: LogicalPlan,
        _ctx: &dyn OptimizerContext,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Limit { input, skip, fetch } = plan else {
            return Ok(Transformed::No(plan));
        };
        let LogicalPlan::Projection {
            input: proj_input,
            exprs,
            schema,
        } = input.as_ref()
        else {
            return Ok(Transformed::No(LogicalPlan::Limit { input, skip, fetch }));
        };

        Ok(Transformed::Yes(LogicalPlan::Projection {
            input: Arc::new(LogicalPlan::Limit {
                input: proj_input.clone(),
                skip,
                fetch,
            }),
            exprs: exprs.clone(),
            schema: schema.clone(),
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
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema, SchemaRef};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap())
    }

    fn col(i: usize) -> Expr {
        Expr::Column {
            index: i,
            data_type: DataType::Int64,
            nullable: false,
        }
    }

    #[test]
    fn pushes_limit_through_projection() {
        let scan =
            LogicalPlanBuilder::scan("t", Arc::new(MemoryTableSource::new(schema(), vec![])))
                .build();
        let projection = LogicalPlan::Projection {
            input: scan,
            exprs: vec![col(0)],
            schema: schema(),
        };
        let plan = LogicalPlan::Limit {
            input: Arc::new(projection),
            skip: 0,
            fetch: Some(10),
        };
        let result = LimitPushdown.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Projection { input, .. } => match input.as_ref() {
                LogicalPlan::Limit { fetch, .. } => assert_eq!(*fetch, Some(10)),
                other => panic!("expected Limit, got {other:?}"),
            },
            other => panic!("expected Projection, got {other:?}"),
        }
    }

    #[test]
    fn does_not_push_through_a_filter() {
        let scan =
            LogicalPlanBuilder::scan("t", Arc::new(MemoryTableSource::new(schema(), vec![])))
                .build();
        let filter = LogicalPlan::Filter {
            input: scan,
            predicate: Expr::Literal(crate::types::value::Value::Boolean(true)),
        };
        let plan = LogicalPlan::Limit {
            input: Arc::new(filter),
            skip: 0,
            fetch: Some(10),
        };
        let result = LimitPushdown.apply(plan, &NoStatistics).unwrap();
        assert!(!result.is_yes());
    }
}
