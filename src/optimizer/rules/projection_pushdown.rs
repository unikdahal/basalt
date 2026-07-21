//! `ProjectionPushdown` — compute the required column set top-down (the
//! root needs its output columns; each node adds what its own expressions
//! reference) and push the minimal set into `TableScan.projection`. See
//! design-docs/basalt-phase3-lld.md §6.2.
//!
//! On columnar storage this is the biggest single I/O win available: a
//! 50-column table where the query touches 3 columns reads ~6% of the
//! bytes. `TableScan.projection` has been sitting unused since Phase 2 for
//! exactly this moment.
//!
//! Implemented as a single top-down pass (not the fixed-point `TreeNode`
//! machinery the other rules use) because "what columns does this node
//! need" only makes sense computed against the *whole* plan from the root
//! down, not node-by-node bottom-up — the same reason the LLD frames it as
//! "compute required columns top-down."

use std::collections::HashSet;
use std::sync::Arc;

use crate::error::Result;
use crate::expr::expr::Expr;
use crate::logical_plan::plan::LogicalPlan;
use crate::optimizer::rule::{ApplyOrder, OptimizerContext, OptimizerRule};
use crate::optimizer::tree_node::Transformed;

use super::common::referenced_columns;

#[derive(Debug)]
pub struct ProjectionPushdown;

impl OptimizerRule for ProjectionPushdown {
    fn name(&self) -> &str {
        "projection_pushdown"
    }

    fn apply_order(&self) -> ApplyOrder {
        ApplyOrder::TopDown
    }

    /// Called once per node by the pass manager's top-down traversal (see
    /// `Optimizer::optimize`), so this only needs to recognize the local
    /// shape "a `Projection`/`Filter` sitting directly over a `TableScan`"
    /// — the traversal itself is what reaches every such pair anywhere in
    /// the tree, not this function recursing on its own. A `TableScan`
    /// with more than one node between it and the column-consuming node
    /// (e.g. `Projection` over `Filter` over `TableScan`) is handled once
    /// `PredicatePushdown` has already moved the `Filter` down to sit
    /// directly on the scan, which the fixed-point loop guarantees happens
    /// before this rule's next pass sees it.
    fn apply(
        &self,
        plan: LogicalPlan,
        _ctx: &dyn OptimizerContext,
    ) -> Result<Transformed<LogicalPlan>> {
        match &plan {
            LogicalPlan::Projection { input, exprs, .. }
                if matches!(input.as_ref(), LogicalPlan::TableScan { .. }) =>
            {
                let needed = required_columns_for_exprs(exprs);
                let input = input.clone();
                push_projection_into_scan(plan, &input, &needed)
            }
            LogicalPlan::Filter { input, predicate }
                if matches!(input.as_ref(), LogicalPlan::TableScan { .. }) =>
            {
                let needed = referenced_columns(predicate).into_iter().collect();
                let input = input.clone();
                push_filter_into_scan(plan, &input, &needed)
            }
            _ => Ok(Transformed::No(plan)),
        }
    }
}

fn required_columns_for_exprs(exprs: &[Expr]) -> HashSet<usize> {
    exprs.iter().flat_map(referenced_columns).collect()
}

fn push_projection_into_scan(
    plan: LogicalPlan,
    scan: &Arc<LogicalPlan>,
    needed: &HashSet<usize>,
) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::TableScan {
        table_name,
        source,
        projection,
        filters,
        schema,
    } = scan.as_ref()
    else {
        unreachable!("caller matched TableScan");
    };
    if projection.is_some() {
        return Ok(Transformed::No(plan)); // Already pushed.
    }
    let mut cols: Vec<usize> = needed.iter().copied().collect();
    cols.sort_unstable();
    if cols.len() == schema.fields().len() {
        return Ok(Transformed::No(plan)); // Every column needed — nothing to prune.
    }

    let new_scan = Arc::new(LogicalPlan::TableScan {
        table_name: table_name.clone(),
        source: source.clone(),
        projection: Some(cols),
        filters: filters.clone(),
        schema: schema.clone(),
    });
    let LogicalPlan::Projection {
        exprs,
        schema: proj_schema,
        ..
    } = plan
    else {
        unreachable!("caller matched Projection");
    };
    Ok(Transformed::Yes(LogicalPlan::Projection {
        input: new_scan,
        exprs,
        schema: proj_schema,
    }))
}

fn push_filter_into_scan(
    plan: LogicalPlan,
    scan: &Arc<LogicalPlan>,
    needed: &HashSet<usize>,
) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::TableScan {
        table_name,
        source,
        projection,
        filters,
        schema,
    } = scan.as_ref()
    else {
        unreachable!("caller matched TableScan");
    };
    if projection.is_some() || needed.len() == schema.fields().len() {
        return Ok(Transformed::No(plan));
    }
    let mut cols: Vec<usize> = needed.iter().copied().collect();
    cols.sort_unstable();
    let new_scan = Arc::new(LogicalPlan::TableScan {
        table_name: table_name.clone(),
        source: source.clone(),
        projection: Some(cols),
        filters: filters.clone(),
        schema: schema.clone(),
    });
    let LogicalPlan::Filter { predicate, .. } = plan else {
        unreachable!("caller matched Filter");
    };
    Ok(Transformed::Yes(LogicalPlan::Filter {
        input: new_scan,
        predicate,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::optimizer::rule::NoStatistics;
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema, SchemaRef};

    fn wide_schema() -> SchemaRef {
        Arc::new(
            Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Int64, false),
                Field::new("c", DataType::Int64, false),
            ])
            .unwrap(),
        )
    }

    fn col(i: usize) -> Expr {
        Expr::Column {
            index: i,
            data_type: DataType::Int64,
            nullable: false,
        }
    }

    #[test]
    fn projection_pushes_only_referenced_columns_into_the_scan() {
        let scan =
            LogicalPlanBuilder::scan("t", Arc::new(MemoryTableSource::new(wide_schema(), vec![])))
                .build();
        let out_schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap());
        let plan = LogicalPlan::Projection {
            input: scan,
            exprs: vec![col(0)],
            schema: out_schema,
        };
        let result = ProjectionPushdown.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Projection { input, .. } => match input.as_ref() {
                LogicalPlan::TableScan { projection, .. } => {
                    assert_eq!(projection.as_deref(), Some(&[0usize][..]));
                }
                other => panic!("expected TableScan, got {other:?}"),
            },
            other => panic!("expected Projection, got {other:?}"),
        }
    }

    #[test]
    fn does_not_push_when_every_column_is_needed() {
        let scan =
            LogicalPlanBuilder::scan("t", Arc::new(MemoryTableSource::new(wide_schema(), vec![])))
                .build();
        let plan = LogicalPlan::Projection {
            input: scan,
            exprs: vec![col(0), col(1), col(2)],
            schema: wide_schema(),
        };
        let result = ProjectionPushdown.apply(plan, &NoStatistics).unwrap();
        assert!(!result.is_yes());
    }
}
