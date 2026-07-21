//! `PredicatePushdown` — move filters as close to the scan as possible, so
//! rows die before anything expensive touches them. Usually the single
//! largest rule-based win. See design-docs/basalt-phase3-lld.md §6.1.
//!
//! The legality table this implements (get it right, it's where the bugs
//! are):
//!
//! | Push through | Rule |
//! |---|---|
//! | `Projection` | Only if the predicate's columns are all produced by the projection as bare passthrough columns (not computed expressions) |
//! | `Filter` | Always — merged by `MergeFilters`, which runs before this rule |
//! | Inner `Join` | Left-only columns -> push left. Right-only -> push right. Both -> becomes a join condition |
//! | Left `Join` | Push to the left (preserved) side only. Never to the right |
//! | Right `Join` | Mirror image |
//! | Full `Join` | Neither side |
//! | `Aggregate` | Only predicates on grouping columns (`HAVING` on group keys). Never on aggregate outputs |
//! | `Sort` | Yes — filtering doesn't change row order, only which rows exist |
//! | `Limit` | **No** — the filter would change which rows survive the limit |
//! | `TableScan` | Recorded into `TableScan.filters` for statistics-based pruning (§6.4) — **not** a substitute for row-level filtering, since Phase 2's scan executors don't apply `filters` themselves; the `Filter` node above the scan stays |
//!
//! **The outer-join restriction deserves its own paragraph, because it's
//! subtle and the bug is silent.** Pushing a predicate into a `LEFT JOIN`'s
//! null-producing (right) side filters rows *before* the join manufactures
//! nulls for unmatched left rows, so a left row whose real match just got
//! filtered out of the right side now finds no match at all and appears
//! null-padded instead of correctly not appearing (or appearing un-padded).
//! The same predicate applied *after* the join sees an actual, non-padded
//! row and rejects it outright — the two orderings produce different
//! results. Pushing into the *preserved* (left) side has no such problem:
//! a left row's fate is the same whether the predicate runs before or
//! after the join, since the join can only ever pass every surviving left
//! row through (matched or null-padded).

use std::sync::Arc;

use crate::error::Result;
use crate::expr::expr::Expr;
use crate::logical_plan::plan::{JoinType, LogicalPlan};
use crate::optimizer::rule::{ApplyOrder, OptimizerContext, OptimizerRule};
use crate::optimizer::tree_node::Transformed;

use super::common::{
    flatten_conjuncts, rebase_columns, rebuild_conjuncts, referenced_columns,
    split_equi_join_conjunct,
};

#[derive(Debug)]
pub struct PredicatePushdown;

impl OptimizerRule for PredicatePushdown {
    fn name(&self) -> &str {
        "predicate_pushdown"
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

        match input.as_ref() {
            LogicalPlan::TableScan { .. } => push_into_scan(input, predicate),
            LogicalPlan::Projection { .. } => push_through_projection(input, predicate),
            LogicalPlan::Join { .. } => push_through_join(input, predicate),
            LogicalPlan::Aggregate { .. } => push_through_aggregate(input, predicate),
            LogicalPlan::Sort {
                input: sort_input,
                exprs,
            } => {
                // Filtering doesn't change row order, only which rows
                // exist — always safe, and reduces the number of rows Sort
                // has to touch.
                let exprs = exprs.clone();
                let sort_input = sort_input.clone();
                Ok(Transformed::Yes(LogicalPlan::Sort {
                    input: Arc::new(LogicalPlan::Filter {
                        input: sort_input,
                        predicate,
                    }),
                    exprs,
                }))
            }
            // Limit: never push through — see the module doc comment.
            // Everything else (EmptyRelation, another Filter already
            // merged by MergeFilters): nothing to do.
            _ => Ok(Transformed::No(LogicalPlan::Filter { input, predicate })),
        }
    }
}

/// Records `predicate`'s conjuncts into `TableScan.filters` for
/// statistics-based pruning, but — critically — keeps the `Filter` node
/// above the scan unchanged, since Phase 2's scan executors don't apply
/// `filters` themselves (see the module doc comment).
fn push_into_scan(input: Arc<LogicalPlan>, predicate: Expr) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::TableScan {
        table_name,
        source,
        projection,
        filters,
        schema,
    } = input.as_ref()
    else {
        unreachable!("caller matched TableScan");
    };
    let new_conjuncts: Vec<Expr> = flatten_conjuncts(predicate.clone())
        .into_iter()
        .filter(|c| !filters.contains(c))
        .collect();
    if new_conjuncts.is_empty() {
        return Ok(Transformed::No(LogicalPlan::Filter { input, predicate }));
    }
    let mut filters = filters.clone();
    filters.extend(new_conjuncts);
    let new_scan = LogicalPlan::TableScan {
        table_name: table_name.clone(),
        source: source.clone(),
        projection: projection.clone(),
        filters,
        schema: schema.clone(),
    };
    Ok(Transformed::Yes(LogicalPlan::Filter {
        input: Arc::new(new_scan),
        predicate,
    }))
}

/// Pushes the conjuncts of `predicate` whose every referenced column maps
/// to a bare passthrough `Column` in the projection below `input` — those
/// don't need the projection's computation to have happened yet. Any
/// conjunct referencing a computed output column stays above the
/// projection, unpushed.
fn push_through_projection(
    input: Arc<LogicalPlan>,
    predicate: Expr,
) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Projection {
        input: proj_input,
        exprs,
        schema,
    } = input.as_ref()
    else {
        unreachable!("caller matched Projection");
    };

    let conjuncts = flatten_conjuncts(predicate);
    let mut pushable = Vec::new();
    let mut residual = Vec::new();
    for conjunct in conjuncts {
        match rewrite_through_passthrough(&conjunct, exprs) {
            Some(rewritten) => pushable.push(rewritten),
            None => residual.push(conjunct),
        }
    }

    if pushable.is_empty() {
        return Ok(Transformed::No(LogicalPlan::Filter {
            input,
            predicate: rebuild_conjuncts(residual),
        }));
    }

    let new_proj_input = Arc::new(LogicalPlan::Filter {
        input: proj_input.clone(),
        predicate: rebuild_conjuncts(pushable),
    });
    let new_projection = LogicalPlan::Projection {
        input: new_proj_input,
        exprs: exprs.clone(),
        schema: schema.clone(),
    };
    Ok(Transformed::Yes(if residual.is_empty() {
        new_projection
    } else {
        LogicalPlan::Filter {
            input: Arc::new(new_projection),
            predicate: rebuild_conjuncts(residual),
        }
    }))
}

/// If every column `conjunct` references maps to a bare `Column` in
/// `proj_exprs` (i.e. output column `i` is exactly `proj_exprs[i]` and that
/// expression is itself a plain `Column`, not something computed), rewrites
/// `conjunct`'s column references to point at the underlying input columns
/// and returns it. Otherwise returns `None`.
fn rewrite_through_passthrough(conjunct: &Expr, proj_exprs: &[Expr]) -> Option<Expr> {
    for col in referenced_columns(conjunct) {
        match proj_exprs.get(col) {
            Some(Expr::Column { .. }) => continue,
            _ => return None,
        }
    }
    rewrite_columns(conjunct.clone(), &|i| match proj_exprs[i] {
        Expr::Column { index, .. } => index,
        _ => unreachable!("checked above"),
    })
}

/// Rewrites every `Column` index in `expr` via `f`.
fn rewrite_columns(expr: Expr, f: &impl Fn(usize) -> usize) -> Option<Expr> {
    use crate::optimizer::tree_node::TreeNode;
    expr.transform_up(&mut |e| {
        Ok(match e {
            Expr::Column {
                index,
                data_type,
                nullable,
            } => Transformed::Yes(Expr::Column {
                index: f(index),
                data_type,
                nullable,
            }),
            other => Transformed::No(other),
        })
    })
    .ok()
    .map(Transformed::into_inner)
}

fn push_through_join(input: Arc<LogicalPlan>, predicate: Expr) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Join {
        left,
        right,
        on,
        filter,
        join_type,
        schema,
    } = input.as_ref()
    else {
        unreachable!("caller matched Join");
    };
    if *join_type == JoinType::Full {
        // Neither side — see the module doc comment.
        return Ok(Transformed::No(LogicalPlan::Filter { input, predicate }));
    }

    let left_width = left.schema().fields().len();
    let conjuncts = flatten_conjuncts(predicate);
    let mut push_left = Vec::new();
    let mut push_right = Vec::new();
    let mut promote_to_on = Vec::new();
    let mut residual = Vec::new();

    for conjunct in conjuncts {
        let cols = referenced_columns(&conjunct);
        let touches_left = cols.iter().any(|&c| c < left_width);
        let touches_right = cols.iter().any(|&c| c >= left_width);

        match (touches_left, touches_right, *join_type) {
            (true, false, _) => push_left.push(conjunct),
            // Right-only: safe for Inner/Right (right is preserved or
            // both sides matched); never safe for Left (right is the
            // null-producing side).
            (false, true, JoinType::Left) => residual.push(conjunct),
            (false, true, _) => match rebase_columns(conjunct.clone(), left_width) {
                Ok(rebased) => push_right.push(rebased),
                Err(_) => residual.push(conjunct),
            },
            // Both sides: only safe to relocate at all for Inner (Left/
            // Right must keep it as a residual post-join Filter, since
            // Join.filter is applied before null-padding but a mixed
            // conjunct on an outer join was written expecting to see the
            // padded result).
            (true, true, JoinType::Inner) => {
                match split_equi_join_conjunct(&conjunct, left_width) {
                    Some(pair) => promote_to_on.push(pair),
                    // Not a promotable equi-join edge: keep it as a residual
                    // join filter (evaluated against the combined schema,
                    // which is exactly what `Join.filter` already is for an
                    // Inner join).
                    None => residual.push(conjunct),
                }
            }
            _ => residual.push(conjunct),
        }
    }

    if push_left.is_empty() && push_right.is_empty() && promote_to_on.is_empty() {
        return Ok(Transformed::No(LogicalPlan::Filter {
            input,
            predicate: rebuild_conjuncts(residual),
        }));
    }

    let new_left = if push_left.is_empty() {
        left.clone()
    } else {
        Arc::new(LogicalPlan::Filter {
            input: left.clone(),
            predicate: rebuild_conjuncts(push_left),
        })
    };
    let new_right = if push_right.is_empty() {
        right.clone()
    } else {
        Arc::new(LogicalPlan::Filter {
            input: right.clone(),
            predicate: rebuild_conjuncts(push_right),
        })
    };

    let mut new_on = on.clone();
    new_on.extend(promote_to_on);
    // Any residual mixed-side conjunct on an Inner join becomes part of
    // Join.filter (still evaluated over the combined schema, matching
    // where it already was); Left/Right joins never reach here with a
    // mixed conjunct in `residual` from the (true, true, ...) arm — only
    // via the explicit (false, true, JoinType::Left) arm, which is a
    // single-side (right-only) conjunct kept untouched as a post-join
    // Filter, not merged into Join.filter.
    let new_join_filter = if *join_type == JoinType::Inner && !residual.is_empty() {
        let combined = match filter {
            Some(existing) => rebuild_conjuncts({
                let mut v = residual.clone();
                v.insert(0, existing.clone());
                v
            }),
            None => rebuild_conjuncts(residual.clone()),
        };
        residual.clear();
        Some(combined)
    } else {
        filter.clone()
    };

    let new_join = LogicalPlan::Join {
        left: new_left,
        right: new_right,
        on: new_on,
        filter: new_join_filter,
        join_type: *join_type,
        schema: schema.clone(),
    };

    Ok(Transformed::Yes(if residual.is_empty() {
        new_join
    } else {
        LogicalPlan::Filter {
            input: Arc::new(new_join),
            predicate: rebuild_conjuncts(residual),
        }
    }))
}

/// Pushes conjuncts referencing only grouping columns (a `HAVING` clause
/// evaluable pre-aggregation) below the `Aggregate`, rewritten to reference
/// the underlying input column — only when the corresponding group
/// expression is itself a bare `Column`. Conjuncts touching any aggregate
/// output stay above, unpushed (`Aggregate` outputs don't exist until the
/// aggregation runs).
fn push_through_aggregate(
    input: Arc<LogicalPlan>,
    predicate: Expr,
) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Aggregate {
        input: agg_input,
        group_expr,
        aggr_expr,
        schema,
    } = input.as_ref()
    else {
        unreachable!("caller matched Aggregate");
    };

    let conjuncts = flatten_conjuncts(predicate);
    let mut pushable = Vec::new();
    let mut residual = Vec::new();
    for conjunct in conjuncts {
        let cols = referenced_columns(&conjunct);
        let all_group_cols = cols.iter().all(|&c| c < group_expr.len());
        if all_group_cols
            && cols
                .iter()
                .all(|&c| matches!(group_expr[c], Expr::Column { .. }))
        {
            if let Some(rewritten) = rewrite_columns(conjunct.clone(), &|i| match group_expr[i] {
                Expr::Column { index, .. } => index,
                _ => unreachable!("checked above"),
            }) {
                pushable.push(rewritten);
                continue;
            }
        }
        residual.push(conjunct);
    }

    if pushable.is_empty() {
        return Ok(Transformed::No(LogicalPlan::Filter {
            input,
            predicate: rebuild_conjuncts(residual),
        }));
    }

    let new_agg_input = Arc::new(LogicalPlan::Filter {
        input: agg_input.clone(),
        predicate: rebuild_conjuncts(pushable),
    });
    let new_agg = LogicalPlan::Aggregate {
        input: new_agg_input,
        group_expr: group_expr.clone(),
        aggr_expr: aggr_expr.clone(),
        schema: schema.clone(),
    };
    Ok(Transformed::Yes(if residual.is_empty() {
        new_agg
    } else {
        LogicalPlan::Filter {
            input: Arc::new(new_agg),
            predicate: rebuild_conjuncts(residual),
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::optimizer::rule::NoStatistics;
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::types::coercion::BinaryOp;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema, SchemaRef};
    use crate::types::value::Value;

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

    fn scan(name: &str) -> Arc<LogicalPlan> {
        LogicalPlanBuilder::scan(name, Arc::new(MemoryTableSource::new(schema(name), vec![])))
            .build()
    }

    fn join(join_type: JoinType) -> LogicalPlan {
        LogicalPlan::Join {
            left: scan("l"),
            right: scan("r"),
            on: vec![(col(0), col(0))],
            filter: None,
            join_type,
            schema: joined_schema(),
        }
    }

    #[test]
    fn pushes_into_table_scan_filters_but_keeps_the_filter_node() {
        let plan = LogicalPlan::Filter {
            input: scan("t"),
            predicate: Expr::Literal(Value::Boolean(true)),
        };
        let result = PredicatePushdown.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Filter { input, .. } => match input.as_ref() {
                LogicalPlan::TableScan { filters, .. } => assert_eq!(filters.len(), 1),
                other => panic!("expected TableScan, got {other:?}"),
            },
            other => panic!("expected Filter, got {other:?}"),
        }
    }

    #[test]
    fn inner_join_left_only_predicate_pushes_to_left_side() {
        let predicate = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal(Value::Int64(5))),
        };
        let plan = LogicalPlan::Filter {
            input: Arc::new(join(JoinType::Inner)),
            predicate,
        };
        let result = PredicatePushdown.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Join { left, .. } => {
                assert!(matches!(left.as_ref(), LogicalPlan::Filter { .. }));
            }
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn left_join_right_only_predicate_is_never_pushed_to_the_right() {
        // The outer-join asymmetry this module's doc comment warns about:
        // a predicate on the right (null-producing) side of a LEFT JOIN
        // must stay a post-join Filter, never become a pre-join filter on
        // the right side or part of Join.filter.
        let predicate = Expr::Binary {
            left: Box::new(col(1)), // column 1 = right side's column 0, in the joined schema
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal(Value::Int64(5))),
        };
        let plan = LogicalPlan::Filter {
            input: Arc::new(join(JoinType::Left)),
            predicate: predicate.clone(),
        };
        let result = PredicatePushdown.apply(plan, &NoStatistics).unwrap();
        assert!(
            !result.is_yes(),
            "must not push a right-side predicate through a LEFT JOIN"
        );
        match result.into_inner() {
            LogicalPlan::Filter {
                input,
                predicate: p,
            } => {
                assert_eq!(p, predicate);
                assert!(matches!(input.as_ref(), LogicalPlan::Join { .. }));
                if let LogicalPlan::Join { right, filter, .. } = input.as_ref() {
                    assert!(
                        matches!(right.as_ref(), LogicalPlan::TableScan { .. }),
                        "right side must stay unfiltered"
                    );
                    assert!(
                        filter.is_none(),
                        "must not become part of Join.filter either"
                    );
                }
            }
            other => panic!("expected Filter, got {other:?}"),
        }
    }

    #[test]
    fn left_join_left_only_predicate_pushes_to_the_preserved_side() {
        let predicate = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal(Value::Int64(5))),
        };
        let plan = LogicalPlan::Filter {
            input: Arc::new(join(JoinType::Left)),
            predicate,
        };
        let result = PredicatePushdown.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Join { left, .. } => {
                assert!(matches!(left.as_ref(), LogicalPlan::Filter { .. }));
            }
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn full_join_never_pushes_either_side() {
        let predicate = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal(Value::Int64(5))),
        };
        let plan = LogicalPlan::Filter {
            input: Arc::new(join(JoinType::Full)),
            predicate,
        };
        let result = PredicatePushdown.apply(plan, &NoStatistics).unwrap();
        assert!(!result.is_yes());
    }

    #[test]
    fn pushes_through_a_passthrough_projection() {
        let projection = LogicalPlan::Projection {
            input: scan("t"),
            exprs: vec![col(0)],
            schema: schema("t"),
        };
        let predicate = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal(Value::Int64(5))),
        };
        let plan = LogicalPlan::Filter {
            input: Arc::new(projection),
            predicate,
        };
        let result = PredicatePushdown.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Projection { input, .. } => {
                assert!(matches!(input.as_ref(), LogicalPlan::Filter { .. }));
            }
            other => panic!("expected Projection, got {other:?}"),
        }
    }

    #[test]
    fn does_not_push_through_a_computed_projection_column() {
        let computed = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Add,
            right: Box::new(Expr::Literal(Value::Int64(1))),
        };
        let projection = LogicalPlan::Projection {
            input: scan("t"),
            exprs: vec![computed],
            schema: schema("t"),
        };
        let predicate = Expr::Binary {
            left: Box::new(col(0)), // references the *computed* output column
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal(Value::Int64(5))),
        };
        let plan = LogicalPlan::Filter {
            input: Arc::new(projection),
            predicate,
        };
        let result = PredicatePushdown.apply(plan, &NoStatistics).unwrap();
        assert!(!result.is_yes());
    }

    #[test]
    fn does_not_push_through_limit() {
        let limit = LogicalPlan::Limit {
            input: scan("t"),
            skip: 0,
            fetch: Some(10),
        };
        let plan = LogicalPlan::Filter {
            input: Arc::new(limit),
            predicate: Expr::Literal(Value::Boolean(true)),
        };
        let result = PredicatePushdown.apply(plan, &NoStatistics).unwrap();
        assert!(!result.is_yes());
    }

    #[test]
    fn pushes_through_sort() {
        let sort = LogicalPlan::Sort {
            input: scan("t"),
            exprs: vec![],
        };
        let predicate = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal(Value::Int64(5))),
        };
        let plan = LogicalPlan::Filter {
            input: Arc::new(sort),
            predicate,
        };
        let result = PredicatePushdown.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Sort { input, .. } => {
                assert!(matches!(input.as_ref(), LogicalPlan::Filter { .. }));
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }
}
