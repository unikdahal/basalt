//! `CommonSubexpressionElimination` — find a repeated subexpression across
//! a `Projection`'s output list, compute it once. See
//! design-docs/basalt-phase3-lld.md §5.3.
//!
//! **Narrower than the LLD's one-liner** ("find repeated subtrees, compute
//! once... `Arc` sharing from Phase 2's plan design makes this natural"):
//! `Expr`'s children are `Box`, not `Arc` (only `LogicalPlan` shares
//! subtrees via `Arc`), so there's no free structural sharing to exploit
//! inside a single expression tree. What's implemented here instead:
//! detect a non-trivial subexpression repeated *across* a `Projection`'s
//! output expressions (e.g. `SELECT a+b, (a+b)*2`), and rewrite it into a
//! lower `Projection` that computes the shared subexpression once as an
//! extra column, with the original expressions referencing that column
//! instead of recomputing it. One repeated subexpression is eliminated per
//! rule application; the fixed-point pass manager (§5.2) catches any
//! further opportunities on the next iteration — a real, correct
//! optimization, just a smaller scope than "any repeated subtree anywhere
//! in the plan."

use std::sync::Arc;

use crate::error::Result;
use crate::expr::expr::Expr;
use crate::logical_plan::plan::LogicalPlan;
use crate::optimizer::rule::{ApplyOrder, OptimizerContext, OptimizerRule};
use crate::optimizer::tree_node::{Transformed, TreeNode};
use crate::types::schema::{Field, Schema};

#[derive(Debug)]
pub struct CommonSubexprEliminate;

impl OptimizerRule for CommonSubexprEliminate {
    fn name(&self) -> &str {
        "common_subexpression_elimination"
    }

    fn apply_order(&self) -> ApplyOrder {
        ApplyOrder::BottomUp
    }

    fn apply(
        &self,
        plan: LogicalPlan,
        _ctx: &dyn OptimizerContext,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Projection {
            input,
            exprs,
            schema,
        } = plan
        else {
            return Ok(Transformed::No(plan));
        };

        let Some(repeated) = find_most_repeated_subexpr(&exprs) else {
            return Ok(Transformed::No(LogicalPlan::Projection {
                input,
                exprs,
                schema,
            }));
        };

        let Ok(data_type) = repeated.data_type() else {
            return Ok(Transformed::No(LogicalPlan::Projection {
                input,
                exprs,
                schema,
            }));
        };
        let new_col_index = input.schema().fields().len();
        let nullable = repeated.nullable();

        let mut lower_fields: Vec<Field> = input.schema().fields().to_vec();
        lower_fields.push(Field::new("__cse", data_type, nullable));
        let lower_schema = Arc::new(Schema::new_allow_duplicate_names(lower_fields));

        let lower_exprs: Vec<Expr> = (0..input.schema().fields().len())
            .map(|i| {
                let f = &input.schema().fields()[i];
                Expr::Column {
                    index: i,
                    data_type: f.data_type,
                    nullable: f.nullable,
                }
            })
            .chain(std::iter::once(repeated.clone()))
            .collect();
        let lower = LogicalPlan::Projection {
            input,
            exprs: lower_exprs,
            schema: lower_schema,
        };

        let replacement = Expr::Column {
            index: new_col_index,
            data_type,
            nullable,
        };
        let rewritten_exprs = exprs
            .into_iter()
            .map(|e| replace_subexpr(e, &repeated, &replacement))
            .collect::<Result<Vec<_>>>()?;

        Ok(Transformed::Yes(LogicalPlan::Projection {
            input: Arc::new(lower),
            exprs: rewritten_exprs,
            schema,
        }))
    }
}

/// A subexpression is worth sharing if it's not already a bare leaf
/// (`Column`/`Literal`) — sharing those would add a column for something
/// cheaper than the column reference itself.
fn is_shareable(expr: &Expr) -> bool {
    !matches!(expr, Expr::Column { .. } | Expr::Literal(_))
}

fn all_subexprs(expr: &Expr, out: &mut Vec<Expr>) {
    if is_shareable(expr) {
        out.push(expr.clone());
    }
    for child in expr.children() {
        all_subexprs(child, out);
    }
}

/// Finds the subexpression appearing most often (>1) across `exprs`,
/// preferring larger (more node-count) subexpressions when counts tie, so a
/// bigger shared computation is hoisted before a smaller one nested inside
/// it — avoids the rule immediately re-finding the same opportunity at a
/// finer grain on the very next iteration.
fn find_most_repeated_subexpr(exprs: &[Expr]) -> Option<Expr> {
    let mut candidates: Vec<Expr> = Vec::new();
    for e in exprs {
        all_subexprs(e, &mut candidates);
    }

    let mut counted: Vec<(Expr, usize)> = Vec::new();
    for c in &candidates {
        if let Some(entry) = counted.iter_mut().find(|(e, _)| e == c) {
            entry.1 += 1;
        } else {
            counted.push((c.clone(), 1));
        }
    }

    counted
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .max_by_key(|(e, count)| (*count, node_count(e)))
        .map(|(e, _)| e)
}

fn node_count(expr: &Expr) -> usize {
    1 + expr.children().iter().map(|c| node_count(c)).sum::<usize>()
}

/// Replaces every occurrence of `target` (by structural equality) within
/// `expr` with `replacement`, via `TreeNode::transform_up` so nested
/// occurrences are all caught.
fn replace_subexpr(expr: Expr, target: &Expr, replacement: &Expr) -> Result<Expr> {
    let result = expr.transform_up(&mut |e| {
        Ok(if &e == target {
            Transformed::Yes(replacement.clone())
        } else {
            Transformed::No(e)
        })
    })?;
    Ok(result.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::optimizer::rule::NoStatistics;
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::types::coercion::BinaryOp;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field as SField, Schema as SSchema};

    fn schema() -> crate::types::schema::SchemaRef {
        Arc::new(
            SSchema::new(vec![
                SField::new("a", DataType::Int64, false),
                SField::new("b", DataType::Int64, false),
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

    fn a_plus_b() -> Expr {
        Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Add,
            right: Box::new(col(1)),
        }
    }

    #[test]
    fn hoists_a_subexpression_repeated_across_projection_outputs() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let scan = LogicalPlanBuilder::scan("t", source).build();
        let exprs = vec![
            a_plus_b(),
            Expr::Binary {
                left: Box::new(a_plus_b()),
                op: BinaryOp::Mul,
                right: Box::new(Expr::Literal(crate::types::value::Value::Int64(2))),
            },
        ];
        let out_schema = Arc::new(
            SSchema::new(vec![
                SField::new("sum", DataType::Int64, false),
                SField::new("doubled", DataType::Int64, false),
            ])
            .unwrap(),
        );
        let plan = LogicalPlan::Projection {
            input: scan,
            exprs,
            schema: out_schema,
        };
        let result = CommonSubexprEliminate.apply(plan, &NoStatistics).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Projection { input, exprs, .. } => {
                // Both output expressions should now just reference the
                // hoisted column (index 2, after a=0, b=1).
                assert_eq!(exprs[0], col(2));
                match &input.as_ref() {
                    LogicalPlan::Projection {
                        exprs: lower_exprs, ..
                    } => {
                        assert_eq!(lower_exprs.len(), 3);
                        assert_eq!(lower_exprs[2], a_plus_b());
                    }
                    other => panic!("expected a lower Projection, got {other:?}"),
                }
            }
            other => panic!("expected Projection, got {other:?}"),
        }
    }

    #[test]
    fn no_repeated_subexpression_leaves_plan_unchanged() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let scan = LogicalPlanBuilder::scan("t", source).build();
        let plan = LogicalPlan::Projection {
            input: scan,
            exprs: vec![col(0), col(1)],
            schema: schema(),
        };
        let result = CommonSubexprEliminate.apply(plan, &NoStatistics).unwrap();
        assert!(!result.is_yes());
    }
}
