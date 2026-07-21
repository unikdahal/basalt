//! Shared helper for rules that rewrite every `Expr` a `LogicalPlan` node
//! carries (constant folding, expression simplification) without touching
//! plan structure — not part of the LLD's module list, factored out to
//! avoid duplicating the same per-variant `Expr`-field enumeration twice.

use crate::error::Result;
use crate::expr::expr::Expr;
use crate::logical_plan::plan::LogicalPlan;
use crate::optimizer::tree_node::Transformed;

/// Applies `f` to every `Expr` this plan node directly carries (not its
/// children's expressions — the caller's rule runs bottom-up via
/// `TreeNode::transform_up`, so children were already visited).
pub fn map_all_exprs(
    plan: LogicalPlan,
    f: impl Fn(Expr) -> Result<Expr>,
) -> Result<Transformed<LogicalPlan>> {
    let mut changed = false;
    let mut apply = |e: Expr| -> Result<Expr> {
        let rewritten = f(e.clone())?;
        if rewritten != e {
            changed = true;
        }
        Ok(rewritten)
    };

    let mapped = match plan {
        LogicalPlan::Projection {
            input,
            exprs,
            schema,
        } => LogicalPlan::Projection {
            input,
            exprs: exprs.into_iter().map(&mut apply).collect::<Result<_>>()?,
            schema,
        },
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input,
            predicate: apply(predicate)?,
        },
        LogicalPlan::Join {
            left,
            right,
            on,
            filter,
            join_type,
            schema,
        } => {
            let on = on
                .into_iter()
                .map(|(l, r)| Ok((apply(l)?, apply(r)?)))
                .collect::<Result<Vec<_>>>()?;
            let filter = filter.map(&mut apply).transpose()?;
            LogicalPlan::Join {
                left,
                right,
                on,
                filter,
                join_type,
                schema,
            }
        }
        LogicalPlan::TableScan {
            table_name,
            source,
            projection,
            filters,
            schema,
        } => LogicalPlan::TableScan {
            table_name,
            source,
            projection,
            filters: filters.into_iter().map(&mut apply).collect::<Result<_>>()?,
            schema,
        },
        LogicalPlan::Sort { input, exprs } => LogicalPlan::Sort {
            input,
            exprs: exprs
                .into_iter()
                .map(|s| {
                    Ok(crate::logical_plan::plan::SortExpr {
                        expr: apply(s.expr)?,
                        options: s.options,
                    })
                })
                .collect::<Result<_>>()?,
        },
        other => other,
    };

    Ok(if changed {
        Transformed::Yes(mapped)
    } else {
        Transformed::No(mapped)
    })
}

/// Splits `predicate` on top-level `AND`s into its conjuncts (`a AND b AND
/// c` -> `[a, b, c]`); a non-`AND` expression is a single conjunct. Shared
/// by `EliminateCrossJoin` and `PredicatePushdown`, which both need to
/// reason about a `WHERE` clause conjunct-by-conjunct rather than as one
/// opaque expression.
pub fn flatten_conjuncts(predicate: Expr) -> Vec<Expr> {
    match predicate {
        Expr::Binary {
            left,
            op: crate::types::coercion::BinaryOp::And,
            right,
        } => {
            let mut out = flatten_conjuncts(*left);
            out.extend(flatten_conjuncts(*right));
            out
        }
        other => vec![other],
    }
}

/// The inverse of `flatten_conjuncts`: ANDs a non-empty list of conjuncts
/// back into one expression.
///
/// # Panics
/// Panics if `conjuncts` is empty — every call site only reaches here with
/// at least one leftover conjunct to rebuild.
pub fn rebuild_conjuncts(mut conjuncts: Vec<Expr>) -> Expr {
    let mut acc = conjuncts.remove(0);
    for c in conjuncts {
        acc = Expr::Binary {
            left: Box::new(acc),
            op: crate::types::coercion::BinaryOp::And,
            right: Box::new(c),
        };
    }
    acc
}

/// Every column index `expr` references (via `TreeNode::visit`).
pub fn referenced_columns(expr: &Expr) -> Vec<usize> {
    use crate::optimizer::tree_node::{TreeNode, VisitRecursion};
    let mut indices = Vec::new();
    let _ = expr.visit(&mut |node| {
        if let Expr::Column { index, .. } = node {
            indices.push(*index);
        }
        Ok(VisitRecursion::Continue)
    });
    indices
}

/// If `conjunct` is `Column(i) = Column(j)` with exactly one of `i`/`j`
/// below `left_width` (a left-side column) and the other at or above it (a
/// right-side column), returns `(left_expr, right_expr)` with the
/// right-side column's index rebased to be relative to the right schema
/// alone — the shape `Join.on` expects (each side's `Expr` indexes into
/// that side's own schema, not the joined one). Shared by
/// `EliminateCrossJoin` and `PredicatePushdown`.
pub fn split_equi_join_conjunct(conjunct: &Expr, left_width: usize) -> Option<(Expr, Expr)> {
    let Expr::Binary {
        left,
        op: crate::types::coercion::BinaryOp::Eq,
        right,
    } = conjunct
    else {
        return None;
    };
    let (
        Expr::Column {
            index: i,
            data_type: dt_i,
            nullable: n_i,
        },
        Expr::Column {
            index: j,
            data_type: dt_j,
            nullable: n_j,
        },
    ) = (left.as_ref(), right.as_ref())
    else {
        return None;
    };

    let i_is_left = *i < left_width;
    let j_is_left = *j < left_width;
    if i_is_left == j_is_left {
        return None; // Both columns on the same side — not a join edge.
    }
    let (left_col, right_col) = if i_is_left {
        (
            Expr::Column {
                index: *i,
                data_type: *dt_i,
                nullable: *n_i,
            },
            Expr::Column {
                index: *j - left_width,
                data_type: *dt_j,
                nullable: *n_j,
            },
        )
    } else {
        (
            Expr::Column {
                index: *j,
                data_type: *dt_j,
                nullable: *n_j,
            },
            Expr::Column {
                index: *i - left_width,
                data_type: *dt_i,
                nullable: *n_i,
            },
        )
    };
    Some((left_col, right_col))
}

/// Rebases every `Column` reference in `expr` by subtracting `offset` from
/// its index — used to convert a column index relative to a joined
/// (concatenated) schema into one relative to just the right side's own
/// schema.
///
/// # Errors
/// Propagates a rebuild error from `Expr::with_new_children` (arity
/// mismatches don't occur for a well-formed tree).
pub fn rebase_columns(expr: Expr, offset: usize) -> Result<Expr> {
    use crate::optimizer::tree_node::TreeNode;
    let result = expr.transform_up(&mut |e| {
        Ok(match e {
            Expr::Column {
                index,
                data_type,
                nullable,
            } => Transformed::Yes(Expr::Column {
                index: index - offset,
                data_type,
                nullable,
            }),
            other => Transformed::No(other),
        })
    })?;
    Ok(result.into_inner())
}
