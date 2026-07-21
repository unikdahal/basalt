//! Constant folding — `2 * 3 -> 6`, `CAST('5' AS INT) -> 5`. See
//! design-docs/basalt-phase3-lld.md §5.3.
//!
//! Folds bottom-up over `Expr` (via `TreeNode::transform_up`, so a nested
//! `(2 + 3) * (4 - 1)` folds its subexpressions before the parent sees
//! them). Must not panic at plan time on overflow (`i64::MAX + 1`) — the
//! expression is returned unfolded instead, and the overflow becomes a
//! runtime error only if that branch of the plan actually executes, exactly
//! matching Phase 1/2's own checked-arithmetic-errors-don't-panic
//! discipline.

use crate::error::Result;
use crate::expr::expr::Expr;
use crate::logical_plan::plan::LogicalPlan;
use crate::optimizer::rule::{ApplyOrder, OptimizerContext, OptimizerRule};
use crate::optimizer::tree_node::{Transformed, TreeNode};
use crate::types::coercion::{BinaryOp, UnaryOp};
use crate::types::value::Value;

#[derive(Debug)]
pub struct ConstantFolding;

impl OptimizerRule for ConstantFolding {
    fn name(&self) -> &str {
        "constant_folding"
    }

    fn apply_order(&self) -> ApplyOrder {
        ApplyOrder::BottomUp
    }

    fn apply(&self, plan: LogicalPlan, _ctx: &dyn OptimizerContext) -> Result<Transformed<LogicalPlan>> {
        super::common::map_all_exprs(plan, fold_one)
    }
}

fn fold_one(expr: Expr) -> Result<Expr> {
    let result = expr.transform_up(&mut |e| Ok(fold_node(e)))?;
    Ok(result.into_inner())
}

fn fold_node(expr: Expr) -> Transformed<Expr> {
    match &expr {
        Expr::Binary { left, op, right } => {
            if let (Expr::Literal(l), Expr::Literal(r)) = (left.as_ref(), right.as_ref()) {
                if let Some(folded) = fold_binary(l, *op, r) {
                    return Transformed::Yes(Expr::Literal(folded));
                }
            }
            Transformed::No(expr)
        }
        Expr::Unary { op, expr: inner } => {
            if let Expr::Literal(v) = inner.as_ref() {
                if let Some(folded) = fold_unary(*op, v) {
                    return Transformed::Yes(Expr::Literal(folded));
                }
            }
            Transformed::No(expr)
        }
        Expr::Cast { expr: inner, to } => {
            if let Expr::Literal(v) = inner.as_ref() {
                if let Ok(folded) = v.cast_to(*to) {
                    return Transformed::Yes(Expr::Literal(folded));
                }
            }
            Transformed::No(expr)
        }
        Expr::IsNull(inner) => {
            if let Expr::Literal(v) = inner.as_ref() {
                return Transformed::Yes(Expr::Literal(Value::Boolean(matches!(v, Value::Null))));
            }
            Transformed::No(expr)
        }
        Expr::IsNotNull(inner) => {
            if let Expr::Literal(v) = inner.as_ref() {
                return Transformed::Yes(Expr::Literal(Value::Boolean(!matches!(v, Value::Null))));
            }
            Transformed::No(expr)
        }
        _ => Transformed::No(expr),
    }
}

/// Returns `None` (leaves the expression unfolded) on overflow/division by
/// zero/a null operand this simple folder doesn't special-case — never
/// panics.
fn fold_binary(l: &Value, op: BinaryOp, r: &Value) -> Option<Value> {
    if matches!(l, Value::Null) || matches!(r, Value::Null) {
        return None; // Three-valued logic: let eval's null handling apply, don't guess here.
    }
    match (l, op, r) {
        (Value::Int64(a), BinaryOp::Add, Value::Int64(b)) => a.checked_add(*b).map(Value::Int64),
        (Value::Int64(a), BinaryOp::Sub, Value::Int64(b)) => a.checked_sub(*b).map(Value::Int64),
        (Value::Int64(a), BinaryOp::Mul, Value::Int64(b)) => a.checked_mul(*b).map(Value::Int64),
        (Value::Int64(a), BinaryOp::Div, Value::Int64(b)) if *b != 0 => {
            a.checked_div(*b).map(Value::Int64)
        }
        (Value::Int64(a), BinaryOp::Mod, Value::Int64(b)) if *b != 0 => {
            a.checked_rem(*b).map(Value::Int64)
        }
        (Value::Float64(a), BinaryOp::Add, Value::Float64(b)) => Some(Value::Float64(a + b)),
        (Value::Float64(a), BinaryOp::Sub, Value::Float64(b)) => Some(Value::Float64(a - b)),
        (Value::Float64(a), BinaryOp::Mul, Value::Float64(b)) => Some(Value::Float64(a * b)),
        (Value::Float64(a), BinaryOp::Div, Value::Float64(b)) if *b != 0.0 => {
            Some(Value::Float64(a / b))
        }
        (Value::Int64(a), BinaryOp::Eq, Value::Int64(b)) => Some(Value::Boolean(a == b)),
        (Value::Int64(a), BinaryOp::NotEq, Value::Int64(b)) => Some(Value::Boolean(a != b)),
        (Value::Int64(a), BinaryOp::Lt, Value::Int64(b)) => Some(Value::Boolean(a < b)),
        (Value::Int64(a), BinaryOp::LtEq, Value::Int64(b)) => Some(Value::Boolean(a <= b)),
        (Value::Int64(a), BinaryOp::Gt, Value::Int64(b)) => Some(Value::Boolean(a > b)),
        (Value::Int64(a), BinaryOp::GtEq, Value::Int64(b)) => Some(Value::Boolean(a >= b)),
        (Value::Boolean(a), BinaryOp::And, Value::Boolean(b)) => Some(Value::Boolean(*a && *b)),
        (Value::Boolean(a), BinaryOp::Or, Value::Boolean(b)) => Some(Value::Boolean(*a || *b)),
        _ => None,
    }
}

fn fold_unary(op: UnaryOp, v: &Value) -> Option<Value> {
    if matches!(v, Value::Null) {
        return None;
    }
    match (op, v) {
        (UnaryOp::Neg, Value::Int64(a)) => a.checked_neg().map(Value::Int64),
        (UnaryOp::Neg, Value::Float64(a)) => Some(Value::Float64(-a)),
        (UnaryOp::Not, Value::Boolean(a)) => Some(Value::Boolean(!a)),
        _ => None,
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

    #[test]
    fn folds_nested_arithmetic() {
        let expr = Expr::Binary {
            left: Box::new(Expr::Binary {
                left: Box::new(Expr::Literal(Value::Int64(2))),
                op: BinaryOp::Add,
                right: Box::new(Expr::Literal(Value::Int64(3))),
            }),
            op: BinaryOp::Mul,
            right: Box::new(Expr::Literal(Value::Int64(4))),
        };
        let folded = fold_one(expr).unwrap();
        assert_eq!(folded, Expr::Literal(Value::Int64(20)));
    }

    #[test]
    fn overflow_leaves_expression_unfolded_not_panicking() {
        let expr = Expr::Binary {
            left: Box::new(Expr::Literal(Value::Int64(i64::MAX))),
            op: BinaryOp::Add,
            right: Box::new(Expr::Literal(Value::Int64(1))),
        };
        let folded = fold_one(expr.clone()).unwrap();
        assert_eq!(folded, expr);
    }

    #[test]
    fn division_by_zero_leaves_expression_unfolded() {
        let expr = Expr::Binary {
            left: Box::new(Expr::Literal(Value::Int64(10))),
            op: BinaryOp::Div,
            right: Box::new(Expr::Literal(Value::Int64(0))),
        };
        let folded = fold_one(expr.clone()).unwrap();
        assert_eq!(folded, expr);
    }

    #[test]
    fn null_operand_is_not_folded_here() {
        let expr = Expr::Binary {
            left: Box::new(Expr::Literal(Value::Null)),
            op: BinaryOp::Add,
            right: Box::new(Expr::Literal(Value::Int64(1))),
        };
        let folded = fold_one(expr.clone()).unwrap();
        assert_eq!(folded, expr);
    }

    #[test]
    fn is_null_on_a_literal_folds_to_a_boolean() {
        let folded = fold_one(Expr::IsNull(Box::new(Expr::Literal(Value::Null)))).unwrap();
        assert_eq!(folded, Expr::Literal(Value::Boolean(true)));
    }

    #[test]
    fn rule_folds_expressions_inside_a_filter_node() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let plan = LogicalPlanBuilder::scan("t", source)
            .filter(Expr::Binary {
                left: Box::new(Expr::Literal(Value::Int64(1))),
                op: BinaryOp::Add,
                right: Box::new(Expr::Literal(Value::Int64(1))),
            })
            .build();
        let result = crate::optimizer::rules::common::map_all_exprs(plan.as_ref().clone(), fold_one).unwrap();
        assert!(result.is_yes());
        match result.into_inner() {
            LogicalPlan::Filter { predicate, .. } => {
                assert_eq!(predicate, Expr::Literal(Value::Int64(2)));
            }
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn rule_is_idempotent() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let plan = LogicalPlanBuilder::scan("t", source)
            .filter(Expr::Literal(Value::Boolean(true)))
            .build();
        let rule = ConstantFolding;
        let once = rule
            .apply(plan.as_ref().clone(), &NoStatistics)
            .unwrap()
            .into_inner();
        let twice = rule.apply(once.clone(), &NoStatistics).unwrap();
        assert!(!twice.is_yes(), "second application should find nothing left to fold");
    }
}
