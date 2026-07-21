//! Expression simplification — and this is where the null trap lives. See
//! design-docs/basalt-phase3-lld.md §5.3.
//!
//! `x * 0 -> 0`, `x = x -> true`, and `x - x -> 0` are **only valid when
//! `x` is non-nullable**: `NULL * 0` is `NULL`, not `0`; `NULL = NULL` is
//! `NULL`, not `true`. Three-valued logic reaching into the optimizer is a
//! classic silent-correctness bug class, which is exactly why every
//! nullable-only rewrite here is guarded by `Expr::nullable()` (built in
//! Phase 1, earning its keep here) and property-tested with a nullable
//! input.

use crate::error::Result;
use crate::expr::expr::Expr;
use crate::logical_plan::plan::LogicalPlan;
use crate::optimizer::rule::{ApplyOrder, OptimizerContext, OptimizerRule};
use crate::optimizer::tree_node::{Transformed, TreeNode};
use crate::types::coercion::{BinaryOp, UnaryOp};
use crate::types::value::Value;

#[derive(Debug)]
pub struct SimplifyExpressions;

impl OptimizerRule for SimplifyExpressions {
    fn name(&self) -> &str {
        "simplify_expressions"
    }

    fn apply_order(&self) -> ApplyOrder {
        ApplyOrder::BottomUp
    }

    fn apply(&self, plan: LogicalPlan, _ctx: &dyn OptimizerContext) -> Result<Transformed<LogicalPlan>> {
        super::common::map_all_exprs(plan, simplify_one)
    }
}

fn simplify_one(expr: Expr) -> Result<Expr> {
    let result = expr.transform_up(&mut |e| Ok(simplify_node(e)))?;
    Ok(result.into_inner())
}

fn is_literal_bool(e: &Expr, want: bool) -> bool {
    matches!(e, Expr::Literal(Value::Boolean(b)) if *b == want)
}

fn exprs_equal(a: &Expr, b: &Expr) -> bool {
    a == b
}

fn simplify_node(expr: Expr) -> Transformed<Expr> {
    match &expr {
        // true AND x -> x; false AND x -> false (always valid: AND's
        // short-circuit truth table doesn't depend on x's nullability when
        // the *other* operand is already a known non-null boolean).
        Expr::Binary { left, op: BinaryOp::And, right } => {
            if is_literal_bool(left, true) {
                return Transformed::Yes((**right).clone());
            }
            if is_literal_bool(right, true) {
                return Transformed::Yes((**left).clone());
            }
            if is_literal_bool(left, false) || is_literal_bool(right, false) {
                return Transformed::Yes(Expr::Literal(Value::Boolean(false)));
            }
            Transformed::No(expr)
        }
        // x OR true -> true; false OR x -> x.
        Expr::Binary { left, op: BinaryOp::Or, right } => {
            if is_literal_bool(left, true) || is_literal_bool(right, true) {
                return Transformed::Yes(Expr::Literal(Value::Boolean(true)));
            }
            if is_literal_bool(left, false) {
                return Transformed::Yes((**right).clone());
            }
            if is_literal_bool(right, false) {
                return Transformed::Yes((**left).clone());
            }
            Transformed::No(expr)
        }
        // NOT NOT x -> x.
        Expr::Unary { op: UnaryOp::Not, expr: inner } => {
            if let Expr::Unary { op: UnaryOp::Not, expr: innermost } = inner.as_ref() {
                return Transformed::Yes((**innermost).clone());
            }
            Transformed::No(expr)
        }
        // x + 0 -> x (always valid for integers: NULL + 0 is still NULL,
        // and this rewrite doesn't change that — it just stops computing
        // it).
        Expr::Binary { left, op: BinaryOp::Add, right } => {
            if let Expr::Literal(Value::Int64(0)) = right.as_ref() {
                return Transformed::Yes((**left).clone());
            }
            if let Expr::Literal(Value::Int64(0)) = left.as_ref() {
                return Transformed::Yes((**right).clone());
            }
            Transformed::No(expr)
        }
        // x * 1 -> x (always valid, same reasoning as x + 0).
        Expr::Binary { left, op: BinaryOp::Mul, right } => {
            if let Expr::Literal(Value::Int64(1)) = right.as_ref() {
                return Transformed::Yes((**left).clone());
            }
            if let Expr::Literal(Value::Int64(1)) = left.as_ref() {
                return Transformed::Yes((**right).clone());
            }
            // x * 0 -> 0, only if x is non-nullable: NULL * 0 is NULL.
            if let Expr::Literal(Value::Int64(0)) = right.as_ref() {
                if !left.nullable() {
                    return Transformed::Yes(Expr::Literal(Value::Int64(0)));
                }
            }
            if let Expr::Literal(Value::Int64(0)) = left.as_ref() {
                if !right.nullable() {
                    return Transformed::Yes(Expr::Literal(Value::Int64(0)));
                }
            }
            Transformed::No(expr)
        }
        // x = x -> true, only if x is non-nullable: NULL = NULL is NULL.
        Expr::Binary { left, op: BinaryOp::Eq, right } => {
            if exprs_equal(left, right) && !left.nullable() {
                return Transformed::Yes(Expr::Literal(Value::Boolean(true)));
            }
            Transformed::No(expr)
        }
        // x - x -> 0, only if x is non-nullable.
        Expr::Binary { left, op: BinaryOp::Sub, right } => {
            if exprs_equal(left, right) && !left.nullable() {
                return Transformed::Yes(Expr::Literal(Value::Int64(0)));
            }
            Transformed::No(expr)
        }
        _ => Transformed::No(expr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::data_type::DataType;

    fn col(i: usize, nullable: bool) -> Expr {
        Expr::Column {
            index: i,
            data_type: DataType::Int64,
            nullable,
        }
    }

    fn bool_lit(b: bool) -> Expr {
        Expr::Literal(Value::Boolean(b))
    }

    #[test]
    fn true_and_x_simplifies_to_x() {
        let e = Expr::Binary {
            left: Box::new(bool_lit(true)),
            op: BinaryOp::And,
            right: Box::new(col(0, false)),
        };
        assert_eq!(simplify_one(e).unwrap(), col(0, false));
    }

    #[test]
    fn false_and_x_simplifies_to_false() {
        let e = Expr::Binary {
            left: Box::new(bool_lit(false)),
            op: BinaryOp::And,
            right: Box::new(col(0, false)),
        };
        assert_eq!(simplify_one(e).unwrap(), bool_lit(false));
    }

    #[test]
    fn x_or_true_simplifies_to_true() {
        let e = Expr::Binary {
            left: Box::new(col(0, false)),
            op: BinaryOp::Or,
            right: Box::new(bool_lit(true)),
        };
        assert_eq!(simplify_one(e).unwrap(), bool_lit(true));
    }

    #[test]
    fn not_not_x_simplifies_to_x() {
        let e = Expr::Unary {
            op: UnaryOp::Not,
            expr: Box::new(Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(col(0, false)),
            }),
        };
        assert_eq!(simplify_one(e).unwrap(), col(0, false));
    }

    #[test]
    fn x_plus_zero_simplifies_to_x_even_when_nullable() {
        // Valid unconditionally: this doesn't change x's null-ness, it just
        // skips computing "+ 0".
        let e = Expr::Binary {
            left: Box::new(col(0, true)),
            op: BinaryOp::Add,
            right: Box::new(Expr::Literal(Value::Int64(0))),
        };
        assert_eq!(simplify_one(e).unwrap(), col(0, true));
    }

    #[test]
    fn x_times_zero_is_not_simplified_when_x_is_nullable() {
        // The null trap: NULL * 0 is NULL, not 0.
        let e = Expr::Binary {
            left: Box::new(col(0, true)),
            op: BinaryOp::Mul,
            right: Box::new(Expr::Literal(Value::Int64(0))),
        };
        let result = simplify_one(e.clone()).unwrap();
        assert_eq!(result, e, "must not fold x*0 to 0 when x is nullable");
    }

    #[test]
    fn x_times_zero_simplifies_when_x_is_non_nullable() {
        let e = Expr::Binary {
            left: Box::new(col(0, false)),
            op: BinaryOp::Mul,
            right: Box::new(Expr::Literal(Value::Int64(0))),
        };
        assert_eq!(simplify_one(e).unwrap(), Expr::Literal(Value::Int64(0)));
    }

    #[test]
    fn x_eq_x_is_not_simplified_when_nullable() {
        let e = Expr::Binary {
            left: Box::new(col(0, true)),
            op: BinaryOp::Eq,
            right: Box::new(col(0, true)),
        };
        let result = simplify_one(e.clone()).unwrap();
        assert_eq!(result, e, "must not fold x=x to true when x is nullable");
    }

    #[test]
    fn x_eq_x_simplifies_to_true_when_non_nullable() {
        let e = Expr::Binary {
            left: Box::new(col(0, false)),
            op: BinaryOp::Eq,
            right: Box::new(col(0, false)),
        };
        assert_eq!(simplify_one(e).unwrap(), bool_lit(true));
    }

    #[test]
    fn x_minus_x_is_not_simplified_when_nullable() {
        let e = Expr::Binary {
            left: Box::new(col(0, true)),
            op: BinaryOp::Sub,
            right: Box::new(col(0, true)),
        };
        let result = simplify_one(e.clone()).unwrap();
        assert_eq!(result, e);
    }

    #[test]
    fn x_minus_x_simplifies_to_zero_when_non_nullable() {
        let e = Expr::Binary {
            left: Box::new(col(0, false)),
            op: BinaryOp::Sub,
            right: Box::new(col(0, false)),
        };
        assert_eq!(simplify_one(e).unwrap(), Expr::Literal(Value::Int64(0)));
    }
}
