//! Static type and nullability propagation for bound expressions.
//!
//! Evaluates the result type of expressions statically (at plan time) using
//! relational type rules and coercion logic. It also propagates nullability.

use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;
use crate::types::coercion::{coerce_binary, UnaryOp};
use crate::expr::expr::Expr;

impl Expr {
    /// Compute the static output type of the expression.
    pub fn data_type(&self) -> Result<DataType> {
        match self {
            Expr::Column { data_type, .. } => Ok(*data_type),
            Expr::Literal(val) => {
                val.data_type().ok_or_else(|| BasaltError::Type {
                    message: "untyped NULL literal has no static type".to_string(),
                })
            }
            Expr::Binary { left, op, right } => {
                let lhs = left.data_type()?;
                let rhs = right.data_type()?;
                let plan = coerce_binary(*op, lhs, rhs)?;
                Ok(plan.output)
            }
            Expr::Unary { op, expr } => {
                let t = expr.data_type()?;
                match op {
                    UnaryOp::Neg => {
                        if t.is_numeric() {
                            Ok(t)
                        } else {
                            Err(BasaltError::Type {
                                message: format!("cannot apply unary negation (-) to non-numeric type {t}"),
                            })
                        }
                    }
                    UnaryOp::Not => {
                        if t == DataType::Boolean {
                            Ok(DataType::Boolean)
                        } else {
                            Err(BasaltError::Type {
                                message: format!("cannot apply logical NOT to non-boolean type {t}"),
                            })
                        }
                    }
                }
            }
            Expr::Cast { to, .. } => Ok(*to),
            Expr::IsNull(_) | Expr::IsNotNull(_) => Ok(DataType::Boolean),
        }
    }

    /// Determine if the expression can produce a NULL value at runtime.
    pub fn nullable(&self) -> bool {
        match self {
            Expr::Column { nullable, .. } => *nullable,
            Expr::Literal(val) => val.is_null(),
            Expr::Binary { left, right, .. } => {
                // Arithmetic and comparison operators propagate NULLs.
                // Logical AND/OR can produce NULL if either operand is NULL.
                left.nullable() || right.nullable()
            }
            Expr::Unary { expr, .. } => expr.nullable(),
            Expr::Cast { expr, .. } => expr.nullable(),
            Expr::IsNull(_) | Expr::IsNotNull(_) => false, // IS NULL predicates always return boolean
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::expr::BinaryOp;

    #[test]
    fn test_column_typing() {
        let col = Expr::Column { index: 0, data_type: DataType::Int64, nullable: false };
        assert_eq!(col.data_type().unwrap(), DataType::Int64);
        assert!(!col.nullable());
    }

    #[test]
    fn test_binary_arithmetic_typing() {
        let col_int = Expr::Column { index: 0, data_type: DataType::Int64, nullable: false };
        let col_float = Expr::Column { index: 1, data_type: DataType::Float64, nullable: true };
        
        // Int64 + Float64 -> Float64
        let add = Expr::Binary {
            left: Box::new(col_int.clone()),
            op: BinaryOp::Add,
            right: Box::new(col_float.clone()),
        };
        assert_eq!(add.data_type().unwrap(), DataType::Float64);
        assert!(add.nullable());
    }

    #[test]
    fn test_unary_typing() {
        let col_int = Expr::Column { index: 0, data_type: DataType::Int64, nullable: false };
        let neg = Expr::Unary { op: UnaryOp::Neg, expr: Box::new(col_int.clone()) };
        assert_eq!(neg.data_type().unwrap(), DataType::Int64);

        let col_utf8 = Expr::Column { index: 1, data_type: DataType::Utf8, nullable: false };
        let bad_neg = Expr::Unary { op: UnaryOp::Neg, expr: Box::new(col_utf8) };
        assert!(bad_neg.data_type().is_err());
    }
}
