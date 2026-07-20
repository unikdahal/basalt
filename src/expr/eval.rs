//! Row-at-a-time evaluation of bound expressions.
//!
//! Evaluates expressions row-by-row over a `RecordBatch`. Correctly implements
//! three-valued SQL logic (where `NULL` represents 'unknown') and handles
//! arithmetic null propagation.

use crate::batch::RecordBatch;
use crate::error::{BasaltError, Result};
use crate::types::value::Value;
use crate::expr::expr::{Expr, BinaryOp, UnaryOp};

/// Evaluate an expression for a single row of a RecordBatch.
pub fn eval(expr: &Expr, batch: &RecordBatch, row: usize) -> Result<Value> {
    match expr {
        Expr::Column { index, .. } => {
            let col = batch.column(*index).ok_or_else(|| {
                BasaltError::Internal(format!("column index {index} out of bounds for batch"))
            })?;
            col.get(row).ok_or_else(|| {
                BasaltError::Internal(format!(
                    "row index {row} out of bounds for column of length {}",
                    col.len()
                ))
            })
        }
        Expr::Literal(val) => Ok(val.clone()),
        Expr::Binary { left, op, right } => {
            // Logical AND and OR require short-circuit three-valued logic:
            // true OR NULL -> true
            // false AND NULL -> false
            if *op == BinaryOp::And || *op == BinaryOp::Or {
                return eval_logical(*op, left, right, batch, row);
            }

            let lhs = eval(left, batch, row)?;
            let rhs = eval(right, batch, row)?;

            // Null propagation for arithmetic and comparison operators:
            // Any NULL input produces a NULL output
            if lhs.is_null() || rhs.is_null() {
                return Ok(Value::Null);
            }

            eval_binary_non_null(*op, lhs, rhs)
        }
        Expr::Unary { op, expr } => {
            let val = eval(expr, batch, row)?;
            if val.is_null() {
                return Ok(Value::Null);
            }
            match op {
                UnaryOp::Neg => match val {
                    Value::Int64(x) => Ok(Value::Int64(-x)),
                    Value::Float64(x) => Ok(Value::Float64(-x)),
                    other => Err(BasaltError::Type {
                        message: format!("cannot apply unary minus to non-numeric type {other}"),
                    }),
                }
                UnaryOp::Not => match val {
                    Value::Boolean(x) => Ok(Value::Boolean(!x)),
                    other => Err(BasaltError::Type {
                        message: format!("cannot apply logical NOT to non-boolean type {other}"),
                    }),
                }
            }
        }
        Expr::Cast { expr, to } => {
            let val = eval(expr, batch, row)?;
            val.cast_to(*to)
        }
        Expr::IsNull(expr) => {
            let val = eval(expr, batch, row)?;
            Ok(Value::Boolean(val.is_null()))
        }
        Expr::IsNotNull(expr) => {
            let val = eval(expr, batch, row)?;
            Ok(Value::Boolean(!val.is_null()))
        }
    }
}

/// Evaluate a predicate across all rows of a batch, returning matching row indices.
/// NULL and false both mean "not matched" (SQL WHERE filter semantics).
pub fn eval_predicate(expr: &Expr, batch: &RecordBatch) -> Result<Vec<usize>> {
    let mut matched = Vec::new();
    for r in 0..batch.num_rows() {
        let val = eval(expr, batch, r)?;
        match val {
            Value::Boolean(true) => {
                matched.push(r);
            }
            Value::Boolean(false) | Value::Null => {
                // Ignore: SQL WHERE logic rejects False and NULL
            }
            other => {
                return Err(BasaltError::Type {
                    message: format!("WHERE predicate must evaluate to Boolean, found {other}"),
                });
            }
        }
    }
    Ok(matched)
}

/// Evaluates three-valued logical operations.
fn eval_logical(
    op: BinaryOp,
    left: &Expr,
    right: &Expr,
    batch: &RecordBatch,
    row: usize,
) -> Result<Value> {
    let lhs = eval(left, batch, row)?;
    let rhs = eval(right, batch, row)?;

    match op {
        BinaryOp::And => match (lhs, rhs) {
            (Value::Boolean(false), _) | (_, Value::Boolean(false)) => Ok(Value::Boolean(false)),
            (Value::Boolean(true), Value::Boolean(true)) => Ok(Value::Boolean(true)),
            (Value::Boolean(_), Value::Null)
            | (Value::Null, Value::Boolean(_))
            | (Value::Null, Value::Null) => Ok(Value::Null),
            (l, r) => Err(BasaltError::Type {
                message: format!("expected boolean operands for AND, found {l} and {r}"),
            }),
        },
        BinaryOp::Or => match (lhs, rhs) {
            (Value::Boolean(true), _) | (_, Value::Boolean(true)) => Ok(Value::Boolean(true)),
            (Value::Boolean(false), Value::Boolean(false)) => Ok(Value::Boolean(false)),
            (Value::Boolean(_), Value::Null)
            | (Value::Null, Value::Boolean(_))
            | (Value::Null, Value::Null) => Ok(Value::Null),
            (l, r) => Err(BasaltError::Type {
                message: format!("expected boolean operands for OR, found {l} and {r}"),
            }),
        },
        _ => unreachable!(),
    }
}

/// Evaluates binary operations on non-null values.
fn eval_binary_non_null(op: BinaryOp, lhs: Value, rhs: Value) -> Result<Value> {
    match op {
        BinaryOp::Add => match (lhs, rhs) {
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a + b)),
            (Value::Float64(a), Value::Float64(b)) => Ok(Value::Float64(a + b)),
            _ => unreachable!(),
        },
        BinaryOp::Sub => match (lhs, rhs) {
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a - b)),
            (Value::Float64(a), Value::Float64(b)) => Ok(Value::Float64(a - b)),
            _ => unreachable!(),
        },
        BinaryOp::Mul => match (lhs, rhs) {
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a * b)),
            (Value::Float64(a), Value::Float64(b)) => Ok(Value::Float64(a * b)),
            _ => unreachable!(),
        },
        BinaryOp::Div => match (lhs, rhs) {
            (Value::Int64(a), Value::Int64(b)) => {
                if b == 0 {
                    Err(BasaltError::DivisionByZero)
                } else {
                    Ok(Value::Int64(a / b))
                }
            }
            (Value::Float64(a), Value::Float64(b)) => {
                if b == 0.0 {
                    Err(BasaltError::DivisionByZero)
                } else {
                    Ok(Value::Float64(a / b))
                }
            }
            _ => unreachable!(),
        },
        BinaryOp::Mod => match (lhs, rhs) {
            (Value::Int64(a), Value::Int64(b)) => {
                if b == 0 {
                    Err(BasaltError::DivisionByZero)
                } else {
                    Ok(Value::Int64(a % b))
                }
            }
            (Value::Float64(a), Value::Float64(b)) => {
                if b == 0.0 {
                    Err(BasaltError::DivisionByZero)
                } else {
                    Ok(Value::Float64(a % b))
                }
            }
            _ => unreachable!(),
        },
        BinaryOp::Eq => Ok(Value::Boolean(lhs == rhs)),
        BinaryOp::NotEq => Ok(Value::Boolean(lhs != rhs)),
        BinaryOp::Lt => match (lhs, rhs) {
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Boolean(a < b)),
            (Value::Float64(a), Value::Float64(b)) => Ok(Value::Boolean(a < b)),
            (Value::Utf8(a), Value::Utf8(b)) => Ok(Value::Boolean(a < b)),
            (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(a < b)),
            _ => unreachable!(),
        },
        BinaryOp::LtEq => match (lhs, rhs) {
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Boolean(a <= b)),
            (Value::Float64(a), Value::Float64(b)) => Ok(Value::Boolean(a <= b)),
            (Value::Utf8(a), Value::Utf8(b)) => Ok(Value::Boolean(a <= b)),
            (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(a <= b)),
            _ => unreachable!(),
        },
        BinaryOp::Gt => match (lhs, rhs) {
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Boolean(a > b)),
            (Value::Float64(a), Value::Float64(b)) => Ok(Value::Boolean(a > b)),
            (Value::Utf8(a), Value::Utf8(b)) => Ok(Value::Boolean(a > b)),
            (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(a > b)),
            _ => unreachable!(),
        },
        BinaryOp::GtEq => match (lhs, rhs) {
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Boolean(a >= b)),
            (Value::Float64(a), Value::Float64(b)) => Ok(Value::Boolean(a >= b)),
            (Value::Utf8(a), Value::Utf8(b)) => Ok(Value::Boolean(a >= b)),
            (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(a >= b)),
            _ => unreachable!(),
        },
        BinaryOp::And | BinaryOp::Or => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::column::{Column, ColumnData};
    use crate::types::schema::{Field, Schema};
    use crate::types::data_type::DataType;

    fn test_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Boolean, true),
        ]).unwrap();
        
        let cols = vec![
            Column::from_parts(ColumnData::Int64(vec![10, 20, 0]), Some(crate::array::validity::Validity::from_flags(vec![true, false, true]))),
            Column::from_parts(ColumnData::Boolean(vec![true, false, false]), Some(crate::array::validity::Validity::from_flags(vec![true, false, true]))),
        ];
        
        RecordBatch::try_new(schema, cols).unwrap()
    }

    #[test]
    fn test_eval_col() {
        let batch = test_batch();
        let col = Expr::Column { index: 0, data_type: DataType::Int64, nullable: true };
        assert_eq!(eval(&col, &batch, 0).unwrap(), Value::Int64(10));
        assert_eq!(eval(&col, &batch, 1).unwrap(), Value::Null);
    }

    #[test]
    fn test_three_valued_logic_and() {
        let batch = test_batch();
        
        // true AND NULL = NULL
        let and_null = Expr::Binary {
            left: Box::new(Expr::Literal(Value::Boolean(true))),
            op: BinaryOp::And,
            right: Box::new(Expr::Literal(Value::Null)),
        };
        assert_eq!(eval(&and_null, &batch, 0).unwrap(), Value::Null);

        // false AND NULL = false
        let and_false = Expr::Binary {
            left: Box::new(Expr::Literal(Value::Boolean(false))),
            op: BinaryOp::And,
            right: Box::new(Expr::Literal(Value::Null)),
        };
        assert_eq!(eval(&and_false, &batch, 0).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_three_valued_logic_or() {
        let batch = test_batch();
        
        // false OR NULL = NULL
        let or_null = Expr::Binary {
            left: Box::new(Expr::Literal(Value::Boolean(false))),
            op: BinaryOp::Or,
            right: Box::new(Expr::Literal(Value::Null)),
        };
        assert_eq!(eval(&or_null, &batch, 0).unwrap(), Value::Null);

        // true OR NULL = true
        let or_true = Expr::Binary {
            left: Box::new(Expr::Literal(Value::Boolean(true))),
            op: BinaryOp::Or,
            right: Box::new(Expr::Literal(Value::Null)),
        };
        assert_eq!(eval(&or_true, &batch, 0).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_eval_div_by_zero() {
        let batch = test_batch();
        let div = Expr::Binary {
            left: Box::new(Expr::Literal(Value::Int64(10))),
            op: BinaryOp::Div,
            right: Box::new(Expr::Literal(Value::Int64(0))),
        };
        let err = eval(&div, &batch, 0).unwrap_err();
        assert!(matches!(err, BasaltError::DivisionByZero));

        let div_float = Expr::Binary {
            left: Box::new(Expr::Literal(Value::Float64(10.0))),
            op: BinaryOp::Div,
            right: Box::new(Expr::Literal(Value::Float64(0.0))),
        };
        let err_float = eval(&div_float, &batch, 0).unwrap_err();
        assert!(matches!(err_float, BasaltError::DivisionByZero));
    }
}
