//! Binary-op type promotion rules. See LLD §5.3.

use super::data_type::DataType;
use crate::error::{BasaltError, Result};

/// Defined here (not in `expr::expr`) so `coerce_binary` doesn't need an
/// upward dependency on `expr`; `expr::expr` re-exports these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

pub struct CoercionPlan {
    pub lhs_cast: Option<DataType>,
    pub rhs_cast: Option<DataType>,
    pub output: DataType,
}

fn no_cast(output: DataType) -> CoercionPlan {
    CoercionPlan { lhs_cast: None, rhs_cast: None, output }
}

pub fn coerce_binary(op: BinaryOp, lhs: DataType, rhs: DataType) -> Result<CoercionPlan> {
    use BinaryOp::*;
    use DataType::*;

    let type_err = |op: BinaryOp, lhs: DataType, rhs: DataType| BasaltError::Type {
        message: format!("no coercion for {op:?} between {lhs} and {rhs}"),
    };

    match op {
        Add | Sub | Mul | Div | Mod => match (lhs, rhs) {
            (Int64, Int64) => Ok(no_cast(Int64)),
            (Int64, Float64) => {
                Ok(CoercionPlan { lhs_cast: Some(Float64), rhs_cast: None, output: Float64 })
            }
            (Float64, Int64) => {
                Ok(CoercionPlan { lhs_cast: None, rhs_cast: Some(Float64), output: Float64 })
            }
            (Float64, Float64) => Ok(no_cast(Float64)),
            _ => Err(type_err(op, lhs, rhs)),
        },
        Eq | NotEq | Lt | LtEq | Gt | GtEq => match (lhs, rhs) {
            (Int64, Float64) => {
                Ok(CoercionPlan { lhs_cast: Some(Float64), rhs_cast: None, output: Boolean })
            }
            (Float64, Int64) => {
                Ok(CoercionPlan { lhs_cast: None, rhs_cast: Some(Float64), output: Boolean })
            }
            (a, b) if a == b => Ok(no_cast(Boolean)),
            _ => Err(type_err(op, lhs, rhs)),
        },
        And | Or => match (lhs, rhs) {
            (Boolean, Boolean) => Ok(no_cast(Boolean)),
            _ => Err(type_err(op, lhs, rhs)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use DataType::*;

    #[test]
    fn arithmetic_int_int() {
        let plan = coerce_binary(BinaryOp::Add, Int64, Int64).unwrap();
        assert_eq!(plan.output, Int64);
        assert!(plan.lhs_cast.is_none() && plan.rhs_cast.is_none());
    }

    #[test]
    fn arithmetic_int_float_widens_lhs() {
        let plan = coerce_binary(BinaryOp::Add, Int64, Float64).unwrap();
        assert_eq!(plan.lhs_cast, Some(Float64));
        assert_eq!(plan.rhs_cast, None);
        assert_eq!(plan.output, Float64);
    }

    #[test]
    fn arithmetic_float_float() {
        let plan = coerce_binary(BinaryOp::Mul, Float64, Float64).unwrap();
        assert_eq!(plan.output, Float64);
    }

    #[test]
    fn arithmetic_rejects_utf8() {
        assert!(coerce_binary(BinaryOp::Add, Utf8, Int64).is_err());
        assert!(coerce_binary(BinaryOp::Add, Boolean, Int64).is_err());
    }

    #[test]
    fn comparison_same_type() {
        let plan = coerce_binary(BinaryOp::Eq, Utf8, Utf8).unwrap();
        assert_eq!(plan.output, Boolean);
        let plan = coerce_binary(BinaryOp::Lt, Boolean, Boolean).unwrap();
        assert_eq!(plan.output, Boolean);
    }

    #[test]
    fn comparison_int_float_widens() {
        let plan = coerce_binary(BinaryOp::Gt, Float64, Int64).unwrap();
        assert_eq!(plan.rhs_cast, Some(Float64));
        assert_eq!(plan.output, Boolean);
    }

    #[test]
    fn comparison_rejects_utf8_vs_other() {
        assert!(coerce_binary(BinaryOp::Eq, Utf8, Int64).is_err());
    }

    #[test]
    fn logical_requires_boolean() {
        assert!(coerce_binary(BinaryOp::And, Boolean, Boolean).is_ok());
        assert!(coerce_binary(BinaryOp::Or, Boolean, Int64).is_err());
    }
}
