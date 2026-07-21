//! Bound query expression tree nodes.
//!
//! Defines the `Expr` enum, representing type-checked expressions where table
//! and column references have been resolved to indices, and casts have been made explicit.

pub use crate::types::coercion::{BinaryOp, UnaryOp};
use crate::types::data_type::DataType;
use crate::types::value::Value;

/// A bound expression: columns are resolved to schemas by index, and types are verified.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Column reference by ordinal (index in schema) and nullability.
    Column {
        index: usize,
        data_type: DataType,
        nullable: bool,
    },
    /// A constant literal value.
    Literal(Value),
    /// A binary operator expression.
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    /// A unary operator expression.
    Unary { op: UnaryOp, expr: Box<Expr> },
    /// Explicit type cast (e.g. integer to float promotion).
    Cast { expr: Box<Expr>, to: DataType },
    /// Postfix IS NULL check.
    IsNull(Box<Expr>),
    /// Postfix IS NOT NULL check.
    IsNotNull(Box<Expr>),
}

impl Expr {
    /// Direct children, in evaluation order. Empty for leaves (`Column`,
    /// `Literal`). The tree-rewriting primitive Phase 3's optimizer rules
    /// build on, mirroring `LogicalPlan::inputs`/`with_new_inputs`.
    pub fn children(&self) -> Vec<&Expr> {
        match self {
            Expr::Column { .. } | Expr::Literal(_) => vec![],
            Expr::Binary { left, right, .. } => vec![left, right],
            Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => vec![expr],
            Expr::IsNull(expr) | Expr::IsNotNull(expr) => vec![expr],
        }
    }

    /// Rebuilds this node with new children, in the same order
    /// `children()` returned them.
    ///
    /// # Errors
    /// Errors if `children`'s length doesn't match this variant's arity.
    pub fn with_new_children(&self, children: Vec<Expr>) -> crate::error::Result<Expr> {
        use crate::error::BasaltError;
        fn one(children: Vec<Expr>) -> crate::error::Result<Expr> {
            match <[Expr; 1]>::try_from(children) {
                Ok([e]) => Ok(e),
                Err(other) => Err(BasaltError::Internal(format!(
                    "expected exactly 1 child, got {}",
                    other.len()
                ))),
            }
        }

        match self {
            Expr::Column { .. } | Expr::Literal(_) => {
                if !children.is_empty() {
                    return Err(BasaltError::Internal(format!(
                        "{self:?} takes 0 children, got {}",
                        children.len()
                    )));
                }
                Ok(self.clone())
            }
            Expr::Binary { op, .. } => match <[Expr; 2]>::try_from(children) {
                Ok([left, right]) => Ok(Expr::Binary {
                    left: Box::new(left),
                    op: *op,
                    right: Box::new(right),
                }),
                Err(other) => Err(BasaltError::Internal(format!(
                    "Binary takes exactly 2 children, got {}",
                    other.len()
                ))),
            },
            Expr::Unary { op, .. } => Ok(Expr::Unary {
                op: *op,
                expr: Box::new(one(children)?),
            }),
            Expr::Cast { to, .. } => Ok(Expr::Cast {
                expr: Box::new(one(children)?),
                to: *to,
            }),
            Expr::IsNull(_) => Ok(Expr::IsNull(Box::new(one(children)?))),
            Expr::IsNotNull(_) => Ok(Expr::IsNotNull(Box::new(one(children)?))),
        }
    }
}

#[cfg(test)]
mod tree_tests {
    use super::*;
    use crate::types::data_type::DataType;
    use crate::types::value::Value;

    fn col(i: usize) -> Expr {
        Expr::Column {
            index: i,
            data_type: DataType::Int64,
            nullable: false,
        }
    }

    #[test]
    fn leaves_have_no_children() {
        assert!(col(0).children().is_empty());
        assert!(Expr::Literal(Value::Int64(1)).children().is_empty());
    }

    #[test]
    fn binary_children_are_left_then_right() {
        let e = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Add,
            right: Box::new(col(1)),
        };
        let children = e.children();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0], &col(0));
        assert_eq!(children[1], &col(1));
    }

    #[test]
    fn with_new_children_rebuilds_binary() {
        let e = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Add,
            right: Box::new(col(1)),
        };
        let rebuilt = e.with_new_children(vec![col(5), col(6)]).unwrap();
        match rebuilt {
            Expr::Binary { left, op, right } => {
                assert_eq!(*left, col(5));
                assert_eq!(op, BinaryOp::Add);
                assert_eq!(*right, col(6));
            }
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn with_new_children_rejects_wrong_arity() {
        let e = Expr::IsNull(Box::new(col(0)));
        assert!(e.with_new_children(vec![]).is_err());
        assert!(e.with_new_children(vec![col(0), col(1)]).is_err());
    }

    #[test]
    fn leaf_with_new_children_requires_zero() {
        let e = col(0);
        assert!(e.with_new_children(vec![]).is_ok());
        assert!(e.with_new_children(vec![col(1)]).is_err());
    }
}
