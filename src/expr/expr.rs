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
