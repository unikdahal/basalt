//! SQL unbound Abstract Syntax Tree (AST).
//!
//! This module defines the raw structural trees parsed from SQL statement strings.
//! They are "unbound" because table and column references are still named strings
//! rather than checked catalog offsets, and expressions are not yet type-checked.

use crate::types::data_type::DataType;
use crate::types::coercion::{BinaryOp, UnaryOp};

/// Statements supported by the parser.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// A query SELECT statement.
    Select(SelectStatement),
}

/// A parsed SELECT query representation containing projection fields, source table,
/// and optional filter (WHERE), ordering (ORDER BY), and limit clauses.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectStatement {
    /// Projection items: select fields or wildcard.
    pub projections: Vec<SelectItem>,
    /// Source table specification.
    pub from: TableRef,
    /// Selection expression (`WHERE` filter clause).
    pub selection: Option<Expr>,
    /// Sort expressions (`ORDER BY` clauses).
    pub order_by: Vec<OrderByExpr>,
    /// Row limit constraint.
    pub limit: Option<u64>,
}

/// Projection column items in a SELECT statement.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem {
    /// Wildcard select (`*`).
    Wildcard,
    /// Projection expression with optional column alias (e.g. `c AS alias`).
    Expr {
        expr: Expr,
        alias: Option<String>,
    },
}

/// A reference to a source table.
#[derive(Debug, Clone, PartialEq)]
pub struct TableRef {
    /// Table name.
    pub name: String,
    /// Optional table alias name.
    pub alias: Option<String>,
}

/// A sort ordering key.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderByExpr {
    /// Sort key expression.
    pub expr: Expr,
    /// Boolean flag where true indicates ascending order, false indicates descending order.
    pub asc: bool,
}

/// Unbound SQL expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Column reference by name identifier.
    Identifier(String),
    /// SQL literals.
    Literal(Literal),
    /// Binary operations.
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    /// Unary operations.
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    /// SQL Cast expression.
    Cast {
        expr: Box<Expr>,
        to: DataType,
    },
    /// Postfix `IS NULL` or `IS NOT NULL` check.
    IsNull {
        expr: Box<Expr>,
        negated: bool,
    },
    /// Parenthesized subexpression mapping to keep parsing precedence order.
    Nested(Box<Expr>),
}

/// Literal values in unbound syntax nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
}
