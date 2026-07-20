//! Crate-wide error taxonomy. See design-docs/basalt-phase1-lld.md §3.

use crate::sql::span::Span;

#[derive(Debug, thiserror::Error)]
pub enum BasaltError {
    // ---- I/O and ingestion ----
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("csv error at line {line}: {message}")]
    Csv { line: usize, message: String },

    // ---- SQL frontend ----
    #[error("syntax error at {span:?}: {message}")]
    Syntax { span: Span, message: String },

    // ---- binding / semantic analysis ----
    #[error("unknown column '{name}'")]
    UnknownColumn { name: String },

    #[error("type error: {message}")]
    Type { message: String },

    // ---- execution ----
    #[error("division by zero")]
    DivisionByZero,

    #[error("internal error: {0}")]
    Internal(String),

    // ---- schema/data model ----
    #[error("schema error: {message}")]
    Schema { message: String },
}

pub type Result<T> = std::result::Result<T, BasaltError>;
