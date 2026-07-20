//! Crate-wide error taxonomy. See design-docs/basalt-phase1-lld.md §3.

#[derive(Debug, thiserror::Error)]
pub enum BasaltError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("not yet implemented: {0}")]
    Unimplemented(&'static str),
}

pub type Result<T> = std::result::Result<T, BasaltError>;
