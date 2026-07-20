//! Basalt — a distributed, Arrow-native SQL query engine over Apache Iceberg.
//!
//! Crate root and public re-exports. Module layout and dependency direction
//! (strictly downward: types -> array -> batch -> expr -> plan -> exec) are
//! specified in `design-docs/basalt-phase1-lld.md`.

pub mod error;

pub mod array;
pub mod batch;
pub mod exec;
pub mod expr;
pub mod io;
pub mod plan;
pub mod sql;
pub mod types;

pub use error::{BasaltError, Result};
