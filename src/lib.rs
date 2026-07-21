//! Basalt — a distributed, Arrow-native SQL query engine over Apache Iceberg.
//!
//! Crate root and public re-exports. Phase 1's module layout and dependency
//! direction (`types -> array -> batch -> expr -> plan -> exec`) are
//! specified in `design-docs/basalt-phase1-lld.md`. Phase 2 extends this with
//! `buffer`, sitting below `array` (`buffer -> array -> ...`); see
//! `design-docs/basalt-phase2-lld.md`.
//!
//! `array` currently holds two generations side by side: Phase 1's
//! `Column`/`Validity` (row-oriented, still what `exec`/`expr`/`io` run on)
//! and Phase 2's `Array`/`PrimitiveArray`/etc. (columnar, backed by
//! `buffer::Buffer`). The migration wiring the rest of the engine onto the
//! Phase 2 types is tracked separately — see the Phase 2 LLD §1's migration
//! map for what changes and why the seam is placed here.

pub mod error;

pub mod array;
pub mod batch;
pub mod buffer;
pub mod compute;
pub mod exec;
pub mod expr;
pub mod io;
pub mod logical_plan;
pub mod physical_expr;
pub mod physical_plan;
pub mod plan;
pub mod scalar;
pub mod sql;
pub mod types;

pub use error::{BasaltError, Result};
