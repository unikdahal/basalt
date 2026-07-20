//! Columnar storage: null tracking, typed columns, and incremental builders.
//! See design-docs/basalt-phase1-lld.md §2.3–§2.5.

pub mod validity;
pub mod column;
pub mod builder;
