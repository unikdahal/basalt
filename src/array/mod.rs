//! Columnar storage.
//!
//! Two generations live side by side here:
//! - **Phase 1** (`validity`, `column`, `builder`): row-oriented `Value`/
//!   `Column`, still what `batch`/`exec`/`expr`/`io` run on. See
//!   design-docs/basalt-phase1-lld.md §2.3–§2.5.
//! - **Phase 2** (`array`, `types`, `primitive`, `boolean`, `string`):
//!   Arrow-native columnar arrays backed by `buffer::Buffer`/`Bitmap`, not
//!   yet wired into execution. See design-docs/basalt-phase2-lld.md §3.3–3.4.

pub mod builder;
pub mod column;
pub mod validity;

// LLD names this file `array/array.rs` for the `Array` trait specifically,
// mirrored by `array/primitive.rs`, `array/boolean.rs`, `array/string.rs`;
// clippy reads it as a name clash with the parent module, but it's intentional.
#[allow(clippy::module_inception)]
pub mod array;
pub mod boolean;
pub mod primitive;
pub mod string;
pub mod types;
