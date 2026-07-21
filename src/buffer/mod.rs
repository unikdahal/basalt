//! The Arrow memory model: immutable shared buffers and bit-packed validity.
//! See design-docs/basalt-phase2-lld.md §3.1–§3.2.
//!
//! Sits below `array`: `buffer -> array -> batch -> compute -> physical_expr
//! -> physical_plan`. Nothing in this module depends on anything above it.

pub mod bitmap;
// LLD names this file `buffer/buffer.rs` for the `Buffer` type specifically,
// mirrored by `buffer/mutable.rs`, `buffer/bitmap.rs`; clippy reads it as a
// name clash with the parent module, but it's intentional.
#[allow(clippy::module_inception)]
pub mod buffer;
pub mod mutable;
pub mod native;

pub use bitmap::{Bitmap, BitmapBuilder};
pub use buffer::{Buffer, ALIGNMENT};
pub use mutable::MutableBuffer;
pub use native::NativeType;
