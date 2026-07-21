//! Maps an Arrow logical type to its Rust representation. See
//! design-docs/basalt-phase2-lld.md §3.3.

use crate::buffer::NativeType;
use crate::types::data_type::DataType;

/// A primitive (fixed-width) Arrow type: a compile-time pairing of a Rust
/// native type with the runtime [`DataType`] tag it represents.
pub trait ArrowPrimitiveType: Send + Sync + 'static {
    type Native: NativeType + Default;
    const DATA_TYPE: DataType;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Int64Type;

impl ArrowPrimitiveType for Int64Type {
    type Native = i64;
    const DATA_TYPE: DataType = DataType::Int64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Float64Type;

impl ArrowPrimitiveType for Float64Type {
    type Native = f64;
    const DATA_TYPE: DataType = DataType::Float64;
}
