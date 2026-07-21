//! The `Array` trait — the central columnar abstraction of Phase 2. See
//! design-docs/basalt-phase2-lld.md §3.3.

use std::any::Any;
use std::sync::Arc;

use crate::buffer::Bitmap;
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;

/// A columnar array of values of a single type.
pub trait Array: std::fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn data_type(&self) -> DataType;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn null_count(&self) -> usize;
    fn is_null(&self, i: usize) -> bool;
    fn is_valid(&self, i: usize) -> bool {
        !self.is_null(i)
    }
    fn validity(&self) -> Option<&Bitmap>;

    /// O(1) zero-copy slice.
    fn slice(&self, offset: usize, len: usize) -> ArrayRef;
}

pub type ArrayRef = Arc<dyn Array>;

/// Downcast a `dyn Array` to a concrete `PrimitiveArray<T>`.
///
/// # Errors
/// Returns an error if `array`'s concrete type doesn't match `T`.
pub fn as_primitive<T: super::types::ArrowPrimitiveType>(
    array: &dyn Array,
) -> Result<&super::primitive::PrimitiveArray<T>> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| BasaltError::Internal("downcast to PrimitiveArray failed".to_string()))
}

/// Downcast a `dyn Array` to a concrete `BooleanArray`.
///
/// # Errors
/// Returns an error if `array`'s concrete type isn't `BooleanArray`.
pub fn as_boolean(array: &dyn Array) -> Result<&super::boolean::BooleanArray> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| BasaltError::Internal("downcast to BooleanArray failed".to_string()))
}

/// Downcast a `dyn Array` to a concrete `StringArray`.
///
/// # Errors
/// Returns an error if `array`'s concrete type isn't `StringArray`.
pub fn as_string(array: &dyn Array) -> Result<&super::string::StringArray> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| BasaltError::Internal("downcast to StringArray failed".to_string()))
}
