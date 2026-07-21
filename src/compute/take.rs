//! `take` — gather rows at given positions, in the given order. See
//! design-docs/basalt-phase2-lld.md §4.5.
//!
//! This is one of the two structural kernels (with `filter`): joins, sort,
//! and filter all reduce to "produce a new array from these row positions."
//! `take` is inherently random-access, so unlike `filter` there's no
//! selectivity-based fast path to add here — the gather itself is the cost.

use std::sync::Arc;

use super::index::UInt32Array;
use crate::array::array::{as_boolean, as_primitive, as_string, Array, ArrayRef};
use crate::array::boolean::BooleanBuilder;
use crate::array::primitive::PrimitiveBuilder;
use crate::array::string::StringBuilder;
use crate::array::types::{ArrowPrimitiveType, Float64Type, Int64Type};
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;

/// Gather rows at the given positions, in the given order.
///
/// A null entry in `indices` produces a null output row (the case an outer
/// join hits: unmatched rows are gathered via a null index). An out-of-range
/// non-null index is an error, not a panic.
///
/// # Errors
/// Errors if any non-null index in `indices` is `>= array.len()`.
pub fn take(array: &dyn Array, indices: &UInt32Array) -> Result<ArrayRef> {
    match array.data_type() {
        DataType::Int64 => take_primitive::<Int64Type>(as_primitive(array)?, indices),
        DataType::Float64 => take_primitive::<Float64Type>(as_primitive(array)?, indices),
        DataType::Boolean => take_boolean(as_boolean(array)?, indices),
        DataType::Utf8 => take_string(as_string(array)?, indices),
    }
}

/// A hoisted bitmap: `(bytes, bit_offset)`, indexable via `buffer::bit_at`.
type HoistedBitmap<'a> = (&'a [u8], usize);

/// Validates bounds while also handing back the hoisted `(u32 values,
/// validity)` the caller's gather loop needs — one pass over `indices`
/// instead of two, and `indices.values()` derived once instead of via
/// `indices.value(i)` (which re-derives its slice from the underlying
/// buffer every call) on every element of both the check and the gather.
fn checked_indices(
    indices: &UInt32Array,
    len: usize,
) -> Result<(&[u32], Option<HoistedBitmap<'_>>)> {
    let values = indices.values();
    let validity = indices.validity().map(|v| (v.as_bytes(), v.bit_offset()));
    for (i, &idx) in values.iter().enumerate() {
        let is_null =
            validity.is_some_and(|(bytes, offset)| !crate::buffer::bit_at(bytes, offset, i));
        if is_null {
            continue;
        }
        if idx as usize >= len {
            return Err(BasaltError::Internal(format!(
                "take index {idx} out of bounds for array of length {len}"
            )));
        }
    }
    Ok((values, validity))
}

fn take_primitive<T: ArrowPrimitiveType>(
    array: &crate::array::primitive::PrimitiveArray<T>,
    indices: &UInt32Array,
) -> Result<ArrayRef> {
    let (idx_values, idx_validity) = checked_indices(indices, array.len())?;
    let source = array.values();
    let src_validity = array.validity().map(|v| (v.as_bytes(), v.bit_offset()));
    let mut builder = PrimitiveBuilder::<T>::with_capacity(indices.len());
    for (i, &idx) in idx_values.iter().enumerate() {
        if idx_validity.is_some_and(|(bytes, offset)| !crate::buffer::bit_at(bytes, offset, i)) {
            builder.append_null();
            continue;
        }
        let idx = idx as usize;
        if src_validity.is_some_and(|(bytes, offset)| !crate::buffer::bit_at(bytes, offset, idx)) {
            builder.append_null();
        } else {
            builder.append_value(source[idx]);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn take_boolean(
    array: &crate::array::boolean::BooleanArray,
    indices: &UInt32Array,
) -> Result<ArrayRef> {
    let (idx_values, idx_validity) = checked_indices(indices, array.len())?;
    let (src_bytes, src_offset) = (array.values().as_bytes(), array.values().bit_offset());
    let src_validity = array.validity().map(|v| (v.as_bytes(), v.bit_offset()));
    let mut builder = BooleanBuilder::with_capacity(indices.len());
    for (i, &idx) in idx_values.iter().enumerate() {
        if idx_validity.is_some_and(|(bytes, offset)| !crate::buffer::bit_at(bytes, offset, i)) {
            builder.append_null();
            continue;
        }
        let idx = idx as usize;
        if src_validity.is_some_and(|(bytes, offset)| !crate::buffer::bit_at(bytes, offset, idx)) {
            builder.append_null();
        } else {
            builder.append_value(crate::buffer::bit_at(src_bytes, src_offset, idx));
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn take_string(
    array: &crate::array::string::StringArray,
    indices: &UInt32Array,
) -> Result<ArrayRef> {
    let (idx_values, idx_validity) = checked_indices(indices, array.len())?;
    let mut builder = StringBuilder::with_capacity(indices.len(), 0);
    for (i, &idx) in idx_values.iter().enumerate() {
        if idx_validity.is_some_and(|(bytes, offset)| !crate::buffer::bit_at(bytes, offset, i)) {
            builder.append_null();
            continue;
        }
        let idx = idx as usize;
        if array.is_null(idx) {
            builder.append_null();
        } else {
            builder.append_value(array.value(idx))?;
        }
    }
    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_primitive as downcast_primitive;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::compute::index::UInt32Builder;

    fn int_array() -> ArrayRef {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(4);
        b.append_value(10);
        b.append_null();
        b.append_value(30);
        b.append_value(40);
        Arc::new(b.finish())
    }

    #[test]
    fn take_reorders_and_duplicates() {
        let arr = int_array();
        let mut idx = UInt32Builder::with_capacity(3);
        idx.append_value(3);
        idx.append_value(0);
        idx.append_value(0);
        let taken = take(arr.as_ref(), &idx.finish()).unwrap();
        let taken = downcast_primitive::<Int64Type>(taken.as_ref()).unwrap();
        assert_eq!(taken.value(0), 40);
        assert_eq!(taken.value(1), 10);
        assert_eq!(taken.value(2), 10);
    }

    #[test]
    fn take_propagates_source_nulls() {
        let arr = int_array();
        let mut idx = UInt32Builder::with_capacity(1);
        idx.append_value(1); // the null slot
        let taken = take(arr.as_ref(), &idx.finish()).unwrap();
        assert!(taken.is_null(0));
    }

    #[test]
    fn null_index_produces_null_output_row() {
        let arr = int_array();
        let mut idx = UInt32Builder::with_capacity(2);
        idx.append_value(0);
        idx.append_null();
        let taken = take(arr.as_ref(), &idx.finish()).unwrap();
        assert!(!taken.is_null(0));
        assert!(taken.is_null(1));
    }

    #[test]
    fn out_of_range_index_errors_not_panics() {
        let arr = int_array();
        let mut idx = UInt32Builder::with_capacity(1);
        idx.append_value(99);
        assert!(take(arr.as_ref(), &idx.finish()).is_err());
    }

    #[test]
    fn take_with_empty_indices_yields_empty_array() {
        let arr = int_array();
        let idx = UInt32Builder::with_capacity(0).finish();
        let taken = take(arr.as_ref(), &idx).unwrap();
        assert_eq!(taken.len(), 0);
    }

    #[test]
    fn take_on_string_array() {
        let mut b = StringBuilder::with_capacity(2, 8);
        b.append_value("a").unwrap();
        b.append_value("b").unwrap();
        let arr: ArrayRef = Arc::new(b.finish());
        let mut idx = UInt32Builder::with_capacity(2);
        idx.append_value(1);
        idx.append_value(0);
        let taken = take(arr.as_ref(), &idx.finish()).unwrap();
        let taken = as_string(taken.as_ref()).unwrap();
        assert_eq!(taken.value(0), "b");
        assert_eq!(taken.value(1), "a");
    }
}
