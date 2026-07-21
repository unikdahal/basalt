//! `PrimitiveArray<T>` — a fixed-width, generic columnar array. See
//! design-docs/basalt-phase2-lld.md §3.3–3.4.

use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use super::array::{Array, ArrayRef};
use super::types::{ArrowPrimitiveType, Float64Type, Int64Type};
use crate::buffer::{Bitmap, BitmapBuilder, Buffer, MutableBuffer};
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;

/// A fixed-width array, generic over the Arrow type; monomorphized per type.
///
/// # Invariants
/// - `values.len()` is an exact multiple of `size_of::<T::Native>()`, and
///   `values` is aligned for `T::Native` (checked once, in `try_new`).
/// - If `validity` is `Some`, its length equals `len`.
/// - Null slots contain arbitrary (but initialized) values — never read
///   without checking validity first.
pub struct PrimitiveArray<T: ArrowPrimitiveType> {
    values: Buffer,
    validity: Option<Bitmap>,
    offset: usize,
    len: usize,
    _phantom: PhantomData<T>,
}

pub type Int64Array = PrimitiveArray<Int64Type>;
pub type Float64Array = PrimitiveArray<Float64Type>;

// Manual `Clone`/`Debug` impls rather than `#[derive]`: deriving would add a
// spurious `T: Clone + Debug` bound (derive macros bound every generic
// parameter, whether or not it's actually stored), forcing every future
// `ArrowPrimitiveType` marker to implement them for no reason. `T` here is a
// zero-sized tag living only in `PhantomData`.
impl<T: ArrowPrimitiveType> Clone for PrimitiveArray<T> {
    fn clone(&self) -> Self {
        PrimitiveArray {
            values: self.values.clone(),
            validity: self.validity.clone(),
            offset: self.offset,
            len: self.len,
            _phantom: PhantomData,
        }
    }
}

impl<T: ArrowPrimitiveType> std::fmt::Debug for PrimitiveArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimitiveArray")
            .field("data_type", &T::DATA_TYPE)
            .field("len", &self.len)
            .field(
                "null_count",
                &self.validity.as_ref().map_or(0, Bitmap::null_count),
            )
            .finish()
    }
}

impl<T: ArrowPrimitiveType> PrimitiveArray<T> {
    /// Validates the buffer/validity invariants once, up front.
    ///
    /// # Errors
    /// Errors if `values`'s length isn't a multiple of `size_of::<T::Native>()`,
    /// isn't aligned for `T::Native`, or `validity`'s length doesn't match.
    pub fn try_new(values: Buffer, validity: Option<Bitmap>) -> Result<Self> {
        let elem_size = std::mem::size_of::<T::Native>();
        if !values.len().is_multiple_of(elem_size) {
            return Err(BasaltError::Internal(format!(
                "buffer length {} is not a multiple of element size {elem_size}",
                values.len()
            )));
        }
        let len = values.len() / elem_size;
        if let Some(v) = &validity {
            if v.len() != len {
                return Err(BasaltError::Internal(format!(
                    "validity length {} does not match array length {len}",
                    v.len()
                )));
            }
        }
        // Validates alignment now, once, so `values()` never has to.
        values.typed_data::<T::Native>()?;
        Ok(PrimitiveArray {
            values,
            validity,
            offset: 0,
            len,
            _phantom: PhantomData,
        })
    }

    /// Constructs directly from parts a builder just produced itself — no
    /// untrusted input, so the invariants above are guaranteed by
    /// construction rather than re-checked. Invariants are enforced once, at
    /// the real boundary (`try_new`, for data arriving from outside).
    pub(crate) fn from_parts_unchecked(
        values: Buffer,
        validity: Option<Bitmap>,
        len: usize,
    ) -> Self {
        debug_assert_eq!(values.len(), len * std::mem::size_of::<T::Native>());
        if let Some(v) = &validity {
            debug_assert_eq!(v.len(), len);
        }
        PrimitiveArray {
            values,
            validity,
            offset: 0,
            len,
            _phantom: PhantomData,
        }
    }

    /// Typed access to the raw values, including null slots.
    pub fn values(&self) -> &[T::Native] {
        // SAFETY: validated once in `try_new` (or guaranteed by the
        // `from_parts_unchecked` caller contract); `Buffer` is immutable, so
        // the invariant holds for the array's whole lifetime.
        let full = unsafe { self.values.typed_data_unchecked::<T::Native>() };
        &full[self.offset..self.offset + self.len]
    }

    /// The value at `i`. Does NOT check validity — caller must.
    pub fn value(&self, i: usize) -> T::Native {
        self.values()[i]
    }
}

impl<T: ArrowPrimitiveType> Array for PrimitiveArray<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self) -> DataType {
        T::DATA_TYPE
    }

    fn len(&self) -> usize {
        self.len
    }

    fn null_count(&self) -> usize {
        self.validity.as_ref().map_or(0, Bitmap::null_count)
    }

    fn is_null(&self, i: usize) -> bool {
        self.validity.as_ref().is_some_and(|v| !v.get(i))
    }

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    fn slice(&self, offset: usize, len: usize) -> ArrayRef {
        debug_assert!(
            offset + len <= self.len,
            "PrimitiveArray::slice out of bounds"
        );
        let elem_size = std::mem::size_of::<T::Native>();
        Arc::new(PrimitiveArray::<T> {
            values: self
                .values
                .slice((self.offset + offset) * elem_size, len * elem_size),
            validity: self.validity.as_ref().map(|v| v.slice(offset, len)),
            offset: 0,
            len,
            _phantom: PhantomData,
        })
    }
}

/// Incremental, typed construction of a `PrimitiveArray<T>`.
pub struct PrimitiveBuilder<T: ArrowPrimitiveType> {
    values: MutableBuffer,
    validity: BitmapBuilder,
    null_count: usize,
    _phantom: PhantomData<T>,
}

impl<T: ArrowPrimitiveType> PrimitiveBuilder<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        PrimitiveBuilder {
            values: MutableBuffer::with_capacity(capacity * std::mem::size_of::<T::Native>()),
            validity: BitmapBuilder::with_capacity(capacity),
            null_count: 0,
            _phantom: PhantomData,
        }
    }

    pub fn append_value(&mut self, v: T::Native) {
        self.values.push(v);
        self.validity.push(true);
    }

    pub fn append_null(&mut self) {
        self.values.push(T::Native::default());
        self.validity.push(false);
        self.null_count += 1;
    }

    /// Bulk-append a slice with no nulls. Dramatically faster than repeated
    /// `append_value` calls for a chunk known to have no nulls (e.g. a CSV
    /// or Parquet column read straight through).
    pub fn append_slice(&mut self, values: &[T::Native]) {
        self.values.extend_from_slice(values);
        for _ in 0..values.len() {
            self.validity.push(true);
        }
    }

    pub fn len(&self) -> usize {
        self.validity.len()
    }

    pub fn is_empty(&self) -> bool {
        self.validity.is_empty()
    }

    /// Consumes the builder. Drops validity entirely if no nulls were appended.
    pub fn finish(self) -> PrimitiveArray<T> {
        let len = self.validity.len();
        let values = self.values.freeze();
        let validity = if self.null_count == 0 {
            None
        } else {
            Some(self.validity.finish())
        };
        PrimitiveArray::from_parts_unchecked(values, validity, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::{as_primitive, Array};

    fn int_array_with_nulls() -> Int64Array {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(4);
        b.append_value(10);
        b.append_null();
        b.append_value(30);
        b.append_slice(&[40, 50]);
        b.finish()
    }

    #[test]
    fn builder_round_trips_values_and_nulls() {
        let arr = int_array_with_nulls();
        assert_eq!(arr.len(), 5);
        assert_eq!(arr.null_count(), 1);
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
        assert_eq!(arr.value(0), 10);
        assert_eq!(arr.value(3), 40);
        assert_eq!(arr.value(4), 50);
    }

    #[test]
    fn no_nulls_seen_means_validity_is_none() {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(2);
        b.append_value(1);
        b.append_value(2);
        let arr = b.finish();
        assert!(arr.validity().is_none());
        assert_eq!(arr.null_count(), 0);
    }

    #[test]
    fn slice_is_zero_copy_and_reindexes_correctly() {
        let arr = int_array_with_nulls();
        let sliced = arr.slice(1, 3); // [NULL, 30, 40]
        let sliced = as_primitive::<Int64Type>(sliced.as_ref()).unwrap();
        assert_eq!(sliced.len(), 3);
        assert!(sliced.is_null(0));
        assert_eq!(sliced.value(1), 30);
        assert_eq!(sliced.value(2), 40);
    }

    #[test]
    fn try_new_rejects_length_not_a_multiple_of_element_size() {
        let buf = Buffer::from_vec(vec![1u8, 2, 3]); // 3 bytes, not a multiple of 8
        assert!(PrimitiveArray::<Int64Type>::try_new(buf, None).is_err());
    }

    #[test]
    fn try_new_rejects_validity_length_mismatch() {
        let buf = Buffer::from_vec(vec![1i64, 2, 3]);
        let validity = Bitmap::new_all_set(2); // should be 3
        assert!(PrimitiveArray::<Int64Type>::try_new(buf, Some(validity)).is_err());
    }

    #[test]
    fn downcast_to_wrong_type_errors() {
        let arr = int_array_with_nulls();
        let arr_ref: ArrayRef = Arc::new(arr);
        assert!(as_primitive::<Float64Type>(arr_ref.as_ref()).is_err());
    }

    #[test]
    fn empty_array_has_zero_length() {
        let b = PrimitiveBuilder::<Int64Type>::with_capacity(0);
        let arr = b.finish();
        assert!(arr.is_empty());
        assert_eq!(arr.null_count(), 0);
    }
}
