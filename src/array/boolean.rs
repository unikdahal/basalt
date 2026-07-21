//! `BooleanArray` — bit-packed boolean values. See
//! design-docs/basalt-phase2-lld.md §3.3.

use std::any::Any;
use std::sync::Arc;

use super::array::{Array, ArrayRef};
use crate::buffer::{Bitmap, BitmapBuilder};
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;

/// A boolean column is *two* bitmaps: values and validity, both bit-packed.
///
/// # Invariants
/// - If `validity` is `Some`, its length equals `values.len()`.
#[derive(Clone, Debug)]
pub struct BooleanArray {
    values: Bitmap,
    validity: Option<Bitmap>,
}

impl BooleanArray {
    /// # Errors
    /// Errors if `validity`'s length doesn't match `values`'s.
    pub fn try_new(values: Bitmap, validity: Option<Bitmap>) -> Result<Self> {
        if let Some(v) = &validity {
            if v.len() != values.len() {
                return Err(BasaltError::Internal(format!(
                    "validity length {} does not match array length {}",
                    v.len(),
                    values.len()
                )));
            }
        }
        Ok(BooleanArray { values, validity })
    }

    /// The value at `i`. Does NOT check validity — caller must.
    pub fn value(&self, i: usize) -> bool {
        self.values.get(i)
    }

    pub fn values(&self) -> &Bitmap {
        &self.values
    }
}

impl Array for BooleanArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self) -> DataType {
        DataType::Boolean
    }

    fn len(&self) -> usize {
        self.values.len()
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
            offset + len <= self.len(),
            "BooleanArray::slice out of bounds"
        );
        Arc::new(BooleanArray {
            values: self.values.slice(offset, len),
            validity: self.validity.as_ref().map(|v| v.slice(offset, len)),
        })
    }
}

/// Incremental, bit-packed construction of a `BooleanArray`.
pub struct BooleanBuilder {
    values: BitmapBuilder,
    validity: BitmapBuilder,
    null_count: usize,
}

impl BooleanBuilder {
    pub fn with_capacity(capacity: usize) -> Self {
        BooleanBuilder {
            values: BitmapBuilder::with_capacity(capacity),
            validity: BitmapBuilder::with_capacity(capacity),
            null_count: 0,
        }
    }

    pub fn append_value(&mut self, v: bool) {
        self.values.push(v);
        self.validity.push(true);
    }

    pub fn append_null(&mut self) {
        self.values.push(false);
        self.validity.push(false);
        self.null_count += 1;
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Consumes the builder. Drops validity entirely if no nulls were appended.
    pub fn finish(self) -> BooleanArray {
        let validity = if self.null_count == 0 {
            None
        } else {
            Some(self.validity.finish())
        };
        BooleanArray {
            values: self.values.finish(),
            validity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_round_trips_values_and_nulls() {
        let mut b = BooleanBuilder::with_capacity(4);
        b.append_value(true);
        b.append_null();
        b.append_value(false);
        let arr = b.finish();

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        assert!(arr.value(0));
        assert!(arr.is_null(1));
        assert!(!arr.value(2));
    }

    #[test]
    fn no_nulls_seen_means_validity_is_none() {
        let mut b = BooleanBuilder::with_capacity(2);
        b.append_value(true);
        b.append_value(false);
        let arr = b.finish();
        assert!(arr.validity().is_none());
    }

    #[test]
    fn slice_is_zero_copy_and_reindexes_correctly() {
        let mut b = BooleanBuilder::with_capacity(4);
        b.append_value(true);
        b.append_null();
        b.append_value(false);
        b.append_value(true);
        let arr = b.finish();

        let sliced = arr.slice(1, 2);
        assert_eq!(sliced.len(), 2);
        assert!(sliced.is_null(0));
        assert!(!sliced.is_null(1));
    }

    #[test]
    fn try_new_rejects_validity_length_mismatch() {
        let values = Bitmap::new_all_set(3);
        let validity = Bitmap::new_all_set(2);
        assert!(BooleanArray::try_new(values, Some(validity)).is_err());
    }
}
