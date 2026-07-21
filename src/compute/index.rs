//! `UInt32Array` — row-position indices for `take`/`filter`/sort/join.
//!
//! Deliberately **not** an `Array` trait implementor and not routed through
//! `DataType`: index arrays are a physical-execution concept (row positions
//! into another array), never a SQL-visible column type. Adding a `UInt32`
//! variant to `DataType` — the logical SQL type lattice Phase 1 deliberately
//! kept to four members — would let a purely internal concept leak into
//! schemas and `CAST` targets for no benefit. Keeping it a standalone type
//! avoids that without touching Phase 1's type lattice at all.

use crate::buffer::{Bitmap, BitmapBuilder, Buffer, MutableBuffer};
use crate::error::{BasaltError, Result};

/// # Invariants
/// - `values.len()` is a multiple of `size_of::<u32>()`.
/// - If `validity` is `Some`, its length equals the element count.
#[derive(Clone, Debug)]
pub struct UInt32Array {
    values: Buffer,
    validity: Option<Bitmap>,
    len: usize,
}

impl UInt32Array {
    /// # Errors
    /// Errors if `values`'s length isn't a multiple of 4 bytes, isn't
    /// aligned for `u32`, or `validity`'s length doesn't match.
    pub fn try_new(values: Buffer, validity: Option<Bitmap>) -> Result<Self> {
        let elem_size = std::mem::size_of::<u32>();
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
        values.typed_data::<u32>()?;
        Ok(UInt32Array {
            values,
            validity,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_null(&self, i: usize) -> bool {
        self.validity.as_ref().is_some_and(|v| !v.get(i))
    }

    /// The index at `i`. Does NOT check validity — caller must.
    pub fn value(&self, i: usize) -> u32 {
        // SAFETY: validated once in `try_new`; `Buffer` is immutable.
        let full = unsafe { self.values.typed_data_unchecked::<u32>() };
        full[i]
    }
}

pub struct UInt32Builder {
    values: MutableBuffer,
    validity: BitmapBuilder,
    null_count: usize,
}

impl UInt32Builder {
    pub fn with_capacity(capacity: usize) -> Self {
        UInt32Builder {
            values: MutableBuffer::with_capacity(capacity * std::mem::size_of::<u32>()),
            validity: BitmapBuilder::with_capacity(capacity),
            null_count: 0,
        }
    }

    pub fn append_value(&mut self, v: u32) {
        self.values.push(v);
        self.validity.push(true);
    }

    pub fn append_null(&mut self) {
        self.values.push(0u32);
        self.validity.push(false);
        self.null_count += 1;
    }

    pub fn len(&self) -> usize {
        self.validity.len()
    }

    pub fn is_empty(&self) -> bool {
        self.validity.is_empty()
    }

    pub fn finish(self) -> UInt32Array {
        let len = self.validity.len();
        let values = self.values.freeze();
        let validity = if self.null_count == 0 {
            None
        } else {
            Some(self.validity.finish())
        };
        UInt32Array {
            values,
            validity,
            len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_round_trips_values_and_nulls() {
        let mut b = UInt32Builder::with_capacity(3);
        b.append_value(5);
        b.append_null();
        b.append_value(7);
        let arr = b.finish();

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value(0), 5);
        assert!(arr.is_null(1));
        assert_eq!(arr.value(2), 7);
    }

    #[test]
    fn no_nulls_seen_means_no_validity_allocated() {
        let mut b = UInt32Builder::with_capacity(2);
        b.append_value(1);
        b.append_value(2);
        let arr = b.finish();
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(1));
    }

    #[test]
    fn try_new_rejects_length_not_a_multiple_of_four() {
        let buf = Buffer::from_vec(vec![1u8, 2, 3]);
        assert!(UInt32Array::try_new(buf, None).is_err());
    }
}
