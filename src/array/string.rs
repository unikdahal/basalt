//! `StringArray` — variable-length UTF-8 text, offsets + values layout. See
//! design-docs/basalt-phase2-lld.md §3.3.

use std::any::Any;
use std::sync::Arc;

use super::array::{Array, ArrayRef};
use crate::buffer::{Bitmap, BitmapBuilder, Buffer, MutableBuffer};
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;

/// Two buffers, no per-string allocation: `value(i)` is two offset loads
/// plus a slice, not a pointer chase.
///
/// # Invariants
/// - `offsets` holds `len + 1` `i32` entries, monotonically non-decreasing.
/// - String `i` occupies `values[offsets[i]..offsets[i+1]]`.
/// - All bytes in `values` are valid UTF-8 (checked once in `try_new`), and
///   every offset lands on a UTF-8 character boundary — guaranteed by
///   `StringBuilder`, which only ever appends whole `&str`s and therefore
///   only ever records offsets at whole-string boundaries.
#[derive(Clone, Debug)]
pub struct StringArray {
    offsets: Buffer,
    values: Buffer,
    validity: Option<Bitmap>,
    len: usize,
}

impl StringArray {
    /// # Errors
    /// Errors if `offsets` is empty, not non-decreasing, if `values` isn't
    /// valid UTF-8, or if `validity`'s length doesn't match.
    pub fn try_new(offsets: Buffer, values: Buffer, validity: Option<Bitmap>) -> Result<Self> {
        let offsets_typed = offsets.typed_data::<i32>()?;
        if offsets_typed.is_empty() {
            return Err(BasaltError::Internal(
                "offsets buffer must have at least 1 entry".to_string(),
            ));
        }
        let len = offsets_typed.len() - 1;
        for w in offsets_typed.windows(2) {
            if w[1] < w[0] {
                return Err(BasaltError::Internal(
                    "offsets must be non-decreasing".to_string(),
                ));
            }
        }
        std::str::from_utf8(values.as_slice())
            .map_err(|e| BasaltError::Internal(format!("values buffer is not valid UTF-8: {e}")))?;
        if let Some(v) = &validity {
            if v.len() != len {
                return Err(BasaltError::Internal(format!(
                    "validity length {} does not match array length {len}",
                    v.len()
                )));
            }
        }
        Ok(StringArray {
            offsets,
            values,
            validity,
            len,
        })
    }

    /// The string at `i`. Does NOT check validity — caller must.
    pub fn value(&self, i: usize) -> &str {
        // SAFETY: `try_new` validated the whole `values` buffer as UTF-8 once,
        // and every offset was recorded by `StringBuilder` at a whole-`&str`
        // boundary, so `[start, end)` always lands on a char boundary.
        let offsets = unsafe { self.offsets.typed_data_unchecked::<i32>() };
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        unsafe { std::str::from_utf8_unchecked(&self.values.as_slice()[start..end]) }
    }
}

impl Array for StringArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self) -> DataType {
        DataType::Utf8
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
        debug_assert!(offset + len <= self.len, "StringArray::slice out of bounds");
        // `values` is shared as-is (still O(1)/zero-copy: an Arc clone, no
        // byte copy); only the window over `offsets` changes. The absolute
        // byte offsets already stored still index correctly into the
        // unchanged `values` buffer.
        let elem_size = std::mem::size_of::<i32>();
        Arc::new(StringArray {
            offsets: self
                .offsets
                .slice(offset * elem_size, (len + 1) * elem_size),
            values: self.values.clone(),
            validity: self.validity.as_ref().map(|v| v.slice(offset, len)),
            len,
        })
    }
}

/// Incremental construction of a `StringArray`.
pub struct StringBuilder {
    offsets: MutableBuffer,
    values: Vec<u8>,
    validity: BitmapBuilder,
    null_count: usize,
    len: usize,
}

impl StringBuilder {
    pub fn with_capacity(capacity: usize, data_capacity: usize) -> Self {
        let mut offsets = MutableBuffer::with_capacity((capacity + 1) * std::mem::size_of::<i32>());
        offsets.push(0i32);
        StringBuilder {
            offsets,
            values: Vec::with_capacity(data_capacity),
            validity: BitmapBuilder::with_capacity(capacity),
            null_count: 0,
            len: 0,
        }
    }

    /// # Errors
    /// Errors if the cumulative byte length of appended strings would
    /// overflow `i32`.
    pub fn append_value(&mut self, s: &str) -> Result<()> {
        self.values.extend_from_slice(s.as_bytes());
        let end = i32::try_from(self.values.len())
            .map_err(|_| BasaltError::Internal("StringArray offsets overflowed i32".to_string()))?;
        self.offsets.push(end);
        self.validity.push(true);
        self.len += 1;
        Ok(())
    }

    pub fn append_null(&mut self) {
        // Zero-length span at the current end — no bytes written, no offset
        // change other than repeating the last one.
        let end = self.values.len() as i32;
        self.offsets.push(end);
        self.validity.push(false);
        self.null_count += 1;
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Consumes the builder. Drops validity entirely if no nulls were appended.
    pub fn finish(self) -> StringArray {
        let validity = if self.null_count == 0 {
            None
        } else {
            Some(self.validity.finish())
        };
        let offsets = self.offsets.freeze();
        let values = Buffer::from_vec(self.values);
        StringArray {
            offsets,
            values,
            validity,
            len: self.len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_round_trips_values_and_nulls() {
        let mut b = StringBuilder::with_capacity(3, 16);
        b.append_value("hello").unwrap();
        b.append_null();
        b.append_value("world").unwrap();
        let arr = b.finish();

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        assert_eq!(arr.value(0), "hello");
        assert!(arr.is_null(1));
        assert_eq!(arr.value(2), "world");
    }

    #[test]
    fn empty_strings_are_distinct_from_nulls() {
        let mut b = StringBuilder::with_capacity(2, 0);
        b.append_value("").unwrap();
        b.append_null();
        let arr = b.finish();

        assert!(!arr.is_null(0));
        assert_eq!(arr.value(0), "");
        assert!(arr.is_null(1));
    }

    #[test]
    fn non_ascii_utf8_round_trips() {
        let mut b = StringBuilder::with_capacity(1, 16);
        b.append_value("héllo wörld 日本語").unwrap();
        let arr = b.finish();
        assert_eq!(arr.value(0), "héllo wörld 日本語");
    }

    #[test]
    fn no_nulls_seen_means_validity_is_none() {
        let mut b = StringBuilder::with_capacity(1, 8);
        b.append_value("x").unwrap();
        let arr = b.finish();
        assert!(arr.validity().is_none());
    }

    #[test]
    fn slice_shares_the_values_buffer_zero_copy() {
        let mut b = StringBuilder::with_capacity(3, 16);
        b.append_value("aa").unwrap();
        b.append_value("bb").unwrap();
        b.append_value("cc").unwrap();
        let arr = b.finish();

        let sliced = arr.slice(1, 2);
        assert_eq!(sliced.len(), 2);
        let sliced = crate::array::array::as_string(sliced.as_ref()).unwrap();
        assert_eq!(sliced.value(0), "bb");
        assert_eq!(sliced.value(1), "cc");
    }

    #[test]
    fn try_new_rejects_non_utf8_values() {
        let offsets = Buffer::from_vec(vec![0i32, 3]);
        let values = Buffer::from_vec(vec![0xFFu8, 0xFE, 0xFD]); // invalid UTF-8
        assert!(StringArray::try_new(offsets, values, None).is_err());
    }

    #[test]
    fn try_new_rejects_decreasing_offsets() {
        let offsets = Buffer::from_vec(vec![0i32, 5, 2]);
        let values = Buffer::from_vec(b"hello".to_vec());
        assert!(StringArray::try_new(offsets, values, None).is_err());
    }
}
