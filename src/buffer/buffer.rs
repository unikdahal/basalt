//! `Buffer` — immutable, reference-counted, aligned byte storage. See
//! design-docs/basalt-phase2-lld.md §3.1.

use std::sync::Arc;

use super::native::NativeType;
use crate::error::{BasaltError, Result};

pub const ALIGNMENT: usize = 64;

/// Backing allocation for one or more `Buffer` views.
///
/// Deliberately *not* a custom raw allocator: `raw` is a plain `Vec<u8>`
/// over-allocated by up to `ALIGNMENT` bytes, and `align_offset` is computed
/// once so every read through `as_slice()` starts on a genuinely
/// `ALIGNMENT`-byte-aligned address. This trades a few dozen wasted bytes per
/// buffer for zero unsafe allocator code — the LLD's own non-goal is "no
/// `unsafe` unless measured," and a hand-rolled aligned allocator is exactly
/// the kind of unsafe that isn't justified without a benchmark showing this
/// approach's overhead actually matters.
#[derive(Debug)]
struct AlignedBytes {
    raw: Vec<u8>,
    align_offset: usize,
    len: usize,
}

impl AlignedBytes {
    fn new(len: usize) -> Self {
        let raw = vec![0u8; len + ALIGNMENT];
        let ptr = raw.as_ptr() as usize;
        let align_offset = (ALIGNMENT - (ptr % ALIGNMENT)) % ALIGNMENT;
        AlignedBytes {
            raw,
            align_offset,
            len,
        }
    }

    fn as_slice(&self) -> &[u8] {
        &self.raw[self.align_offset..self.align_offset + self.len]
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        let start = self.align_offset;
        let end = start + self.len;
        &mut self.raw[start..end]
    }
}

/// An immutable, reference-counted, aligned region of bytes.
///
/// # Invariants
/// - `offset + len <= data.len()` (byte units).
/// - The start of the underlying allocation is aligned to [`ALIGNMENT`] bytes.
/// - Contents never change after construction.
#[derive(Clone, Debug)]
pub struct Buffer {
    data: Arc<AlignedBytes>,
    offset: usize,
    len: usize,
}

impl Buffer {
    /// Copies `values` into a freshly allocated, aligned buffer.
    pub fn from_vec<T: NativeType>(values: Vec<T>) -> Self {
        let elem_size = std::mem::size_of::<T>();
        let byte_len = values.len() * elem_size;
        let mut aligned = AlignedBytes::new(byte_len);
        // SAFETY: `T: NativeType` guarantees no padding and validity for any
        // bit pattern, so reading `values` as `byte_len` bytes is
        // well-defined; we only read through this slice, never mutate or
        // alias it mutably.
        let src = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) };
        aligned.as_mut_slice().copy_from_slice(src);
        Buffer {
            data: Arc::new(aligned),
            offset: 0,
            len: byte_len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data.as_slice()[self.offset..self.offset + self.len]
    }

    /// O(1). Shares the same allocation; only offset and len change.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        debug_assert!(offset + len <= self.len, "Buffer::slice out of bounds");
        Buffer {
            data: Arc::clone(&self.data),
            offset: self.offset + offset,
            len,
        }
    }

    /// Reinterpret as a typed slice.
    ///
    /// # Errors
    /// Returns an error if the length isn't a multiple of `size_of::<T>()`
    /// or the starting address isn't aligned for `T`.
    pub fn typed_data<T: NativeType>(&self) -> Result<&[T]> {
        let elem_size = std::mem::size_of::<T>();
        if !self.len.is_multiple_of(elem_size) {
            return Err(BasaltError::Internal(format!(
                "buffer length {} is not a multiple of element size {elem_size}",
                self.len
            )));
        }
        let ptr = self.as_slice().as_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return Err(BasaltError::Internal(
                "buffer is not aligned for the requested type".to_string(),
            ));
        }
        // SAFETY: length and alignment just checked above; `T: NativeType`
        // guarantees any bit pattern is a valid `T`.
        Ok(unsafe { self.typed_data_unchecked() })
    }

    /// # Safety
    /// Caller must ensure `self.len()` is a multiple of `size_of::<T>()` and
    /// that `self.as_slice().as_ptr()` is aligned for `T` — both verified
    /// once by `typed_data`, or by a constructor that maintains the
    /// invariant itself (e.g. `PrimitiveArray::try_new`).
    pub(crate) unsafe fn typed_data_unchecked<T: NativeType>(&self) -> &[T] {
        let elem_size = std::mem::size_of::<T>();
        let slice = self.as_slice();
        std::slice::from_raw_parts(slice.as_ptr().cast::<T>(), slice.len() / elem_size)
    }

    #[cfg(test)]
    pub(crate) fn strong_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_is_aligned_to_64_bytes() {
        let buf = Buffer::from_vec(vec![1i64, 2, 3]);
        assert_eq!(buf.as_slice().as_ptr() as usize % ALIGNMENT, 0);
    }

    #[test]
    fn typed_data_round_trips() {
        let buf = Buffer::from_vec(vec![1i64, 2, 3, -4]);
        let values = buf.typed_data::<i64>().unwrap();
        assert_eq!(values, &[1, 2, 3, -4]);
    }

    #[test]
    fn typed_data_errors_on_length_mismatch() {
        // 3 bytes can't be reinterpreted as any whole number of i64s.
        let buf = Buffer::from_vec(vec![1u8, 2, 3]);
        assert!(buf.typed_data::<i64>().is_err());
    }

    #[test]
    fn slice_shares_the_allocation_zero_copy() {
        let buf = Buffer::from_vec(vec![1i64, 2, 3, 4, 5]);
        let before = buf.strong_count();
        let sliced = buf.slice(8, 16); // elements 1..3, i.e. [2, 3]
        assert_eq!(buf.strong_count(), before + 1);
        assert_eq!(sliced.typed_data::<i64>().unwrap(), &[2, 3]);
        // Slicing at byte offset 0 must yield the exact same start address —
        // proof that no copy happened.
        let identity_slice = buf.slice(0, buf.len());
        assert_eq!(identity_slice.as_slice().as_ptr(), buf.as_slice().as_ptr());
    }

    #[test]
    fn empty_buffer_has_zero_length() {
        let buf = Buffer::from_vec(Vec::<i64>::new());
        assert!(buf.is_empty());
        assert_eq!(buf.typed_data::<i64>().unwrap(), &[] as &[i64]);
    }

    #[test]
    fn slice_of_bytes_can_be_misaligned_for_wider_types() {
        // Slicing at a 1-byte offset breaks 8-byte alignment for i64 by
        // construction — typed_data must report that, not read garbage.
        let buf = Buffer::from_vec(vec![0u8; 32]);
        let misaligned = buf.slice(1, 16);
        assert!(misaligned.typed_data::<i64>().is_err());
    }
}
