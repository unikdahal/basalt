//! `MutableBuffer` — the construction-time counterpart to `Buffer`: growable,
//! exclusively owned, then frozen. See design-docs/basalt-phase2-lld.md §3.1.

use super::buffer::{Buffer, ALIGNMENT};
use super::native::NativeType;

/// `data`'s first `align_offset` bytes are inert padding, reserved up front
/// so the *content* that follows starts on an `ALIGNMENT`-byte boundary —
/// mirroring `Buffer`'s own `AlignedBytes` layout exactly. That lets
/// `freeze()` hand `data` straight to `Buffer` as a move in the common case,
/// instead of allocating a second buffer and copying into it purely to fix
/// up alignment. Measured via `examples/kernel_only_manual.rs`: the old
/// always-copy `freeze()` cost as much as the entire rest of a `col + 1`
/// kernel call combined at N=1,000,000 (a second full-buffer allocation and
/// memcpy on top of the one actually computing the result).
pub struct MutableBuffer {
    data: Vec<u8>,
    align_offset: usize,
}

impl MutableBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity + ALIGNMENT);
        let ptr = data.as_ptr() as usize;
        let align_offset = (ALIGNMENT - (ptr % ALIGNMENT)) % ALIGNMENT;
        data.resize(align_offset, 0);
        MutableBuffer { data, align_offset }
    }

    pub fn len(&self) -> usize {
        self.data.len() - self.align_offset
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push<T: NativeType>(&mut self, value: T) {
        let size = std::mem::size_of::<T>();
        // SAFETY: `T: NativeType` guarantees no padding / validity for any
        // bit pattern; we only read `value`'s own bytes, once, into `data`.
        let bytes = unsafe { std::slice::from_raw_parts((&value as *const T).cast::<u8>(), size) };
        self.data.extend_from_slice(bytes);
    }

    pub fn extend_from_slice<T: NativeType>(&mut self, values: &[T]) {
        let byte_len = std::mem::size_of_val(values);
        // SAFETY: same contract as `push`, extended over the whole slice.
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) };
        self.data.extend_from_slice(bytes);
    }

    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Consumes the builder, producing an immutable, aligned `Buffer`. No
    /// `MutableBuffer` can outlive this call, so there is no way to alias
    /// the resulting immutable bytes through a mutable path.
    pub fn freeze(self) -> Buffer {
        Buffer::from_padded_vec(self.data, self.align_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_freeze_round_trips_values() {
        let mut b = MutableBuffer::with_capacity(0);
        b.push(1i64);
        b.push(2i64);
        b.push(3i64);
        let buf = b.freeze();
        assert_eq!(buf.typed_data::<i64>().unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn extend_from_slice_matches_repeated_push() {
        let mut a = MutableBuffer::with_capacity(0);
        a.extend_from_slice(&[1i64, 2, 3]);

        let mut b = MutableBuffer::with_capacity(0);
        b.push(1i64);
        b.push(2i64);
        b.push(3i64);

        assert_eq!(
            a.freeze().typed_data::<i64>().unwrap(),
            b.freeze().typed_data::<i64>().unwrap()
        );
    }

    #[test]
    fn empty_buffer_freezes_to_empty() {
        let b = MutableBuffer::with_capacity(16);
        assert!(b.is_empty());
        let buf = b.freeze();
        assert!(buf.is_empty());
    }

    #[test]
    fn frozen_buffer_is_aligned() {
        let mut b = MutableBuffer::with_capacity(0);
        b.push(1i64);
        let buf = b.freeze();
        assert_eq!(
            buf.as_slice().as_ptr() as usize % super::super::buffer::ALIGNMENT,
            0
        );
    }

    #[test]
    fn frozen_buffer_is_aligned_even_when_capacity_is_underestimated() {
        // Under-reserving forces `Vec` to reallocate mid-construction, which
        // can land the backing allocation at a different alignment relative
        // to the padding computed at `with_capacity` time. `freeze` must
        // still produce a correctly aligned `Buffer` in that case (falling
        // back to a copy internally rather than trusting stale padding).
        let mut b = MutableBuffer::with_capacity(0);
        for i in 0..10_000i64 {
            b.push(i);
        }
        let buf = b.freeze();
        assert_eq!(
            buf.as_slice().as_ptr() as usize % super::super::buffer::ALIGNMENT,
            0
        );
        assert_eq!(buf.typed_data::<i64>().unwrap().len(), 10_000);
        assert_eq!(buf.typed_data::<i64>().unwrap()[9999], 9999);
    }
}
