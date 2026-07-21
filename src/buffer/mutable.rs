//! `MutableBuffer` — the construction-time counterpart to `Buffer`: growable,
//! exclusively owned, then frozen. See design-docs/basalt-phase2-lld.md §3.1.

use super::buffer::Buffer;
use super::native::NativeType;

pub struct MutableBuffer {
    data: Vec<u8>,
}

impl MutableBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        MutableBuffer {
            data: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
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
        Buffer::from_vec(self.data)
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
}
