//! `Bitmap` — bit-packed validity/boolean sequence. See
//! design-docs/basalt-phase2-lld.md §3.2.
//!
//! `offset`/`len` are in **bits**, not bytes — slicing at a non-byte boundary
//! is the normal case (e.g. after `RecordBatch::slice`), and every operation
//! must handle it correctly. The LLD flags this as the highest bug-density
//! area in Phase 2; the test suite below exercises offsets 1, 7, 8, 9, 63,
//! 64, and 65 explicitly, as instructed.
//!
//! `and`/`or`/`not` are implemented bit-at-a-time here, not word-at-a-time.
//! That's a deliberate correctness-first choice: get every offset case right
//! with the simplest possible implementation, then optimize once a benchmark
//! shows it matters (this project's own rule: no performance claim without a
//! measurement). The word-at-a-time fast path described in the LLD is a
//! follow-up, not a shortcut taken here.

use super::buffer::Buffer;
use crate::error::{BasaltError, Result};

/// Reads bit `bit_offset + i` from an already-hoisted byte slice — the
/// bit-level twin of indexing a hoisted `values()` slice in
/// `compute::arith`. Pair with [`Bitmap::as_bytes`] / [`Bitmap::bit_offset`]
/// to avoid re-deriving the slice on every element in a loop.
#[inline]
pub fn bit_at(bytes: &[u8], bit_offset: usize, i: usize) -> bool {
    let bit = bit_offset + i;
    (bytes[bit / 8] >> (bit % 8)) & 1 != 0
}

#[derive(Clone, Debug)]
pub struct Bitmap {
    buffer: Buffer,
    offset: usize,
    len: usize,
    null_count: usize,
}

impl Bitmap {
    pub fn new_all_set(len: usize) -> Self {
        let byte_len = len.div_ceil(8);
        let buffer = Buffer::from_vec(vec![0xFFu8; byte_len]);
        Bitmap {
            buffer,
            offset: 0,
            len,
            null_count: 0,
        }
    }

    pub fn new_all_unset(len: usize) -> Self {
        let byte_len = len.div_ceil(8);
        let buffer = Buffer::from_vec(vec![0u8; byte_len]);
        Bitmap {
            buffer,
            offset: 0,
            len,
            null_count: len,
        }
    }

    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.len, "Bitmap::get index out of bounds");
        let bit = self.offset + i;
        let byte = self.buffer.as_slice()[bit / 8];
        (byte >> (bit % 8)) & 1 != 0
    }

    /// Raw byte access for hot per-element loops that iterate the whole
    /// bitmap: `Buffer::as_slice()` re-derives its slice (`Arc` deref plus
    /// two nested offset/len computations) on every call, so calling `get`
    /// in a tight loop pays that cost once per bit instead of once for the
    /// whole scan — the same class of bug `compute::arith`'s `value(i)` had
    /// (see that module's doc comment). Pair with [`Self::bit_offset`] and
    /// [`bit_at`] to hoist the slice once before the loop.
    pub fn as_bytes(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    /// The bit offset to add to a logical index before indexing
    /// [`Self::as_bytes`] — see [`bit_at`].
    pub fn bit_offset(&self) -> usize {
        self.offset
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn null_count(&self) -> usize {
        self.null_count
    }

    /// Shares the buffer; only the bit offset changes (O(1)). `null_count`
    /// for the new range is recomputed with a linear scan — cheap relative
    /// to the batch sizes this is used at, and correct regardless of the bit
    /// offset, which is the property that matters most here.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        debug_assert!(offset + len <= self.len, "Bitmap::slice out of bounds");
        let sliced = Bitmap {
            buffer: self.buffer.clone(),
            offset: self.offset + offset,
            len,
            null_count: 0,
        };
        let null_count = (0..len).filter(|&i| !sliced.get(i)).count();
        Bitmap {
            null_count,
            ..sliced
        }
    }

    pub fn and(&self, other: &Bitmap) -> Result<Bitmap> {
        self.zip_with(other, |a, b| a && b)
    }

    pub fn or(&self, other: &Bitmap) -> Result<Bitmap> {
        self.zip_with(other, |a, b| a || b)
    }

    pub fn not(&self) -> Bitmap {
        let bytes = self.as_bytes();
        let offset = self.bit_offset();
        let mut builder = BitmapBuilder::with_capacity(self.len);
        for i in 0..self.len {
            builder.push(!bit_at(bytes, offset, i));
        }
        builder.finish()
    }

    fn zip_with(&self, other: &Bitmap, f: impl Fn(bool, bool) -> bool) -> Result<Bitmap> {
        if self.len != other.len {
            return Err(BasaltError::Internal(format!(
                "bitmap length mismatch: {} vs {}",
                self.len, other.len
            )));
        }
        let (l_bytes, l_offset) = (self.as_bytes(), self.bit_offset());
        let (r_bytes, r_offset) = (other.as_bytes(), other.bit_offset());
        let mut builder = BitmapBuilder::with_capacity(self.len);
        for i in 0..self.len {
            builder.push(f(
                bit_at(l_bytes, l_offset, i),
                bit_at(r_bytes, r_offset, i),
            ));
        }
        Ok(builder.finish())
    }

    /// Iterate set-bit positions in ascending order.
    pub fn set_indices(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter(move |&i| self.get(i))
    }
}

/// Incremental, bit-packed construction of a `Bitmap`.
pub struct BitmapBuilder {
    bytes: Vec<u8>,
    len: usize,
}

impl BitmapBuilder {
    pub fn with_capacity(capacity: usize) -> Self {
        BitmapBuilder {
            bytes: vec![0u8; capacity.div_ceil(8)],
            len: 0,
        }
    }

    pub fn push(&mut self, value: bool) {
        let byte_idx = self.len / 8;
        if byte_idx >= self.bytes.len() {
            self.bytes.push(0);
        }
        if value {
            self.bytes[byte_idx] |= 1 << (self.len % 8);
        }
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn finish(self) -> Bitmap {
        let null_count = (0..self.len)
            .filter(|&i| (self.bytes[i / 8] >> (i % 8)) & 1 == 0)
            .count();
        let buffer = Buffer::from_vec(self.bytes);
        Bitmap {
            buffer,
            offset: 0,
            len: self.len,
            null_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_bools(bits: &[bool]) -> Bitmap {
        let mut b = BitmapBuilder::with_capacity(bits.len());
        for &bit in bits {
            b.push(bit);
        }
        b.finish()
    }

    #[test]
    fn new_all_set_has_no_nulls() {
        let bm = Bitmap::new_all_set(10);
        assert_eq!(bm.null_count(), 0);
        assert!((0..10).all(|i| bm.get(i)));
    }

    #[test]
    fn new_all_unset_is_fully_null() {
        let bm = Bitmap::new_all_unset(10);
        assert_eq!(bm.null_count(), 10);
        assert!((0..10).all(|i| !bm.get(i)));
    }

    #[test]
    fn builder_round_trips_arbitrary_pattern() {
        let pattern = [
            true, false, true, true, false, false, true, false, true, true, false,
        ];
        let bm = from_bools(&pattern);
        for (i, &expected) in pattern.iter().enumerate() {
            assert_eq!(bm.get(i), expected, "mismatch at bit {i}");
        }
        assert_eq!(bm.null_count(), pattern.iter().filter(|&&b| !b).count());
    }

    /// The bit-offset test suite the LLD calls out by name: slicing at these
    /// exact offsets is where naive byte-aligned implementations break.
    #[test]
    fn slice_at_every_flagged_bit_offset() {
        // 80 bits so every flagged offset (1,7,8,9,63,64,65) has room for a
        // meaningful window after it.
        let pattern: Vec<bool> = (0..80).map(|i| i % 3 == 0).collect();
        let bm = from_bools(&pattern);

        for &offset in &[1usize, 7, 8, 9, 63, 64, 65] {
            let len = 10;
            let sliced = bm.slice(offset, len);
            for i in 0..len {
                assert_eq!(
                    sliced.get(i),
                    pattern[offset + i],
                    "offset {offset}, bit {i} mismatch"
                );
            }
            let expected_nulls = pattern[offset..offset + len]
                .iter()
                .filter(|&&b| !b)
                .count();
            assert_eq!(
                sliced.null_count(),
                expected_nulls,
                "offset {offset} null_count mismatch"
            );
        }
    }

    #[test]
    fn and_matches_truth_table_at_nonzero_offset() {
        let a = from_bools(&[
            true, true, false, false, true, true, false, false, true, true,
        ]);
        let b = from_bools(&[
            true, false, true, false, true, false, true, false, true, false,
        ]);
        // Slice both at offset 3 to force non-byte-aligned reads through `and`.
        let a = a.slice(3, 5);
        let b = b.slice(3, 5);
        let result = a.and(&b).unwrap();
        for i in 0..5 {
            assert_eq!(result.get(i), a.get(i) && b.get(i), "AND mismatch at {i}");
        }
    }

    #[test]
    fn or_and_not_match_truth_tables() {
        let a = from_bools(&[true, false, true, false]);
        let b = from_bools(&[true, true, false, false]);
        let or = a.or(&b).unwrap();
        assert!(or.get(0));
        assert!(or.get(1));
        assert!(or.get(2));
        assert!(!or.get(3));

        let not_a = a.not();
        assert!(!not_a.get(0));
        assert!(not_a.get(1));
        assert!(!not_a.get(2));
        assert!(not_a.get(3));
    }

    #[test]
    fn and_rejects_mismatched_lengths() {
        let a = Bitmap::new_all_set(3);
        let b = Bitmap::new_all_set(4);
        assert!(a.and(&b).is_err());
    }

    #[test]
    fn set_indices_yields_only_set_bits_in_order() {
        let bm = from_bools(&[false, true, false, true, true, false]);
        assert_eq!(bm.set_indices().collect::<Vec<_>>(), vec![1, 3, 4]);
    }

    #[test]
    fn empty_bitmap_has_zero_length_and_no_nulls() {
        let bm = Bitmap::new_all_set(0);
        assert!(bm.is_empty());
        assert_eq!(bm.null_count(), 0);
        assert_eq!(bm.set_indices().count(), 0);
    }
}
