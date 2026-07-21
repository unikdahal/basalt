//! `filter` — select rows where a predicate is exactly `TRUE`. See
//! design-docs/basalt-phase2-lld.md §4.5.
//!
//! Implements the two cheap shortcuts (nothing selected, everything
//! selected) plus the general case built on `take`. The LLD's selectivity-
//! based run-detection fast path for the dense case is a real, worthwhile
//! optimization but a deferred one here — this implementation is correct
//! for every selectivity first, matching this project's "measure before
//! optimizing" discipline; the run-based `extend_from_slice` path is a
//! follow-up once a benchmark shows the gather-per-row cost actually
//! dominates a profile.

use crate::array::array::{Array, ArrayRef};
use crate::array::boolean::BooleanArray;
use crate::compute::index::UInt32Builder;
use crate::compute::take::take;
use crate::error::{BasaltError, Result};

/// Select rows where `predicate` is exactly `true` — both `false` and `NULL`
/// reject the row (SQL `WHERE` semantics, matching Phase 1's `eval_predicate`).
///
/// # Errors
/// Errors if `predicate`'s length doesn't match `array`'s.
pub fn filter(array: &dyn Array, predicate: &BooleanArray) -> Result<ArrayRef> {
    if predicate.len() != array.len() {
        return Err(BasaltError::Internal(format!(
            "predicate length {} does not match array length {}",
            predicate.len(),
            array.len()
        )));
    }

    let selected = (0..predicate.len())
        .filter(|&i| predicate.is_valid(i) && predicate.value(i))
        .count();

    if selected == 0 {
        return Ok(array.slice(0, 0));
    }
    if selected == array.len() {
        // Every row passed: an O(1) Arc-shared slice of the whole array,
        // not a copy.
        return Ok(array.slice(0, array.len()));
    }

    let mut indices = UInt32Builder::with_capacity(selected);
    for i in 0..predicate.len() {
        if predicate.is_valid(i) && predicate.value(i) {
            indices.append_value(i as u32);
        }
    }
    take(array, &indices.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_primitive;
    use crate::array::boolean::BooleanBuilder;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use std::sync::Arc;

    fn int_array(values: &[i64]) -> ArrayRef {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            b.append_value(v);
        }
        Arc::new(b.finish())
    }

    fn predicate(bits: &[Option<bool>]) -> BooleanArray {
        let mut b = BooleanBuilder::with_capacity(bits.len());
        for &bit in bits {
            match bit {
                Some(v) => b.append_value(v),
                None => b.append_null(),
            }
        }
        b.finish()
    }

    #[test]
    fn filter_keeps_only_true_rows() {
        let arr = int_array(&[10, 20, 30, 40]);
        let pred = predicate(&[Some(true), Some(false), Some(true), Some(false)]);
        let filtered = filter(arr.as_ref(), &pred).unwrap();
        let filtered = as_primitive::<Int64Type>(filtered.as_ref()).unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered.value(0), 10);
        assert_eq!(filtered.value(1), 30);
    }

    #[test]
    fn filter_rejects_null_predicate_rows_like_false() {
        let arr = int_array(&[10, 20, 30]);
        let pred = predicate(&[Some(true), None, Some(true)]);
        let filtered = filter(arr.as_ref(), &pred).unwrap();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filter_selecting_nothing_yields_empty_array() {
        let arr = int_array(&[10, 20]);
        let pred = predicate(&[Some(false), None]);
        let filtered = filter(arr.as_ref(), &pred).unwrap();
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn filter_selecting_everything_is_zero_copy() {
        let arr = int_array(&[10, 20]);
        let pred = predicate(&[Some(true), Some(true)]);
        let filtered = filter(arr.as_ref(), &pred).unwrap();
        assert_eq!(filtered.len(), 2);
        let filtered = as_primitive::<Int64Type>(filtered.as_ref()).unwrap();
        assert_eq!(filtered.value(0), 10);
        assert_eq!(filtered.value(1), 20);
    }

    #[test]
    fn filter_length_mismatch_errors() {
        let arr = int_array(&[10, 20]);
        let pred = predicate(&[Some(true)]);
        assert!(filter(arr.as_ref(), &pred).is_err());
    }
}
