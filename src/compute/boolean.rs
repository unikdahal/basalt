//! Three-valued (Kleene) logical kernels on `BooleanArray`. See
//! design-docs/basalt-phase2-lld.md §4.4.
//!
//! `AND`/`OR` cannot simply intersect validity bitmaps, because a decisive
//! operand overrides an unknown one: `false AND NULL` is `false`, and
//! `true OR NULL` is `true` (Phase 1's truth tables, carried forward
//! unchanged — see the Phase 1 LLD §5.5). This is implemented per-row here,
//! not via the bitwise-word formulas in the LLD §4.4 — same reasoning as
//! `Bitmap::and`/`or`: correctness first at a bug-dense truth table, with
//! the six-bitwise-op fast path a benchmarked follow-up.
//!
//! `and_kleene`/`or_kleene` are named to distinguish them from a plain
//! bitwise `and`/`or` that would just intersect validity — that version is
//! correct and faster only when the caller already knows neither side can
//! be null, which this module doesn't assume.

use crate::array::array::Array;
use crate::array::boolean::{BooleanArray, BooleanBuilder};
use crate::error::{BasaltError, Result};

pub fn and_kleene(lhs: &BooleanArray, rhs: &BooleanArray) -> Result<BooleanArray> {
    zip_kleene(lhs, rhs, |l, r| match (l, r) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    })
}

pub fn or_kleene(lhs: &BooleanArray, rhs: &BooleanArray) -> Result<BooleanArray> {
    zip_kleene(lhs, rhs, |l, r| match (l, r) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    })
}

/// `NOT NULL` is `NULL` — negation never turns unknown into known.
pub fn not_kleene(array: &BooleanArray) -> BooleanArray {
    let mut builder = BooleanBuilder::with_capacity(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null();
        } else {
            builder.append_value(!array.value(i));
        }
    }
    builder.finish()
}

fn zip_kleene(
    lhs: &BooleanArray,
    rhs: &BooleanArray,
    f: impl Fn(Option<bool>, Option<bool>) -> Option<bool>,
) -> Result<BooleanArray> {
    if lhs.len() != rhs.len() {
        return Err(BasaltError::Internal(format!(
            "boolean array length mismatch: {} vs {}",
            lhs.len(),
            rhs.len()
        )));
    }
    let mut builder = BooleanBuilder::with_capacity(lhs.len());
    for i in 0..lhs.len() {
        let l = if lhs.is_null(i) {
            None
        } else {
            Some(lhs.value(i))
        };
        let r = if rhs.is_null(i) {
            None
        } else {
            Some(rhs.value(i))
        };
        match f(l, r) {
            Some(v) => builder.append_value(v),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bools(vals: &[Option<bool>]) -> BooleanArray {
        let mut b = BooleanBuilder::with_capacity(vals.len());
        for &v in vals {
            match v {
                Some(v) => b.append_value(v),
                None => b.append_null(),
            }
        }
        b.finish()
    }

    fn get(arr: &BooleanArray, i: usize) -> Option<bool> {
        if arr.is_null(i) {
            None
        } else {
            Some(arr.value(i))
        }
    }

    #[test]
    fn and_kleene_truth_table() {
        // (true,true)->true, (true,false)->false, (false,false)->false,
        // (true,NULL)->NULL, (false,NULL)->false, (NULL,NULL)->NULL
        let lhs = bools(&[
            Some(true),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            None,
        ]);
        let rhs = bools(&[Some(true), Some(false), Some(false), None, None, None]);
        let result = and_kleene(&lhs, &rhs).unwrap();
        assert_eq!(get(&result, 0), Some(true));
        assert_eq!(get(&result, 1), Some(false));
        assert_eq!(get(&result, 2), Some(false));
        assert_eq!(get(&result, 3), None);
        assert_eq!(get(&result, 4), Some(false));
        assert_eq!(get(&result, 5), None);
    }

    #[test]
    fn or_kleene_truth_table() {
        // (true,true)->true, (true,false)->true, (false,false)->false,
        // (false,NULL)->NULL, (true,NULL)->true, (NULL,NULL)->NULL
        let lhs = bools(&[
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            None,
        ]);
        let rhs = bools(&[Some(true), Some(false), Some(false), None, None, None]);
        let result = or_kleene(&lhs, &rhs).unwrap();
        assert_eq!(get(&result, 0), Some(true));
        assert_eq!(get(&result, 1), Some(true));
        assert_eq!(get(&result, 2), Some(false));
        assert_eq!(get(&result, 3), None);
        assert_eq!(get(&result, 4), Some(true));
        assert_eq!(get(&result, 5), None);
    }

    #[test]
    fn not_kleene_negates_and_preserves_null() {
        let arr = bools(&[Some(true), Some(false), None]);
        let result = not_kleene(&arr);
        assert_eq!(get(&result, 0), Some(false));
        assert_eq!(get(&result, 1), Some(true));
        assert_eq!(get(&result, 2), None);
    }

    #[test]
    fn length_mismatch_errors() {
        let lhs = bools(&[Some(true)]);
        let rhs = bools(&[Some(true), Some(false)]);
        assert!(and_kleene(&lhs, &rhs).is_err());
    }
}
