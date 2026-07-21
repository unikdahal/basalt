//! Vectorized comparison kernels, producing `BooleanArray`. See
//! design-docs/basalt-phase2-lld.md §4.2.
//!
//! Requires both operands to already share the same concrete type — exactly
//! Phase 1's rule carried forward: coercion is a bind-time decision
//! materialized as explicit `Cast` nodes (see the Phase 1 LLD §5.3), so a
//! compute kernel never performs implicit conversion. An `Int64` vs
//! `Float64` comparison reaching here uncast is a binder bug, and this
//! module reports it as a type error rather than guessing.
//!
//! **Array⊕scalar and scalar⊕array take a dedicated fast path for `Int64`
//! and `Float64`** that never materializes the scalar into a full array and
//! never re-derives `values()`'s slice per element — the exact two bugs
//! `compute::arith` had (see that module's doc comment for the full story).
//! They were found here via `benches/tpch.rs`: a filter with several scalar
//! comparisons chained by `AND` ran *slower* on Phase 2 than Phase 1's
//! row-at-a-time interpreter, tracing back to this module still doing what
//! `arith.rs` used to. `Utf8`/`Boolean` comparisons keep the general,
//! materializing path below.

use std::cmp::Ordering;
use std::sync::Arc;

use super::ColumnarValue;
use crate::array::array::{as_primitive, as_string, Array};
use crate::array::boolean::BooleanBuilder;
use crate::array::types::{Float64Type, Int64Type};
use crate::error::{BasaltError, Result};
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;

pub fn eq(lhs: &ColumnarValue, rhs: &ColumnarValue) -> Result<ColumnarValue> {
    compare(lhs, rhs, |o| o == std::cmp::Ordering::Equal, ScalarEq::Eq)
}

pub fn neq(lhs: &ColumnarValue, rhs: &ColumnarValue) -> Result<ColumnarValue> {
    compare(
        lhs,
        rhs,
        |o| o != std::cmp::Ordering::Equal,
        ScalarEq::NotEq,
    )
}

pub fn lt(lhs: &ColumnarValue, rhs: &ColumnarValue) -> Result<ColumnarValue> {
    compare(lhs, rhs, |o| o == std::cmp::Ordering::Less, ScalarEq::Lt)
}

pub fn lteq(lhs: &ColumnarValue, rhs: &ColumnarValue) -> Result<ColumnarValue> {
    compare(
        lhs,
        rhs,
        |o| o != std::cmp::Ordering::Greater,
        ScalarEq::LtEq,
    )
}

pub fn gt(lhs: &ColumnarValue, rhs: &ColumnarValue) -> Result<ColumnarValue> {
    compare(lhs, rhs, |o| o == std::cmp::Ordering::Greater, ScalarEq::Gt)
}

pub fn gteq(lhs: &ColumnarValue, rhs: &ColumnarValue) -> Result<ColumnarValue> {
    compare(lhs, rhs, |o| o != std::cmp::Ordering::Less, ScalarEq::GtEq)
}

/// Which comparison a scalar/scalar fallback is performing — needed because
/// `Ordering`-based dispatch doesn't apply to `Boolean`/`Utf8` scalar pairs
/// the same way `PartialOrd` does; kept as a tiny enum rather than six
/// near-duplicate scalar functions.
#[derive(Clone, Copy)]
enum ScalarEq {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

fn compare(
    lhs: &ColumnarValue,
    rhs: &ColumnarValue,
    ord_matches: impl Fn(std::cmp::Ordering) -> bool,
    which: ScalarEq,
) -> Result<ColumnarValue> {
    match (lhs, rhs) {
        (ColumnarValue::Scalar(l), ColumnarValue::Scalar(r)) => {
            Ok(ColumnarValue::Scalar(scalar_compare(l, r, which)?))
        }
        (ColumnarValue::Array(l), ColumnarValue::Scalar(r)) => {
            if let Some(result) = array_scalar_compare(l.as_ref(), r, &ord_matches, false)? {
                return Ok(result);
            }
            compare_general(lhs, rhs, ord_matches)
        }
        (ColumnarValue::Scalar(l), ColumnarValue::Array(r)) => {
            if let Some(result) = array_scalar_compare(r.as_ref(), l, &ord_matches, true)? {
                return Ok(result);
            }
            compare_general(lhs, rhs, ord_matches)
        }
        (ColumnarValue::Array(_), ColumnarValue::Array(_)) => {
            compare_general(lhs, rhs, ord_matches)
        }
    }
}

/// Dedicated `Int64`/`Float64` array⊕scalar (or scalar⊕array, via
/// `scalar_on_left`) fast path: never materializes the scalar into a full
/// array via `into_array`, and hoists `values()` once instead of calling
/// `value(i)` per element — the same two bugs `compute::arith` had before
/// its own fast path (see that module's doc comment). Found here via
/// `benches/tpch.rs`: this module still did both, and a filter with several
/// scalar comparisons chained by `AND` ran *slower* on Phase 2 than on
/// Phase 1's row-at-a-time interpreter before this fix. Returns `Ok(None)`
/// for any type this fast path doesn't cover (`Utf8`, `Boolean`, or a null
/// scalar with a type mismatch), falling back to the general path.
fn array_scalar_compare(
    array: &dyn Array,
    scalar: &ScalarValue,
    ord_matches: &impl Fn(Ordering) -> bool,
    scalar_on_left: bool,
) -> Result<Option<ColumnarValue>> {
    let num_rows = array.len();
    match (array.data_type(), scalar) {
        (DataType::Int64, ScalarValue::Int64(s)) => {
            let a = as_primitive::<Int64Type>(array)?;
            let mut builder = BooleanBuilder::with_capacity(num_rows);
            let Some(s) = *s else {
                for _ in 0..num_rows {
                    builder.append_null();
                }
                return Ok(Some(ColumnarValue::Array(Arc::new(builder.finish()))));
            };
            let values = a.values();
            for (i, &v) in values.iter().enumerate() {
                if a.is_null(i) {
                    builder.append_null();
                    continue;
                }
                let ord = if scalar_on_left { s.cmp(&v) } else { v.cmp(&s) };
                builder.append_value(ord_matches(ord));
            }
            Ok(Some(ColumnarValue::Array(Arc::new(builder.finish()))))
        }
        (DataType::Float64, ScalarValue::Float64(s)) => {
            let a = as_primitive::<Float64Type>(array)?;
            let mut builder = BooleanBuilder::with_capacity(num_rows);
            let Some(s) = *s else {
                for _ in 0..num_rows {
                    builder.append_null();
                }
                return Ok(Some(ColumnarValue::Array(Arc::new(builder.finish()))));
            };
            let values = a.values();
            for (i, &v) in values.iter().enumerate() {
                if a.is_null(i) {
                    builder.append_null();
                    continue;
                }
                let ord = if scalar_on_left {
                    s.partial_cmp(&v)
                } else {
                    v.partial_cmp(&s)
                };
                match ord {
                    // NaN compares false to everything, per IEEE-754 — no
                    // special-casing here, matching Phase 1's eval.
                    Some(o) => builder.append_value(ord_matches(o)),
                    None => builder.append_value(false),
                }
            }
            Ok(Some(ColumnarValue::Array(Arc::new(builder.finish()))))
        }
        _ => Ok(None),
    }
}

/// General path: materializes both sides via `into_array` (a no-op when
/// both are already arrays) and hoists `values()`/bitmap access once before
/// each per-element loop. Used for `Utf8`/`Boolean` scalar comparisons and
/// all array⊕array comparisons.
fn compare_general(
    lhs: &ColumnarValue,
    rhs: &ColumnarValue,
    ord_matches: impl Fn(std::cmp::Ordering) -> bool,
) -> Result<ColumnarValue> {
    let num_rows = match (lhs, rhs) {
        (ColumnarValue::Array(a), _) => a.len(),
        (_, ColumnarValue::Array(a)) => a.len(),
        _ => unreachable!("both-scalar case handled by `compare`"),
    };
    let lhs_array = lhs.clone().into_array(num_rows)?;
    let rhs_array = rhs.clone().into_array(num_rows)?;
    if lhs_array.len() != rhs_array.len() {
        return Err(BasaltError::Internal(format!(
            "operand length mismatch: {} vs {}",
            lhs_array.len(),
            rhs_array.len()
        )));
    }
    let (lt, rt) = (lhs_array.data_type(), rhs_array.data_type());
    if lt != rt {
        return Err(BasaltError::Type {
            message: format!("comparison requires matching types, found {lt} and {rt}"),
        });
    }

    let mut builder = BooleanBuilder::with_capacity(num_rows);
    match lt {
        DataType::Int64 => {
            let l = as_primitive::<Int64Type>(lhs_array.as_ref())?;
            let r = as_primitive::<Int64Type>(rhs_array.as_ref())?;
            let (l_values, r_values) = (l.values(), r.values());
            for i in 0..num_rows {
                if l.is_null(i) || r.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(ord_matches(l_values[i].cmp(&r_values[i])));
                }
            }
        }
        DataType::Float64 => {
            let l = as_primitive::<Float64Type>(lhs_array.as_ref())?;
            let r = as_primitive::<Float64Type>(rhs_array.as_ref())?;
            let (l_values, r_values) = (l.values(), r.values());
            for i in 0..num_rows {
                if l.is_null(i) || r.is_null(i) {
                    builder.append_null();
                    continue;
                }
                match l_values[i].partial_cmp(&r_values[i]) {
                    // NaN compares false to everything, per IEEE-754 — no
                    // special-casing here, matching Phase 1's eval.
                    Some(o) => builder.append_value(ord_matches(o)),
                    None => builder.append_value(false),
                }
            }
        }
        DataType::Utf8 => {
            let l = as_string(lhs_array.as_ref())?;
            let r = as_string(rhs_array.as_ref())?;
            for i in 0..num_rows {
                if l.is_null(i) || r.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(ord_matches(l.value(i).cmp(r.value(i))));
                }
            }
        }
        DataType::Boolean => {
            let l = crate::array::array::as_boolean(lhs_array.as_ref())?;
            let r = crate::array::array::as_boolean(rhs_array.as_ref())?;
            let (l_bytes, l_offset) = (l.values().as_bytes(), l.values().bit_offset());
            let (r_bytes, r_offset) = (r.values().as_bytes(), r.values().bit_offset());
            for i in 0..num_rows {
                if l.is_null(i) || r.is_null(i) {
                    builder.append_null();
                } else {
                    let ord = crate::buffer::bit_at(l_bytes, l_offset, i)
                        .cmp(&crate::buffer::bit_at(r_bytes, r_offset, i));
                    builder.append_value(ord_matches(ord));
                }
            }
        }
    }
    Ok(ColumnarValue::Array(Arc::new(builder.finish())))
}

fn scalar_compare(lhs: &ScalarValue, rhs: &ScalarValue, which: ScalarEq) -> Result<ScalarValue> {
    fn apply(o: std::cmp::Ordering, which: ScalarEq) -> bool {
        use std::cmp::Ordering::*;
        match which {
            ScalarEq::Eq => o == Equal,
            ScalarEq::NotEq => o != Equal,
            ScalarEq::Lt => o == Less,
            ScalarEq::LtEq => o != Greater,
            ScalarEq::Gt => o == Greater,
            ScalarEq::GtEq => o != Less,
        }
    }

    let result = match (lhs, rhs) {
        (ScalarValue::Int64(Some(l)), ScalarValue::Int64(Some(r))) => Some(apply(l.cmp(r), which)),
        (ScalarValue::Int64(_), ScalarValue::Int64(_)) => None,
        (ScalarValue::Float64(Some(l)), ScalarValue::Float64(Some(r))) => {
            l.partial_cmp(r).map(|o| apply(o, which)).or(Some(false))
        }
        (ScalarValue::Float64(_), ScalarValue::Float64(_)) => None,
        (ScalarValue::Utf8(Some(l)), ScalarValue::Utf8(Some(r))) => Some(apply(l.cmp(r), which)),
        (ScalarValue::Utf8(_), ScalarValue::Utf8(_)) => None,
        (ScalarValue::Boolean(Some(l)), ScalarValue::Boolean(Some(r))) => {
            Some(apply(l.cmp(r), which))
        }
        (ScalarValue::Boolean(_), ScalarValue::Boolean(_)) => None,
        (lt, rt) => {
            return Err(BasaltError::Type {
                message: format!(
                    "comparison requires matching types, found {} and {}",
                    lt.data_type(),
                    rt.data_type()
                ),
            });
        }
    };
    Ok(ScalarValue::Boolean(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_boolean;
    use crate::array::primitive::PrimitiveBuilder;

    fn int_array(values: &[Option<i64>]) -> ColumnarValue {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            match v {
                Some(v) => b.append_value(v),
                None => b.append_null(),
            }
        }
        ColumnarValue::Array(Arc::new(b.finish()))
    }

    fn as_bool_array(cv: &ColumnarValue) -> &crate::array::boolean::BooleanArray {
        match cv {
            ColumnarValue::Array(a) => as_boolean(a.as_ref()).unwrap(),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn eq_elementwise() {
        let lhs = int_array(&[Some(1), Some(2), Some(3)]);
        let rhs = int_array(&[Some(1), Some(9), Some(3)]);
        let result = eq(&lhs, &rhs).unwrap();
        let result = as_bool_array(&result);
        assert!(result.value(0));
        assert!(!result.value(1));
        assert!(result.value(2));
    }

    #[test]
    fn array_scalar_fast_path_matches_array_array_result() {
        let lhs = int_array(&[Some(1), Some(5), Some(9), None]);
        let rhs_array = int_array(&[Some(5), Some(5), Some(5), Some(5)]);
        let rhs_scalar = ColumnarValue::Scalar(ScalarValue::Int64(Some(5)));

        for op in [lt, lteq, gt, gteq, eq, neq] {
            let via_array = op(&lhs, &rhs_array).unwrap();
            let via_scalar = op(&lhs, &rhs_scalar).unwrap();
            assert_eq!(
                as_bool_array(&via_array).len(),
                as_bool_array(&via_scalar).len()
            );
            for i in 0..4 {
                assert_eq!(
                    as_bool_array(&via_array).is_null(i),
                    as_bool_array(&via_scalar).is_null(i)
                );
                if !as_bool_array(&via_array).is_null(i) {
                    assert_eq!(
                        as_bool_array(&via_array).value(i),
                        as_bool_array(&via_scalar).value(i)
                    );
                }
            }
        }
    }

    #[test]
    fn scalar_lt_array_preserves_operand_order() {
        // Regression test for the array⊕scalar fast path: `<`/`>` aren't
        // symmetric, so `5 < col` must not silently become `col < 5`.
        let lhs = ColumnarValue::Scalar(ScalarValue::Int64(Some(5)));
        let rhs = int_array(&[Some(1), Some(10)]);
        let result = lt(&lhs, &rhs).unwrap();
        let result = as_bool_array(&result);
        assert!(!result.value(0)); // 5 < 1 -> false
        assert!(result.value(1)); // 5 < 10 -> true
    }

    #[test]
    fn array_eq_null_scalar_is_all_null() {
        let lhs = int_array(&[Some(1), Some(2)]);
        let rhs = ColumnarValue::Scalar(ScalarValue::Int64(None));
        let result = eq(&lhs, &rhs).unwrap();
        let result = as_bool_array(&result);
        assert!(result.is_null(0));
        assert!(result.is_null(1));
    }

    #[test]
    fn float_array_scalar_fast_path_handles_nan() {
        let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(2);
        b.append_value(f64::NAN);
        b.append_value(1.0);
        let lhs = ColumnarValue::Array(Arc::new(b.finish()));
        let rhs = ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0)));
        let result = eq(&lhs, &rhs).unwrap();
        let result = as_bool_array(&result);
        assert!(!result.value(0));
        assert!(result.value(1));
    }

    #[test]
    fn null_eq_null_is_null_not_true() {
        let lhs = int_array(&[None]);
        let rhs = int_array(&[None]);
        let result = eq(&lhs, &rhs).unwrap();
        let result = as_bool_array(&result);
        assert!(result.is_null(0));
    }

    #[test]
    fn nan_is_never_equal_or_ordered() {
        let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(1);
        b.append_value(f64::NAN);
        let lhs = ColumnarValue::Array(Arc::new(b.finish()));
        let mut b2 = PrimitiveBuilder::<Float64Type>::with_capacity(1);
        b2.append_value(f64::NAN);
        let rhs = ColumnarValue::Array(Arc::new(b2.finish()));

        let result = eq(&lhs, &rhs).unwrap();
        assert!(!as_bool_array(&result).value(0));
        let result = lt(&lhs, &rhs).unwrap();
        assert!(!as_bool_array(&result).value(0));
    }

    #[test]
    fn mismatched_types_error() {
        let lhs = int_array(&[Some(1)]);
        let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(1);
        b.append_value(1.0);
        let rhs = ColumnarValue::Array(Arc::new(b.finish()));
        assert!(eq(&lhs, &rhs).is_err());
    }

    #[test]
    fn scalar_scalar_comparison_stays_scalar() {
        let lhs = ColumnarValue::Scalar(ScalarValue::Int64(Some(5)));
        let rhs = ColumnarValue::Scalar(ScalarValue::Int64(Some(3)));
        assert_eq!(
            gt(&lhs, &rhs).unwrap(),
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true)))
        );
    }

    #[test]
    fn string_comparison_is_lexicographic() {
        let mut lb = crate::array::string::StringBuilder::with_capacity(2, 8);
        lb.append_value("apple").unwrap();
        lb.append_value("zebra").unwrap();
        let lhs = ColumnarValue::Array(Arc::new(lb.finish()));

        let mut rb = crate::array::string::StringBuilder::with_capacity(2, 8);
        rb.append_value("banana").unwrap();
        rb.append_value("apple").unwrap();
        let rhs = ColumnarValue::Array(Arc::new(rb.finish()));

        let result = lt(&lhs, &rhs).unwrap();
        let result = as_bool_array(&result);
        assert!(result.value(0));
        assert!(!result.value(1));
    }
}
