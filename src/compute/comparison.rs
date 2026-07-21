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
//! Like `arith.rs`, this pass always materializes scalar operands via
//! `ColumnarValue::into_array` rather than the LLD's three-entry-point
//! shape; see that module's doc comment for the reasoning.

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
    if let (ColumnarValue::Scalar(l), ColumnarValue::Scalar(r)) = (lhs, rhs) {
        return Ok(ColumnarValue::Scalar(scalar_compare(l, r, which)?));
    }

    let num_rows = match (lhs, rhs) {
        (ColumnarValue::Array(a), _) => a.len(),
        (_, ColumnarValue::Array(a)) => a.len(),
        _ => unreachable!("both-scalar case handled above"),
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
            for i in 0..num_rows {
                if l.is_null(i) || r.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(ord_matches(l.value(i).cmp(&r.value(i))));
                }
            }
        }
        DataType::Float64 => {
            let l = as_primitive::<Float64Type>(lhs_array.as_ref())?;
            let r = as_primitive::<Float64Type>(rhs_array.as_ref())?;
            for i in 0..num_rows {
                if l.is_null(i) || r.is_null(i) {
                    builder.append_null();
                    continue;
                }
                match l.value(i).partial_cmp(&r.value(i)) {
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
            for i in 0..num_rows {
                if l.is_null(i) || r.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(ord_matches(l.value(i).cmp(&r.value(i))));
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
