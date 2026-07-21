//! Vectorized arithmetic kernels. See design-docs/basalt-phase2-lld.md §4.2.
//!
//! **Deviation from the LLD's "compute over null slots, then mask" pattern,
//! documented here because it's a real correctness call, not an oversight:**
//! that pattern is only safe for operations that can't trap. This project's
//! arithmetic (see Phase 1's `expr::eval`) uses *checked* arithmetic and
//! errors on overflow rather than wrapping — which is the right call for a
//! query engine (a silently wrapped `SUM` is a silently wrong answer). But
//! checked arithmetic over a null slot's *garbage* bytes can spuriously
//! trip a false "overflow" on a row that doesn't matter. So these kernels
//! skip computation entirely for null rows rather than computing-then-
//! masking. The branch this costs is exactly the "per-element `if
//! is_null` branch that defeats auto-vectorization" the LLD warns about —
//! a real, deliberate trade of some vectorization for not risking a bogus
//! error on don't-care data. Revisit once benchmarked.
//!
//! **Also simplified from the LLD's three-entry-point shape** (array⊕array,
//! array⊕scalar, scalar⊕array, keeping scalars unmaterialized). This pass
//! always materializes a scalar operand into a full array first via
//! `ColumnarValue::into_array`. Correct, and a fine first version; the
//! zero-materialization fast path is a benchmarked follow-up like the two
//! deviations above.

use std::sync::Arc;

use super::ColumnarValue;
use crate::array::array::{as_primitive, Array};
use crate::array::primitive::PrimitiveBuilder;
use crate::array::types::{Float64Type, Int64Type};
use crate::error::{BasaltError, Result};
use crate::scalar::ScalarValue;

pub fn add(lhs: &ColumnarValue, rhs: &ColumnarValue) -> Result<ColumnarValue> {
    binary_numeric(
        lhs,
        rhs,
        |a, b| a.checked_add(b).ok_or(BasaltError::NumericOverflow),
        |a, b| Ok(a + b),
    )
}

pub fn sub(lhs: &ColumnarValue, rhs: &ColumnarValue) -> Result<ColumnarValue> {
    binary_numeric(
        lhs,
        rhs,
        |a, b| a.checked_sub(b).ok_or(BasaltError::NumericOverflow),
        |a, b| Ok(a - b),
    )
}

pub fn mul(lhs: &ColumnarValue, rhs: &ColumnarValue) -> Result<ColumnarValue> {
    binary_numeric(
        lhs,
        rhs,
        |a, b| a.checked_mul(b).ok_or(BasaltError::NumericOverflow),
        |a, b| Ok(a * b),
    )
}

pub fn div(lhs: &ColumnarValue, rhs: &ColumnarValue) -> Result<ColumnarValue> {
    binary_numeric(
        lhs,
        rhs,
        |a, b| {
            if b == 0 {
                Err(BasaltError::DivisionByZero)
            } else {
                a.checked_div(b).ok_or(BasaltError::NumericOverflow)
            }
        },
        |a, b| {
            if b == 0.0 {
                Err(BasaltError::DivisionByZero)
            } else {
                Ok(a / b)
            }
        },
    )
}

pub fn rem(lhs: &ColumnarValue, rhs: &ColumnarValue) -> Result<ColumnarValue> {
    binary_numeric(
        lhs,
        rhs,
        |a, b| {
            if b == 0 {
                Err(BasaltError::DivisionByZero)
            } else {
                a.checked_rem(b).ok_or(BasaltError::NumericOverflow)
            }
        },
        |a, b| {
            if b == 0.0 {
                Err(BasaltError::DivisionByZero)
            } else {
                Ok(a % b)
            }
        },
    )
}

/// Shared machinery for the four arithmetic ops above: materializes both
/// sides to the same length, dispatches on whether the (already-coerced,
/// per Phase 1's binder) type is `Int64` or `Float64`, and applies the
/// per-element op with null-skipping.
fn binary_numeric(
    lhs: &ColumnarValue,
    rhs: &ColumnarValue,
    int_op: impl Fn(i64, i64) -> Result<i64>,
    float_op: impl Fn(f64, f64) -> Result<f64>,
) -> Result<ColumnarValue> {
    if let (ColumnarValue::Scalar(l), ColumnarValue::Scalar(r)) = (lhs, rhs) {
        return Ok(ColumnarValue::Scalar(scalar_numeric(
            l, r, &int_op, &float_op,
        )?));
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

    match (lhs_array.data_type(), rhs_array.data_type()) {
        (crate::types::data_type::DataType::Int64, crate::types::data_type::DataType::Int64) => {
            let l = as_primitive::<Int64Type>(lhs_array.as_ref())?;
            let r = as_primitive::<Int64Type>(rhs_array.as_ref())?;
            let mut builder = PrimitiveBuilder::<Int64Type>::with_capacity(num_rows);
            for i in 0..num_rows {
                if l.is_null(i) || r.is_null(i) {
                    builder.append_null();
                    continue;
                }
                builder.append_value(int_op(l.value(i), r.value(i))?);
            }
            Ok(ColumnarValue::Array(Arc::new(builder.finish())))
        }
        (
            crate::types::data_type::DataType::Float64,
            crate::types::data_type::DataType::Float64,
        ) => {
            let l = as_primitive::<Float64Type>(lhs_array.as_ref())?;
            let r = as_primitive::<Float64Type>(rhs_array.as_ref())?;
            let mut builder = PrimitiveBuilder::<Float64Type>::with_capacity(num_rows);
            for i in 0..num_rows {
                if l.is_null(i) || r.is_null(i) {
                    builder.append_null();
                    continue;
                }
                builder.append_value(float_op(l.value(i), r.value(i))?);
            }
            Ok(ColumnarValue::Array(Arc::new(builder.finish())))
        }
        (lt, rt) => Err(BasaltError::Type {
            message: format!("arithmetic requires matching numeric types, found {lt} and {rt}"),
        }),
    }
}

fn scalar_numeric(
    lhs: &ScalarValue,
    rhs: &ScalarValue,
    int_op: impl Fn(i64, i64) -> Result<i64>,
    float_op: impl Fn(f64, f64) -> Result<f64>,
) -> Result<ScalarValue> {
    match (lhs, rhs) {
        (ScalarValue::Int64(l), ScalarValue::Int64(r)) => match (l, r) {
            (Some(l), Some(r)) => Ok(ScalarValue::Int64(Some(int_op(*l, *r)?))),
            _ => Ok(ScalarValue::Int64(None)),
        },
        (ScalarValue::Float64(l), ScalarValue::Float64(r)) => match (l, r) {
            (Some(l), Some(r)) => Ok(ScalarValue::Float64(Some(float_op(*l, *r)?))),
            _ => Ok(ScalarValue::Float64(None)),
        },
        (lt, rt) => Err(BasaltError::Type {
            message: format!(
                "arithmetic requires matching numeric types, found {} and {}",
                lt.data_type(),
                rt.data_type()
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_primitive as downcast_primitive;
    use crate::array::primitive::{Float64Array, Int64Array};

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

    fn float_array(values: &[f64]) -> ColumnarValue {
        let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(values.len());
        for &v in values {
            b.append_value(v);
        }
        ColumnarValue::Array(Arc::new(b.finish()))
    }

    fn as_int64(cv: &ColumnarValue) -> &Int64Array {
        match cv {
            ColumnarValue::Array(a) => downcast_primitive::<Int64Type>(a.as_ref()).unwrap(),
            _ => panic!("expected array"),
        }
    }

    fn as_float64(cv: &ColumnarValue) -> &Float64Array {
        match cv {
            ColumnarValue::Array(a) => downcast_primitive::<Float64Type>(a.as_ref()).unwrap(),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn add_int_arrays_elementwise() {
        let lhs = int_array(&[Some(1), Some(2), Some(3)]);
        let rhs = int_array(&[Some(10), Some(20), Some(30)]);
        let result = add(&lhs, &rhs).unwrap();
        let result = as_int64(&result);
        assert_eq!(result.value(0), 11);
        assert_eq!(result.value(1), 22);
        assert_eq!(result.value(2), 33);
    }

    #[test]
    fn add_propagates_null_on_either_side() {
        let lhs = int_array(&[Some(1), None, Some(3)]);
        let rhs = int_array(&[Some(10), Some(20), None]);
        let result = add(&lhs, &rhs).unwrap();
        let result = as_int64(&result);
        assert!(!result.is_null(0));
        assert!(result.is_null(1));
        assert!(result.is_null(2));
    }

    #[test]
    fn add_int_array_and_scalar_keeps_scalar_applied_to_every_row() {
        let lhs = int_array(&[Some(1), Some(2), Some(3)]);
        let rhs = ColumnarValue::Scalar(ScalarValue::Int64(Some(100)));
        let result = add(&lhs, &rhs).unwrap();
        let result = as_int64(&result);
        assert_eq!(result.value(0), 101);
        assert_eq!(result.value(2), 103);
    }

    #[test]
    fn add_two_scalars_stays_a_scalar() {
        let lhs = ColumnarValue::Scalar(ScalarValue::Int64(Some(2)));
        let rhs = ColumnarValue::Scalar(ScalarValue::Int64(Some(3)));
        let result = add(&lhs, &rhs).unwrap();
        assert_eq!(result, ColumnarValue::Scalar(ScalarValue::Int64(Some(5))));
    }

    #[test]
    fn add_overflow_errors() {
        let lhs = int_array(&[Some(i64::MAX)]);
        let rhs = int_array(&[Some(1)]);
        assert!(matches!(
            add(&lhs, &rhs).unwrap_err(),
            BasaltError::NumericOverflow
        ));
    }

    #[test]
    fn overflow_on_a_null_row_does_not_error() {
        // The garbage value under a null slot must never surface as a
        // spurious overflow — this is exactly the deviation documented
        // at the top of this module.
        let lhs = int_array(&[None]);
        let rhs = int_array(&[Some(1)]);
        assert!(add(&lhs, &rhs).is_ok());
    }

    #[test]
    fn div_by_zero_errors_for_ints_and_floats() {
        let lhs = int_array(&[Some(10)]);
        let rhs = int_array(&[Some(0)]);
        assert!(matches!(
            div(&lhs, &rhs).unwrap_err(),
            BasaltError::DivisionByZero
        ));

        let lhs = float_array(&[10.0]);
        let rhs = float_array(&[0.0]);
        assert!(matches!(
            div(&lhs, &rhs).unwrap_err(),
            BasaltError::DivisionByZero
        ));
    }

    /// Regression test: division by zero and integer overflow are distinct
    /// error variants and must not collapse into each other. A prior
    /// version mapped both `i64::checked_div` misses to `NumericOverflow`
    /// by using `Option` for the zero-divisor case too, silently turning
    /// `10 / 0` into an overflow error instead of a division-by-zero one.
    #[test]
    fn div_by_zero_is_distinct_from_overflow() {
        let lhs = int_array(&[Some(10)]);
        let rhs = int_array(&[Some(0)]);
        let err = div(&lhs, &rhs).unwrap_err();
        assert!(
            matches!(err, BasaltError::DivisionByZero),
            "expected DivisionByZero, got {err:?}"
        );

        let lhs = int_array(&[Some(i64::MIN)]);
        let rhs = int_array(&[Some(-1)]); // MIN / -1 overflows, doesn't divide by zero
        let err = div(&lhs, &rhs).unwrap_err();
        assert!(
            matches!(err, BasaltError::NumericOverflow),
            "expected NumericOverflow, got {err:?}"
        );
    }

    #[test]
    fn rem_by_zero_errors() {
        let lhs = int_array(&[Some(10)]);
        let rhs = int_array(&[Some(0)]);
        assert!(matches!(
            rem(&lhs, &rhs).unwrap_err(),
            BasaltError::DivisionByZero
        ));
    }

    #[test]
    fn float_arithmetic_elementwise() {
        let lhs = float_array(&[1.5, 2.5]);
        let rhs = float_array(&[0.5, 0.5]);
        let result = mul(&lhs, &rhs).unwrap();
        let result = as_float64(&result);
        assert_eq!(result.value(0), 0.75);
        assert_eq!(result.value(1), 1.25);
    }

    #[test]
    fn mismatched_types_error_instead_of_panicking() {
        let lhs = int_array(&[Some(1)]);
        let rhs = float_array(&[1.0]);
        assert!(add(&lhs, &rhs).is_err());
    }
}
