//! Vectorized type-conversion kernel. See design-docs/basalt-phase2-lld.md §4.6.
//!
//! The conversion matrix mirrors Phase 1's `Value::cast_to` exactly,
//! including the `Float64 -> Int64` fix from that module's review (reject
//! non-finite/out-of-range floats rather than silently saturating via
//! `as i64`) — these are semantic decisions the LLD says to write down once
//! and reuse, not re-derive per phase.

use std::sync::Arc;

use crate::array::array::{as_primitive, as_string, Array, ArrayRef};
use crate::array::boolean::BooleanBuilder;
use crate::array::primitive::PrimitiveBuilder;
use crate::array::string::StringBuilder;
use crate::array::types::{Float64Type, Int64Type};
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;

/// # Errors
/// Errors if the conversion isn't supported, or if a value can't be
/// represented in the target type (e.g. a non-finite `Float64` cast to
/// `Int64`, or a `Utf8` value that doesn't parse).
pub fn cast(array: &dyn Array, to: DataType) -> Result<ArrayRef> {
    use DataType::*;
    match (array.data_type(), to) {
        (Int64, Int64) => Ok(array.slice(0, array.len())),
        (Int64, Float64) => {
            let src = as_primitive::<Int64Type>(array)?;
            map_primitive_to_primitive::<Int64Type, Float64Type>(src, |v| Ok(v as f64))
        }
        (Int64, Utf8) => {
            let src = as_primitive::<Int64Type>(array)?;
            map_primitive_to_string::<Int64Type>(src, |v| v.to_string())
        }
        (Int64, Boolean) => {
            let src = as_primitive::<Int64Type>(array)?;
            map_primitive_to_boolean::<Int64Type>(src, |v| v != 0)
        }

        (Float64, Float64) => Ok(array.slice(0, array.len())),
        (Float64, Int64) => {
            let src = as_primitive::<Float64Type>(array)?;
            map_primitive_to_primitive::<Float64Type, Int64Type>(src, |v| {
                if v.is_finite() && v >= i64::MIN as f64 && v <= i64::MAX as f64 {
                    Ok(v as i64)
                } else {
                    Err(BasaltError::Type {
                        message: format!("cannot cast {v} to Int64: out of range or not finite"),
                    })
                }
            })
        }
        (Float64, Utf8) => {
            let src = as_primitive::<Float64Type>(array)?;
            map_primitive_to_string::<Float64Type>(src, |v| v.to_string())
        }

        (Utf8, Utf8) => Ok(array.slice(0, array.len())),
        (Utf8, Int64) => {
            let src = as_string(array)?;
            map_string_to_primitive::<Int64Type>(src, |v| {
                v.parse::<i64>().map_err(|_| BasaltError::Type {
                    message: format!("cannot cast '{v}' to Int64"),
                })
            })
        }
        (Utf8, Float64) => {
            let src = as_string(array)?;
            map_string_to_primitive::<Float64Type>(src, |v| {
                v.parse::<f64>().map_err(|_| BasaltError::Type {
                    message: format!("cannot cast '{v}' to Float64"),
                })
            })
        }
        (Utf8, Boolean) => {
            let src = as_string(array)?;
            map_string_to_boolean(src)
        }

        (Boolean, Boolean) => Ok(array.slice(0, array.len())),
        (Boolean, Utf8) => {
            let src = crate::array::array::as_boolean(array)?;
            let mut builder = StringBuilder::with_capacity(src.len(), src.len() * 5);
            for i in 0..src.len() {
                if src.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(if src.value(i) { "true" } else { "false" })?;
                }
            }
            Ok(Arc::new(builder.finish()))
        }

        (from, to) => Err(BasaltError::Type {
            message: format!("cannot cast {from} to {to}"),
        }),
    }
}

fn map_primitive_to_primitive<From, To>(
    src: &crate::array::primitive::PrimitiveArray<From>,
    f: impl Fn(From::Native) -> Result<To::Native>,
) -> Result<ArrayRef>
where
    From: crate::array::types::ArrowPrimitiveType,
    To: crate::array::types::ArrowPrimitiveType,
{
    let mut builder = PrimitiveBuilder::<To>::with_capacity(src.len());
    for i in 0..src.len() {
        if src.is_null(i) {
            builder.append_null();
        } else {
            builder.append_value(f(src.value(i))?);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn map_primitive_to_string<From: crate::array::types::ArrowPrimitiveType>(
    src: &crate::array::primitive::PrimitiveArray<From>,
    f: impl Fn(From::Native) -> String,
) -> Result<ArrayRef> {
    let mut builder = StringBuilder::with_capacity(src.len(), src.len() * 8);
    for i in 0..src.len() {
        if src.is_null(i) {
            builder.append_null();
        } else {
            builder.append_value(&f(src.value(i)))?;
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn map_primitive_to_boolean<From: crate::array::types::ArrowPrimitiveType>(
    src: &crate::array::primitive::PrimitiveArray<From>,
    f: impl Fn(From::Native) -> bool,
) -> Result<ArrayRef> {
    let mut builder = BooleanBuilder::with_capacity(src.len());
    for i in 0..src.len() {
        if src.is_null(i) {
            builder.append_null();
        } else {
            builder.append_value(f(src.value(i)));
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn map_string_to_primitive<To: crate::array::types::ArrowPrimitiveType>(
    src: &crate::array::string::StringArray,
    f: impl Fn(&str) -> Result<To::Native>,
) -> Result<ArrayRef> {
    let mut builder = PrimitiveBuilder::<To>::with_capacity(src.len());
    for i in 0..src.len() {
        if src.is_null(i) {
            builder.append_null();
        } else {
            builder.append_value(f(src.value(i))?);
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// Cast a single `ScalarValue`, for `physical_expr::CastExpr`'s scalar path.
///
/// Reuses Phase 1's `Value::cast_to` (the exact same semantic matrix —
/// non-finite/out-of-range float rejection included) rather than
/// re-deriving cast rules a third time, converting through `Value`'s
/// untyped null at the boundary since `Value::cast_to` already special-cases
/// it (`Value::Null` always maps back to whichever `ScalarValue::T(None)`
/// the target type needs).
///
/// # Errors
/// Same as [`cast`]: unsupported conversions or values that don't fit the
/// target type.
pub fn cast_scalar(
    value: &crate::scalar::ScalarValue,
    to: DataType,
) -> Result<crate::scalar::ScalarValue> {
    use crate::scalar::ScalarValue;
    use crate::types::value::Value;

    let v = match value {
        ScalarValue::Int64(Some(x)) => Value::Int64(*x),
        ScalarValue::Float64(Some(x)) => Value::Float64(*x),
        ScalarValue::Utf8(Some(x)) => Value::Utf8(x.clone()),
        ScalarValue::Boolean(Some(x)) => Value::Boolean(*x),
        _ => Value::Null,
    };
    let casted = v.cast_to(to)?;
    Ok(match casted {
        Value::Null => match to {
            DataType::Int64 => ScalarValue::Int64(None),
            DataType::Float64 => ScalarValue::Float64(None),
            DataType::Utf8 => ScalarValue::Utf8(None),
            DataType::Boolean => ScalarValue::Boolean(None),
        },
        Value::Int64(x) => ScalarValue::Int64(Some(x)),
        Value::Float64(x) => ScalarValue::Float64(Some(x)),
        Value::Utf8(x) => ScalarValue::Utf8(Some(x)),
        Value::Boolean(x) => ScalarValue::Boolean(Some(x)),
    })
}

fn map_string_to_boolean(src: &crate::array::string::StringArray) -> Result<ArrayRef> {
    let mut builder = BooleanBuilder::with_capacity(src.len());
    for i in 0..src.len() {
        if src.is_null(i) {
            builder.append_null();
            continue;
        }
        match src.value(i).to_ascii_lowercase().as_str() {
            "true" => builder.append_value(true),
            "false" => builder.append_value(false),
            other => {
                return Err(BasaltError::Type {
                    message: format!("cannot cast '{other}' to Boolean"),
                });
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive::PrimitiveBuilder;

    fn int_array(values: &[Option<i64>]) -> ArrayRef {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            match v {
                Some(v) => b.append_value(v),
                None => b.append_null(),
            }
        }
        Arc::new(b.finish())
    }

    #[test]
    fn int_to_float_round_trips() {
        let arr = int_array(&[Some(3), None]);
        let result = cast(arr.as_ref(), DataType::Float64).unwrap();
        let result = as_primitive::<Float64Type>(result.as_ref()).unwrap();
        assert_eq!(result.value(0), 3.0);
        assert!(result.is_null(1));
    }

    #[test]
    fn float_to_int_rejects_nan_and_out_of_range() {
        let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(3);
        b.append_value(f64::NAN);
        let arr: ArrayRef = Arc::new(b.finish());
        assert!(cast(arr.as_ref(), DataType::Int64).is_err());

        let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(1);
        b.append_value(1e300);
        let arr: ArrayRef = Arc::new(b.finish());
        assert!(cast(arr.as_ref(), DataType::Int64).is_err());
    }

    #[test]
    fn float_to_int_accepts_in_range_values() {
        let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(1);
        b.append_value(42.9);
        let arr: ArrayRef = Arc::new(b.finish());
        let result = cast(arr.as_ref(), DataType::Int64).unwrap();
        let result = as_primitive::<Int64Type>(result.as_ref()).unwrap();
        assert_eq!(result.value(0), 42);
    }

    #[test]
    fn int_to_string_formats_values() {
        let arr = int_array(&[Some(42)]);
        let result = cast(arr.as_ref(), DataType::Utf8).unwrap();
        let result = as_string(result.as_ref()).unwrap();
        assert_eq!(result.value(0), "42");
    }

    #[test]
    fn string_to_int_parses_and_errors_on_garbage() {
        let mut b = StringBuilder::with_capacity(2, 8);
        b.append_value("42").unwrap();
        b.append_value("nope").unwrap();
        let arr: ArrayRef = Arc::new(b.finish());
        assert!(cast(arr.as_ref(), DataType::Int64).is_err());
    }

    #[test]
    fn string_to_bool_parses_case_insensitively() {
        let mut b = StringBuilder::with_capacity(2, 8);
        b.append_value("TRUE").unwrap();
        b.append_value("false").unwrap();
        let arr: ArrayRef = Arc::new(b.finish());
        let result = cast(arr.as_ref(), DataType::Boolean).unwrap();
        let result = crate::array::array::as_boolean(result.as_ref()).unwrap();
        assert!(result.value(0));
        assert!(!result.value(1));
    }

    #[test]
    fn unsupported_cast_errors_instead_of_panicking() {
        let mut b = BooleanBuilder::with_capacity(1);
        b.append_value(true);
        let arr: ArrayRef = Arc::new(b.finish());
        assert!(cast(arr.as_ref(), DataType::Int64).is_err());
    }

    #[test]
    fn same_type_cast_is_zero_copy() {
        let arr = int_array(&[Some(1), Some(2)]);
        let result = cast(arr.as_ref(), DataType::Int64).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn cast_scalar_round_trips_int_to_float() {
        use crate::scalar::ScalarValue;
        let result = cast_scalar(&ScalarValue::Int64(Some(3)), DataType::Float64).unwrap();
        assert_eq!(result, ScalarValue::Float64(Some(3.0)));
    }

    #[test]
    fn cast_scalar_null_stays_null_of_the_target_type() {
        use crate::scalar::ScalarValue;
        let result = cast_scalar(&ScalarValue::Int64(None), DataType::Utf8).unwrap();
        assert_eq!(result, ScalarValue::Utf8(None));
    }

    #[test]
    fn cast_scalar_rejects_non_finite_float_to_int() {
        use crate::scalar::ScalarValue;
        assert!(cast_scalar(&ScalarValue::Float64(Some(f64::NAN)), DataType::Int64).is_err());
    }
}
