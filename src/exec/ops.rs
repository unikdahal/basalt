//! Helper comparison functions for execution operators.
//!
//! Provides total ordering comparators for sorting, addressing float NaN sorting edge cases.

use crate::types::value::Value;
use std::cmp::Ordering;

/// Compare two non-null values for stable sorting.
/// Handles float comparison by establishing a total ordering where NaN is considered
/// larger than all other values (aligning with PostgreSQL and Spark conventions).
pub fn compare_values(l: &Value, r: &Value) -> Option<Ordering> {
    match (l, r) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => {
            if a.is_nan() && b.is_nan() {
                Some(Ordering::Equal)
            } else if a.is_nan() {
                Some(Ordering::Greater)
            } else if b.is_nan() {
                Some(Ordering::Less)
            } else {
                a.partial_cmp(b)
            }
        }
        (Value::Utf8(a), Value::Utf8(b)) => Some(a.cmp(b)),
        (Value::Boolean(a), Value::Boolean(b)) => Some(a.cmp(b)),
        _ => None, // Type mismatch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_values_nan_ordering() {
        let nan = Value::Float64(f64::NAN);
        let zero = Value::Float64(0.0);
        let negative = Value::Float64(-1.0);

        assert_eq!(compare_values(&nan, &nan), Some(Ordering::Equal));
        assert_eq!(compare_values(&nan, &zero), Some(Ordering::Greater));
        assert_eq!(compare_values(&negative, &nan), Some(Ordering::Less));
    }

    #[test]
    fn test_compare_values_types() {
        assert_eq!(
            compare_values(&Value::Int64(5), &Value::Int64(10)),
            Some(Ordering::Less)
        );
        assert_eq!(
            compare_values(&Value::Utf8("a".to_string()), &Value::Utf8("b".to_string())),
            Some(Ordering::Less)
        );
        assert_eq!(compare_values(&Value::Int64(5), &Value::Float64(5.0)), None);
    }
}
