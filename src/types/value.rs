//! `Value` — a single scalar. See LLD §2.2.

use super::data_type::DataType;
use crate::error::{BasaltError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int64(i64),
    Float64(f64),
    Utf8(String),
    Boolean(bool),
    Null,
}

impl Value {
    /// None for Null — a null literal has no intrinsic type.
    pub fn data_type(&self) -> Option<DataType> {
        match self {
            Value::Int64(_) => Some(DataType::Int64),
            Value::Float64(_) => Some(DataType::Float64),
            Value::Utf8(_) => Some(DataType::Utf8),
            Value::Boolean(_) => Some(DataType::Boolean),
            Value::Null => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Runtime coercion used by eval. Errors if the cast is illegal.
    pub fn cast_to(&self, target: DataType) -> Result<Value> {
        if self.is_null() {
            return Ok(Value::Null);
        }
        match (self, target) {
            (Value::Int64(v), DataType::Int64) => Ok(Value::Int64(*v)),
            (Value::Int64(v), DataType::Float64) => Ok(Value::Float64(*v as f64)),
            (Value::Int64(v), DataType::Utf8) => Ok(Value::Utf8(v.to_string())),
            (Value::Int64(v), DataType::Boolean) => Ok(Value::Boolean(*v != 0)),

            (Value::Float64(v), DataType::Float64) => Ok(Value::Float64(*v)),
            (Value::Float64(v), DataType::Int64) => Ok(Value::Int64(*v as i64)),
            (Value::Float64(v), DataType::Utf8) => Ok(Value::Utf8(v.to_string())),

            (Value::Utf8(v), DataType::Utf8) => Ok(Value::Utf8(v.clone())),
            (Value::Utf8(v), DataType::Int64) => {
                v.parse::<i64>()
                    .map(Value::Int64)
                    .map_err(|_| BasaltError::Type {
                        message: format!("cannot cast '{v}' to Int64"),
                    })
            }
            (Value::Utf8(v), DataType::Float64) => {
                v.parse::<f64>()
                    .map(Value::Float64)
                    .map_err(|_| BasaltError::Type {
                        message: format!("cannot cast '{v}' to Float64"),
                    })
            }
            (Value::Utf8(v), DataType::Boolean) => match v.to_ascii_lowercase().as_str() {
                "true" => Ok(Value::Boolean(true)),
                "false" => Ok(Value::Boolean(false)),
                _ => Err(BasaltError::Type {
                    message: format!("cannot cast '{v}' to Boolean"),
                }),
            },

            (Value::Boolean(v), DataType::Boolean) => Ok(Value::Boolean(*v)),
            (Value::Boolean(v), DataType::Utf8) => Ok(Value::Utf8(v.to_string())),

            (other, target) => Err(BasaltError::Type {
                message: format!(
                    "cannot cast {} to {target}",
                    other.data_type().map_or("Null", |d| d.name())
                ),
            }),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int64(v) => write!(f, "{v}"),
            Value::Float64(v) => write!(f, "{v}"),
            Value::Utf8(v) => write!(f, "{v}"),
            Value::Boolean(v) => write!(f, "{v}"),
            Value::Null => write!(f, "NULL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_has_no_type() {
        assert_eq!(Value::Null.data_type(), None);
        assert!(Value::Null.is_null());
    }

    #[test]
    fn typed_values_report_type() {
        assert_eq!(Value::Int64(1).data_type(), Some(DataType::Int64));
        assert_eq!(Value::Float64(1.0).data_type(), Some(DataType::Float64));
        assert_eq!(Value::Utf8("x".into()).data_type(), Some(DataType::Utf8));
        assert_eq!(Value::Boolean(true).data_type(), Some(DataType::Boolean));
    }

    #[test]
    fn cast_null_stays_null() {
        assert_eq!(Value::Null.cast_to(DataType::Int64).unwrap(), Value::Null);
    }

    #[test]
    fn cast_i64_min_and_max_to_utf8_round_trip_through_parse() {
        assert_eq!(
            Value::Int64(i64::MAX).cast_to(DataType::Utf8).unwrap(),
            Value::Utf8(i64::MAX.to_string())
        );
        assert_eq!(
            Value::Utf8(i64::MIN.to_string())
                .cast_to(DataType::Int64)
                .unwrap(),
            Value::Int64(i64::MIN)
        );
    }

    #[test]
    fn cast_non_ascii_string_to_int_errors_instead_of_panicking() {
        let err = Value::Utf8("héllo".into())
            .cast_to(DataType::Int64)
            .unwrap_err();
        assert!(matches!(err, BasaltError::Type { .. }));
    }

    #[test]
    fn cast_empty_string_to_int_errors() {
        assert!(Value::Utf8(String::new()).cast_to(DataType::Int64).is_err());
    }

    #[test]
    fn cast_int_to_float() {
        assert_eq!(
            Value::Int64(3).cast_to(DataType::Float64).unwrap(),
            Value::Float64(3.0)
        );
    }

    #[test]
    fn cast_string_to_int_ok_and_err() {
        assert_eq!(
            Value::Utf8("42".into()).cast_to(DataType::Int64).unwrap(),
            Value::Int64(42)
        );
        assert!(Value::Utf8("nope".into()).cast_to(DataType::Int64).is_err());
    }

    #[test]
    fn cast_string_to_bool() {
        assert_eq!(
            Value::Utf8("true".into())
                .cast_to(DataType::Boolean)
                .unwrap(),
            Value::Boolean(true)
        );
        assert!(Value::Utf8("nah".into())
            .cast_to(DataType::Boolean)
            .is_err());
    }

    /// Regression test for the unsupported-cast fallback arm: it must format
    /// the error message from the source value's type without panicking (a
    /// prior version reached this via `.data_type().unwrap()`, which happened
    /// to be safe only because nulls short-circuit earlier, but was still an
    /// unwrap in library code masking that guarantee).
    #[test]
    fn cast_unsupported_combination_reports_source_type_without_panicking() {
        let err = Value::Float64(1.5).cast_to(DataType::Boolean).unwrap_err();
        assert!(matches!(err, BasaltError::Type { .. }));
        assert!(err.to_string().contains("Float64"));
    }

    #[test]
    fn display_formats_values() {
        assert_eq!(Value::Int64(5).to_string(), "5");
        assert_eq!(Value::Null.to_string(), "NULL");
        assert_eq!(Value::Boolean(false).to_string(), "false");
    }
}
