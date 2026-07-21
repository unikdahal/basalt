//! `ScalarValue` — a single value that can stand in for a whole array in a
//! vectorized computation (e.g. the `5` in `col_a + 5`). See
//! design-docs/basalt-phase2-lld.md §4.1.
//!
//! **Deliberately not a reuse of `types::value::Value`, despite the LLD's
//! phrasing ("`Value` becomes `ScalarValue`").** Phase 1's `Value::Null` is
//! untyped by design (§2.2 of the Phase 1 LLD): evaluation always knows the
//! expected type from the bound expression tree, so the value itself never
//! needs to carry one. `ColumnarValue` breaks that assumption — a compute
//! kernel dispatches purely on what's in front of it (an array's own
//! `data_type()`, or a scalar's tag), with no expression tree to consult. A
//! `NULL` scalar of unknown type can't tell an `add` kernel whether to run
//! the `i64` or `f64` loop. So every variant here carries its type even when
//! the value is absent — matching arrow-rs's own `ScalarValue` for the same
//! reason.

use crate::types::data_type::DataType;

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Int64(Option<i64>),
    Float64(Option<f64>),
    Utf8(Option<String>),
    Boolean(Option<bool>),
}

impl ScalarValue {
    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::Int64(_) => DataType::Int64,
            ScalarValue::Float64(_) => DataType::Float64,
            ScalarValue::Utf8(_) => DataType::Utf8,
            ScalarValue::Boolean(_) => DataType::Boolean,
        }
    }

    pub fn is_null(&self) -> bool {
        match self {
            ScalarValue::Int64(v) => v.is_none(),
            ScalarValue::Float64(v) => v.is_none(),
            ScalarValue::Utf8(v) => v.is_none(),
            ScalarValue::Boolean(v) => v.is_none(),
        }
    }
}

impl std::fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarValue::Int64(Some(v)) => write!(f, "{v}"),
            ScalarValue::Float64(Some(v)) => write!(f, "{v}"),
            ScalarValue::Utf8(Some(v)) => write!(f, "{v}"),
            ScalarValue::Boolean(Some(v)) => write!(f, "{v}"),
            _ => write!(f, "NULL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_variants_report_their_type() {
        assert_eq!(ScalarValue::Int64(None).data_type(), DataType::Int64);
        assert_eq!(ScalarValue::Float64(None).data_type(), DataType::Float64);
        assert!(ScalarValue::Int64(None).is_null());
    }

    #[test]
    fn non_null_variants_report_value_and_type() {
        let v = ScalarValue::Int64(Some(5));
        assert!(!v.is_null());
        assert_eq!(v.data_type(), DataType::Int64);
        assert_eq!(v.to_string(), "5");
    }

    #[test]
    fn display_renders_null_uniformly() {
        assert_eq!(ScalarValue::Int64(None).to_string(), "NULL");
        assert_eq!(ScalarValue::Utf8(None).to_string(), "NULL");
    }
}
