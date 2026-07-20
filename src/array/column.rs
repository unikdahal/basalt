//! `Column`, `ColumnData`. See LLD §2.4.

use super::validity::Validity;
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;
use crate::types::value::Value;

/// The typed data of a column. One Vec per supported type.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Utf8(Vec<String>),
    Boolean(Vec<bool>),
}

impl ColumnData {
    fn len(&self) -> usize {
        match self {
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::Utf8(v) => v.len(),
            ColumnData::Boolean(v) => v.len(),
        }
    }

    fn data_type(&self) -> DataType {
        match self {
            ColumnData::Int64(_) => DataType::Int64,
            ColumnData::Float64(_) => DataType::Float64,
            ColumnData::Utf8(_) => DataType::Utf8,
            ColumnData::Boolean(_) => DataType::Boolean,
        }
    }

    fn value_at(&self, index: usize) -> Option<Value> {
        match self {
            ColumnData::Int64(v) => v.get(index).map(|&x| Value::Int64(x)),
            ColumnData::Float64(v) => v.get(index).map(|&x| Value::Float64(x)),
            ColumnData::Utf8(v) => v.get(index).map(|x| Value::Utf8(x.clone())),
            ColumnData::Boolean(v) => v.get(index).map(|&x| Value::Boolean(x)),
        }
    }

    /// `None` if any index is out of bounds for the underlying storage.
    fn take(&self, indices: &[usize]) -> Option<ColumnData> {
        match self {
            ColumnData::Int64(v) => indices
                .iter()
                .map(|&i| v.get(i).copied())
                .collect::<Option<Vec<_>>>()
                .map(ColumnData::Int64),
            ColumnData::Float64(v) => indices
                .iter()
                .map(|&i| v.get(i).copied())
                .collect::<Option<Vec<_>>>()
                .map(ColumnData::Float64),
            ColumnData::Utf8(v) => indices
                .iter()
                .map(|&i| v.get(i).cloned())
                .collect::<Option<Vec<_>>>()
                .map(ColumnData::Utf8),
            ColumnData::Boolean(v) => indices
                .iter()
                .map(|&i| v.get(i).copied())
                .collect::<Option<Vec<_>>>()
                .map(ColumnData::Boolean),
        }
    }
}

/// A column: typed data plus optional null tracking.
/// Invariants:
///   I1. validity.is_none()  =>  the column contains no nulls
///   I2. validity.is_some()  =>  validity.len() == data length
///   I3. slots marked null still hold a well-formed (garbage) value in `data`
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    data: ColumnData,
    validity: Option<Validity>,
}

impl Column {
    /// Constructs directly from parts. Used by `ColumnBuilder::finish`.
    /// `debug_assert`s I2 rather than returning `Result` — a violation here
    /// is an internal bug, not user-facing.
    pub(crate) fn from_parts(data: ColumnData, validity: Option<Validity>) -> Self {
        if let Some(v) = &validity {
            debug_assert_eq!(v.len(), data.len());
        }
        Column { data, validity }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    pub fn data_type(&self) -> DataType {
        self.data.data_type()
    }

    pub fn null_count(&self) -> usize {
        self.validity.as_ref().map_or(0, Validity::null_count)
    }

    pub fn is_null(&self, index: usize) -> bool {
        self.validity.as_ref().is_some_and(|v| v.is_null(index))
    }

    /// Materialize one scalar. None if index out of bounds; Some(Value::Null) if null.
    pub fn get(&self, index: usize) -> Option<Value> {
        if index >= self.len() {
            return None;
        }
        if self.is_null(index) {
            return Some(Value::Null);
        }
        self.data.value_at(index)
    }

    /// Produce a new Column containing only the given row positions, in order.
    /// Used by filter (selection vector) and sort (permutation).
    pub fn take(&self, indices: &[usize]) -> Result<Column> {
        let data = self.data.take(indices).ok_or_else(|| {
            BasaltError::Internal(format!(
                "take index out of bounds for column of length {}",
                self.len()
            ))
        })?;
        let validity = self.validity.as_ref().map(|v| v.take(indices));
        Ok(Column::from_parts(data, validity))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::validity::Validity;

    fn int_column() -> Column {
        Column::from_parts(
            ColumnData::Int64(vec![10, 20, 30]),
            Some(Validity::from_flags(vec![true, false, true])),
        )
    }

    #[test]
    fn no_validity_means_no_nulls() {
        let c = Column::from_parts(ColumnData::Int64(vec![1, 2]), None);
        assert_eq!(c.null_count(), 0);
        assert!(!c.is_null(0));
        assert_eq!(c.get(0), Some(Value::Int64(1)));
    }

    #[test]
    fn get_returns_null_for_null_slots() {
        let c = int_column();
        assert_eq!(c.get(1), Some(Value::Null));
        assert_eq!(c.get(0), Some(Value::Int64(10)));
    }

    #[test]
    fn get_out_of_bounds_is_none() {
        let c = int_column();
        assert_eq!(c.get(99), None);
    }

    #[test]
    fn take_reorders_data_and_validity() {
        let c = int_column();
        let taken = c.take(&[2, 1, 0]).unwrap();
        assert_eq!(taken.get(0), Some(Value::Int64(30)));
        assert_eq!(taken.get(1), Some(Value::Null));
        assert_eq!(taken.get(2), Some(Value::Int64(10)));
    }

    #[test]
    fn take_out_of_bounds_errors() {
        let c = int_column();
        assert!(c.take(&[5]).is_err());
    }

    #[test]
    fn data_type_matches_variant() {
        let c = Column::from_parts(ColumnData::Utf8(vec!["a".into()]), None);
        assert_eq!(c.data_type(), DataType::Utf8);
    }

    #[test]
    fn take_with_empty_indices_yields_empty_column() {
        let c = int_column();
        let taken = c.take(&[]).unwrap();
        assert_eq!(taken.len(), 0);
        assert!(taken.is_empty());
    }

    #[test]
    fn take_can_repeat_and_duplicate_positions() {
        let c = int_column();
        let taken = c.take(&[0, 0, 0]).unwrap();
        assert_eq!(taken.len(), 3);
        assert_eq!(taken.get(0), Some(Value::Int64(10)));
        assert_eq!(taken.get(2), Some(Value::Int64(10)));
    }

    #[test]
    fn empty_column_reports_zero_length_and_no_nulls() {
        let c: Column = Column::from_parts(ColumnData::Int64(vec![]), None);
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert_eq!(c.null_count(), 0);
        assert_eq!(c.get(0), None);
    }

    #[test]
    fn all_null_column_reports_full_null_count() {
        let c = Column::from_parts(
            ColumnData::Int64(vec![0, 0, 0]),
            Some(Validity::from_flags(vec![false, false, false])),
        );
        assert_eq!(c.null_count(), 3);
        assert_eq!(c.get(0), Some(Value::Null));
        assert_eq!(c.get(2), Some(Value::Null));
    }
}
