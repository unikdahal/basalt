//! `ColumnBuilder` — incremental, typed column construction. See LLD §2.5.

use super::column::{Column, ColumnData};
use super::validity::Validity;
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;
use crate::types::value::Value;

pub struct ColumnBuilder {
    data: ColumnData,
    validity: Vec<bool>,
    null_count: usize,
}

impl ColumnBuilder {
    pub fn new(data_type: DataType) -> Self {
        Self::with_capacity(data_type, 0)
    }

    pub fn with_capacity(data_type: DataType, capacity: usize) -> Self {
        let data = match data_type {
            DataType::Int64 => ColumnData::Int64(Vec::with_capacity(capacity)),
            DataType::Float64 => ColumnData::Float64(Vec::with_capacity(capacity)),
            DataType::Utf8 => ColumnData::Utf8(Vec::with_capacity(capacity)),
            DataType::Boolean => ColumnData::Boolean(Vec::with_capacity(capacity)),
        };
        ColumnBuilder {
            data,
            validity: Vec::with_capacity(capacity),
            null_count: 0,
        }
    }

    pub fn data_type(&self) -> DataType {
        match &self.data {
            ColumnData::Int64(_) => DataType::Int64,
            ColumnData::Float64(_) => DataType::Float64,
            ColumnData::Utf8(_) => DataType::Utf8,
            ColumnData::Boolean(_) => DataType::Boolean,
        }
    }

    /// Errors if value's type doesn't match the builder's type.
    pub fn append_value(&mut self, value: Value) -> Result<()> {
        if value.is_null() {
            self.append_null();
            return Ok(());
        }
        match (&mut self.data, value) {
            (ColumnData::Int64(v), Value::Int64(x)) => v.push(x),
            (ColumnData::Float64(v), Value::Float64(x)) => v.push(x),
            (ColumnData::Utf8(v), Value::Utf8(x)) => v.push(x),
            (ColumnData::Boolean(v), Value::Boolean(x)) => v.push(x),
            (_, other) => {
                return Err(BasaltError::Type {
                    message: format!(
                        "cannot append {} into a {} builder",
                        other.data_type().map(|d| d.name()).unwrap_or("Null"),
                        self.data_type()
                    ),
                });
            }
        }
        self.validity.push(true);
        Ok(())
    }

    pub fn append_null(&mut self) {
        match &mut self.data {
            ColumnData::Int64(v) => v.push(i64::default()),
            ColumnData::Float64(v) => v.push(f64::default()),
            ColumnData::Utf8(v) => v.push(String::default()),
            ColumnData::Boolean(v) => v.push(bool::default()),
        }
        self.validity.push(false);
        self.null_count += 1;
    }

    pub fn len(&self) -> usize {
        self.validity.len()
    }

    pub fn is_empty(&self) -> bool {
        self.validity.is_empty()
    }

    /// Consumes the builder. Drops validity entirely if null_count == 0.
    pub fn finish(self) -> Column {
        let validity = if self.null_count == 0 {
            None
        } else {
            Some(Validity::from_flags(self.validity))
        };
        Column::from_parts(self.data, validity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_column_with_no_nulls_seen() {
        let mut b = ColumnBuilder::new(DataType::Int64);
        b.append_value(Value::Int64(1)).unwrap();
        b.append_value(Value::Int64(2)).unwrap();
        let col = b.finish();
        assert_eq!(col.null_count(), 0);
        assert_eq!(col.get(0), Some(Value::Int64(1)));
    }

    #[test]
    fn append_null_tracks_count() {
        let mut b = ColumnBuilder::new(DataType::Utf8);
        b.append_value(Value::Utf8("a".into())).unwrap();
        b.append_null();
        let col = b.finish();
        assert_eq!(col.null_count(), 1);
        assert_eq!(col.get(1), Some(Value::Null));
    }

    #[test]
    fn append_value_null_delegates_to_append_null() {
        let mut b = ColumnBuilder::new(DataType::Int64);
        b.append_value(Value::Null).unwrap();
        let col = b.finish();
        assert_eq!(col.null_count(), 1);
    }

    #[test]
    fn type_mismatch_errors() {
        let mut b = ColumnBuilder::new(DataType::Int64);
        assert!(b.append_value(Value::Utf8("x".into())).is_err());
    }

    #[test]
    fn len_tracks_appends() {
        let mut b = ColumnBuilder::new(DataType::Boolean);
        assert!(b.is_empty());
        b.append_value(Value::Boolean(true)).unwrap();
        assert_eq!(b.len(), 1);
    }
}
