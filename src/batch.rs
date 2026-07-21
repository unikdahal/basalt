//! `RecordBatch` — a schema paired with its columns. See LLD §2.7.

use crate::array::column::Column;
use crate::error::{BasaltError, Result};
use crate::types::schema::Schema;
use crate::types::value::Value;

/// A table: a schema and its columns.
/// Invariants (enforced in try_new, assumed everywhere else):
///   B1. columns.len() == schema.len()
///   B2. all columns have equal length
///   B3. columns[i].data_type() == schema.field(i).data_type
#[derive(Debug, Clone, PartialEq)]
pub struct RecordBatch {
    schema: Schema,
    columns: Vec<Column>,
    num_rows: usize,
}

impl RecordBatch {
    pub fn try_new(schema: Schema, columns: Vec<Column>) -> Result<Self> {
        if columns.len() != schema.len() {
            return Err(BasaltError::Schema {
                message: format!(
                    "schema has {} fields but {} columns were given",
                    schema.len(),
                    columns.len()
                ),
            });
        }
        let num_rows = columns.first().map_or(0, Column::len);
        for (i, (col, field)) in columns.iter().zip(schema.fields()).enumerate() {
            if col.len() != num_rows {
                return Err(BasaltError::Schema {
                    message: format!(
                        "column {i} has length {} but batch length is {num_rows}",
                        col.len()
                    ),
                });
            }
            let field_type = field.data_type;
            if col.data_type() != field_type {
                return Err(BasaltError::Schema {
                    message: format!(
                        "column {i} has type {} but schema declares {}",
                        col.data_type(),
                        field_type
                    ),
                });
            }
        }
        Ok(RecordBatch {
            schema,
            columns,
            num_rows,
        })
    }

    pub fn empty(schema: Schema) -> Self {
        let columns = schema
            .fields()
            .iter()
            .map(|f| crate::array::builder::ColumnBuilder::new(f.data_type).finish())
            .collect();
        RecordBatch {
            schema,
            columns,
            num_rows: 0,
        }
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn column(&self, index: usize) -> Option<&Column> {
        self.columns.get(index)
    }

    pub fn column_by_name(&self, name: &str) -> Option<&Column> {
        self.schema.index_of(name).and_then(|i| self.column(i))
    }

    /// Row-position selection — the primitive under filter and sort.
    pub fn take(&self, indices: &[usize]) -> Result<RecordBatch> {
        let columns = self
            .columns
            .iter()
            .map(|c| c.take(indices))
            .collect::<Result<Vec<_>>>()?;
        let num_rows = indices.len();
        Ok(RecordBatch {
            schema: self.schema.clone(),
            columns,
            num_rows,
        })
    }
}

impl std::fmt::Display for RecordBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let headers: Vec<String> = self
            .schema
            .fields()
            .iter()
            .map(|fld| fld.name.clone())
            .collect();
        let mut rows: Vec<Vec<String>> = Vec::with_capacity(self.num_rows);
        for r in 0..self.num_rows {
            let mut row = Vec::with_capacity(self.num_columns());
            for c in 0..self.num_columns() {
                let v = self.columns[c].get(r).unwrap_or(Value::Null);
                row.push(v.to_string());
            }
            rows.push(row);
        }
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        for row in &rows {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell.len());
            }
        }
        let write_row = |f: &mut std::fmt::Formatter<'_>, cells: &[String]| -> std::fmt::Result {
            write!(f, "|")?;
            for (i, cell) in cells.iter().enumerate() {
                write!(f, " {:width$} |", cell, width = widths[i])?;
            }
            writeln!(f)
        };
        write_row(f, &headers)?;
        write!(f, "|")?;
        for w in &widths {
            write!(f, "-{}-|", "-".repeat(*w))?;
        }
        writeln!(f)?;
        for row in &rows {
            write_row(f, row)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::column::{Column, ColumnData};
    use crate::types::data_type::DataType;
    use crate::types::schema::Field;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
        ])
        .unwrap()
    }

    fn batch() -> RecordBatch {
        let cols = vec![
            Column::from_parts(ColumnData::Int64(vec![1, 2, 3]), None),
            Column::from_parts(
                ColumnData::Utf8(vec!["x".into(), "y".into(), "z".into()]),
                None,
            ),
        ];
        RecordBatch::try_new(schema(), cols).unwrap()
    }

    #[test]
    fn try_new_rejects_column_count_mismatch() {
        let cols = vec![Column::from_parts(ColumnData::Int64(vec![1]), None)];
        assert!(RecordBatch::try_new(schema(), cols).is_err());
    }

    #[test]
    fn try_new_rejects_length_mismatch() {
        let cols = vec![
            Column::from_parts(ColumnData::Int64(vec![1, 2]), None),
            Column::from_parts(ColumnData::Utf8(vec!["x".into()]), None),
        ];
        assert!(RecordBatch::try_new(schema(), cols).is_err());
    }

    #[test]
    fn try_new_rejects_type_mismatch() {
        let cols = vec![
            Column::from_parts(ColumnData::Utf8(vec!["1".into()]), None),
            Column::from_parts(ColumnData::Utf8(vec!["x".into()]), None),
        ];
        assert!(RecordBatch::try_new(schema(), cols).is_err());
    }

    #[test]
    fn empty_batch_has_zero_rows_but_valid_schema() {
        let b = RecordBatch::empty(schema());
        assert_eq!(b.num_rows(), 0);
        assert_eq!(b.num_columns(), 2);
    }

    #[test]
    fn column_by_name_resolves() {
        let b = batch();
        assert_eq!(b.column_by_name("a").unwrap().get(0), Some(Value::Int64(1)));
        assert!(b.column_by_name("nope").is_none());
    }

    #[test]
    fn take_selects_rows() {
        let b = batch();
        let taken = b.take(&[2, 0]).unwrap();
        assert_eq!(taken.num_rows(), 2);
        assert_eq!(taken.column(0).unwrap().get(0), Some(Value::Int64(3)));
        assert_eq!(taken.column(0).unwrap().get(1), Some(Value::Int64(1)));
    }

    #[test]
    fn display_renders_aligned_table() {
        let b = batch();
        let rendered = b.to_string();
        assert!(rendered.contains('a'));
        assert!(rendered.contains('b'));
        assert!(rendered.contains('1'));
        assert!(rendered.contains('x'));
    }

    #[test]
    fn take_with_no_indices_yields_zero_row_batch_with_same_schema() {
        let b = batch();
        let taken = b.take(&[]).unwrap();
        assert_eq!(taken.num_rows(), 0);
        assert_eq!(taken.schema(), b.schema());
    }

    #[test]
    fn no_column_schema_gives_zero_rows_explicitly() {
        // A schema with zero fields has no columns to derive num_rows from,
        // so RecordBatch must store it explicitly rather than infer it — see B2/B3.
        let schema = Schema::new(vec![]).unwrap();
        let b = RecordBatch::try_new(schema, vec![]).unwrap();
        assert_eq!(b.num_rows(), 0);
        assert_eq!(b.num_columns(), 0);
    }

    #[test]
    fn display_renders_header_only_for_zero_row_batch() {
        let b = RecordBatch::empty(schema());
        let rendered = b.to_string();
        assert!(rendered.contains('a'));
        assert!(rendered.contains('b'));
    }

    #[test]
    fn column_returns_none_out_of_bounds() {
        let b = batch();
        assert!(b.column(99).is_none());
    }
}

// ---------------------------------------------------------------------------
// Phase 2: `ColumnarBatch` — the columnar counterpart to `RecordBatch` above.
//
// The Phase 2 LLD (§3.5) describes this as "`RecordBatch`, revised" — an
// in-place evolution. We keep it a distinctly-named, additive type instead,
// for the same reason the rest of Phase 2 lives alongside Phase 1 rather
// than replacing it: Phase 1's `RecordBatch` above is still what `exec`,
// `expr`, and `io` actually run queries on, and retiring it is a separate
// migration. `ColumnarBatch` is what `physical_expr`/`physical_plan` operate
// on; the rename happens at the real cutover, not before.
// ---------------------------------------------------------------------------

use crate::array::array::ArrayRef;
use crate::types::schema::SchemaRef;

/// A table: a shared schema and its columnar arrays.
///
/// # Invariants
/// - `columns.len() == schema.fields().len()`
/// - Every column has length `num_rows`
/// - `columns[i].data_type() == schema.field(i).data_type`
#[derive(Clone, Debug)]
pub struct ColumnarBatch {
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
    num_rows: usize,
}

impl ColumnarBatch {
    /// # Errors
    /// Errors if `columns.len()` doesn't match the schema's field count, if
    /// columns have differing lengths, or if a column's type doesn't match
    /// its declared field type.
    pub fn try_new(schema: SchemaRef, columns: Vec<ArrayRef>) -> Result<Self> {
        if columns.len() != schema.len() {
            return Err(BasaltError::Schema {
                message: format!(
                    "schema has {} fields but {} columns were given",
                    schema.len(),
                    columns.len()
                ),
            });
        }
        let num_rows = columns.first().map_or(0, |c| c.len());
        for (i, (col, field)) in columns.iter().zip(schema.fields()).enumerate() {
            if col.len() != num_rows {
                return Err(BasaltError::Schema {
                    message: format!(
                        "column {i} has length {} but batch length is {num_rows}",
                        col.len()
                    ),
                });
            }
            if col.data_type() != field.data_type {
                return Err(BasaltError::Schema {
                    message: format!(
                        "column {i} has type {} but schema declares {}",
                        col.data_type(),
                        field.data_type
                    ),
                });
            }
        }
        Ok(ColumnarBatch {
            schema,
            columns,
            num_rows,
        })
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn column(&self, index: usize) -> Option<&ArrayRef> {
        self.columns.get(index)
    }

    pub fn column_by_name(&self, name: &str) -> Option<&ArrayRef> {
        self.schema.index_of(name).and_then(|i| self.column(i))
    }

    /// O(1): every column's `slice` is a zero-copy `Arc`-shared window.
    pub fn slice(&self, offset: usize, len: usize) -> ColumnarBatch {
        let columns = self.columns.iter().map(|c| c.slice(offset, len)).collect();
        ColumnarBatch {
            schema: self.schema.clone(),
            columns,
            num_rows: len,
        }
    }

    /// # Errors
    /// Errors if any index is out of range for the schema's field count.
    pub fn project(&self, indices: &[usize]) -> Result<ColumnarBatch> {
        let schema = std::sync::Arc::new(self.schema.project(indices)?);
        let columns = indices
            .iter()
            .map(|&i| {
                self.columns
                    .get(i)
                    .cloned()
                    .ok_or_else(|| BasaltError::Internal(format!("column index {i} out of bounds")))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ColumnarBatch {
            schema,
            columns,
            num_rows: self.num_rows,
        })
    }
}

#[cfg(test)]
mod columnar_batch_tests {
    use super::*;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::types::data_type::DataType;
    use crate::types::schema::Field;
    use std::sync::Arc;

    fn schema() -> SchemaRef {
        Arc::new(
            Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Int64, false),
            ])
            .unwrap(),
        )
    }

    fn int_array(values: &[i64]) -> ArrayRef {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            b.append_value(v);
        }
        Arc::new(b.finish())
    }

    fn batch() -> ColumnarBatch {
        ColumnarBatch::try_new(
            schema(),
            vec![int_array(&[1, 2, 3]), int_array(&[10, 20, 30])],
        )
        .unwrap()
    }

    #[test]
    fn try_new_rejects_column_count_mismatch() {
        assert!(ColumnarBatch::try_new(schema(), vec![int_array(&[1])]).is_err());
    }

    #[test]
    fn try_new_rejects_length_mismatch() {
        assert!(
            ColumnarBatch::try_new(schema(), vec![int_array(&[1, 2]), int_array(&[1])]).is_err()
        );
    }

    #[test]
    fn slice_is_zero_copy_and_reindexes() {
        let b = batch();
        let sliced = b.slice(1, 2);
        assert_eq!(sliced.num_rows(), 2);
        let col = sliced.column(0).unwrap();
        assert_eq!(col.len(), 2);
    }

    #[test]
    fn project_reorders_columns_and_schema_together() {
        let b = batch();
        let projected = b.project(&[1, 0]).unwrap();
        assert_eq!(projected.schema().field(0).unwrap().name, "b");
        assert_eq!(projected.schema().field(1).unwrap().name, "a");
    }

    #[test]
    fn column_by_name_resolves() {
        let b = batch();
        assert!(b.column_by_name("a").is_some());
        assert!(b.column_by_name("nope").is_none());
    }

    #[test]
    fn cloning_a_batch_is_cheap_arc_bumps_only() {
        let b = batch();
        let col_ptr_before = Arc::as_ptr(&b.schema);
        let cloned = b.clone();
        assert_eq!(Arc::as_ptr(&cloned.schema), col_ptr_before);
    }
}
