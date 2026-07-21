//! Parquet I/O. See design-docs/basalt-phase2-lld.md §9.
//!
//! **Deliberate exception to this project's "hand-roll everything" rule**,
//! per the LLD's own explicit recommendation: Parquet's format is
//! genuinely large (Thrift-encoded metadata, several encodings and
//! compression codecs, definition/repetition levels, page indexes), and
//! reimplementing it teaches serialization, not query engineering. Real
//! reads and writes go through the `parquet`/`arrow` crates (Apache-
//! maintained, correct, fast — what DataFusion and Comet use); this module
//! adds a thin conversion layer to/from Basalt's own array types plus the
//! pushdown logic on top, which *is* the part worth building by hand.
//!
//! **Two levels of pushdown implemented; the third is a documented gap:**
//! 1. **Projection pushdown** — [`ProjectionMask`] restricts which column
//!    chunks are even decoded. The single biggest win: selecting 3 of 50
//!    columns reads roughly 6% of the file's bytes.
//! 2. **Row-group skipping via statistics** — [`RowGroupPredicate`]
//!    evaluates a single `column OP literal` condition against each row
//!    group's min/max metadata *without decoding a single page*. This is a
//!    direct precursor to Phase 3's cardinality estimation (same interval
//!    reasoning) and Phase 4's Iceberg partition pruning (same idea, one
//!    level up, using manifest statistics).
//! 3. **Page-level skipping** (page indexes) is not implemented — the LLD
//!    itself ranks it the lowest-payoff of the three, and it needs the same
//!    interval machinery as #2 with none of the novelty.
//!
//! `RowGroupPredicate` is deliberately a small, closed shape (one column,
//! one comparison operator, one literal) rather than accepting an arbitrary
//! `PhysicalExpr` — evaluating a *general* expression tree against interval
//! statistics (rather than concrete values) is real interval-arithmetic
//! work belonging to Phase 3's cost model, not restated here.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::RecordBatchReader;

use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection};
use parquet::arrow::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::file::statistics::Statistics;

use crate::array::array::{as_boolean, as_primitive, as_string, Array, ArrayRef};
use crate::array::boolean::BooleanBuilder;
use crate::array::primitive::PrimitiveBuilder;
use crate::array::string::StringBuilder;
use crate::array::types::{Float64Type, Int64Type};
use crate::batch::ColumnarBatch;
use crate::error::{BasaltError, Result};
use crate::scalar::ScalarValue;
use crate::types::coercion::BinaryOp;
use crate::types::data_type::DataType;
use crate::types::schema::{Field, Schema, SchemaRef};

fn to_basalt_err(e: impl std::fmt::Display) -> BasaltError {
    BasaltError::Internal(format!("parquet error: {e}"))
}

fn arrow_type_to_basalt(dt: &arrow::datatypes::DataType) -> Result<DataType> {
    use arrow::datatypes::DataType as ArrowDataType;
    match dt {
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        ArrowDataType::Boolean => Ok(DataType::Boolean),
        other => Err(BasaltError::Type {
            message: format!("unsupported Parquet column type {other:?}"),
        }),
    }
}

fn basalt_type_to_arrow(dt: DataType) -> arrow::datatypes::DataType {
    use arrow::datatypes::DataType as ArrowDataType;
    match dt {
        DataType::Int64 => ArrowDataType::Int64,
        DataType::Float64 => ArrowDataType::Float64,
        DataType::Utf8 => ArrowDataType::Utf8,
        DataType::Boolean => ArrowDataType::Boolean,
    }
}

fn arrow_schema_to_basalt(schema: &arrow::datatypes::Schema) -> Result<Schema> {
    let fields = schema
        .fields()
        .iter()
        .map(|f| {
            Ok(Field::new(
                f.name().clone(),
                arrow_type_to_basalt(f.data_type())?,
                f.is_nullable(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Schema::new(fields)
}

fn basalt_schema_to_arrow(schema: &Schema) -> arrow::datatypes::Schema {
    let fields: Vec<arrow::datatypes::Field> = schema
        .fields()
        .iter()
        .map(|f| {
            arrow::datatypes::Field::new(&f.name, basalt_type_to_arrow(f.data_type), f.nullable)
        })
        .collect();
    arrow::datatypes::Schema::new(fields)
}

fn arrow_array_to_basalt(array: &dyn arrow::array::Array) -> Result<ArrayRef> {
    use arrow::array::Array as ArrowArray;
    use arrow::datatypes::DataType as ArrowDataType;
    Ok(match array.data_type() {
        ArrowDataType::Int64 => {
            let a = array
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .ok_or_else(|| BasaltError::Internal("expected Int64Array".to_string()))?;
            let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(a.len());
            for i in 0..a.len() {
                if a.is_null(i) {
                    b.append_null();
                } else {
                    b.append_value(a.value(i));
                }
            }
            Arc::new(b.finish())
        }
        ArrowDataType::Float64 => {
            let a = array
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| BasaltError::Internal("expected Float64Array".to_string()))?;
            let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(a.len());
            for i in 0..a.len() {
                if a.is_null(i) {
                    b.append_null();
                } else {
                    b.append_value(a.value(i));
                }
            }
            Arc::new(b.finish())
        }
        ArrowDataType::Utf8 => {
            let a = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| BasaltError::Internal("expected StringArray".to_string()))?;
            let mut b = StringBuilder::with_capacity(a.len(), 0);
            for i in 0..a.len() {
                if a.is_null(i) {
                    b.append_null();
                } else {
                    b.append_value(a.value(i))?;
                }
            }
            Arc::new(b.finish())
        }
        ArrowDataType::Boolean => {
            let a = array
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .ok_or_else(|| BasaltError::Internal("expected BooleanArray".to_string()))?;
            let mut b = BooleanBuilder::with_capacity(a.len());
            for i in 0..a.len() {
                if a.is_null(i) {
                    b.append_null();
                } else {
                    b.append_value(a.value(i));
                }
            }
            Arc::new(b.finish())
        }
        other => {
            return Err(BasaltError::Type {
                message: format!("unsupported Parquet column type {other:?}"),
            })
        }
    })
}

fn basalt_array_to_arrow(array: &dyn Array) -> Result<arrow::array::ArrayRef> {
    Ok(match array.data_type() {
        DataType::Int64 => {
            let a = as_primitive::<Int64Type>(array)?;
            let values: Vec<Option<i64>> = (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect();
            Arc::new(arrow::array::Int64Array::from(values))
        }
        DataType::Float64 => {
            let a = as_primitive::<Float64Type>(array)?;
            let values: Vec<Option<f64>> = (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect();
            Arc::new(arrow::array::Float64Array::from(values))
        }
        DataType::Utf8 => {
            let a = as_string(array)?;
            let values: Vec<Option<&str>> = (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect();
            Arc::new(arrow::array::StringArray::from(values))
        }
        DataType::Boolean => {
            let a = as_boolean(array)?;
            let values: Vec<Option<bool>> = (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect();
            Arc::new(arrow::array::BooleanArray::from(values))
        }
    })
}

/// A single `column OP literal` condition, evaluated against a row group's
/// min/max statistics to decide whether the row group can be skipped
/// entirely. See the module doc for why this is a closed shape rather than
/// a general `PhysicalExpr`.
pub struct RowGroupPredicate {
    /// Index into the *file's* schema (before any projection).
    pub column_index: usize,
    pub op: BinaryOp,
    pub value: ScalarValue,
}

/// Returns `false` only when the statistics *prove* no row in this group
/// can satisfy the predicate — anything uncertain (missing statistics, an
/// unsupported operator/type combination) conservatively returns `true`
/// ("must scan"), since a wrongly-skipped row group is a silently wrong
/// answer and a wrongly-scanned one is just a missed optimization.
fn row_group_may_match(stats: Option<&Statistics>, pred: &RowGroupPredicate) -> bool {
    let Some(stats) = stats else { return true };

    let (min, max): (f64, f64) = match (stats, &pred.value) {
        (Statistics::Int64(s), ScalarValue::Int64(Some(_))) => match (s.min_opt(), s.max_opt()) {
            (Some(&min), Some(&max)) => (min as f64, max as f64),
            _ => return true,
        },
        (Statistics::Double(s), ScalarValue::Float64(Some(_))) => {
            match (s.min_opt(), s.max_opt()) {
                (Some(&min), Some(&max)) => (min, max),
                _ => return true,
            }
        }
        _ => return true, // Utf8/Boolean statistics, or a type mismatch: not supported here.
    };
    let value = match pred.value {
        ScalarValue::Int64(Some(v)) => v as f64,
        ScalarValue::Float64(Some(v)) => v,
        _ => return true,
    };

    match pred.op {
        BinaryOp::Gt => max > value,
        BinaryOp::GtEq => max >= value,
        BinaryOp::Lt => min < value,
        BinaryOp::LtEq => min <= value,
        BinaryOp::Eq => min <= value && value <= max,
        _ => true, // NotEq and non-comparison ops: statistics can't prove exclusion.
    }
}

/// A zero-length array of the given type — used when every row group was
/// skipped (or the file has none), so there's no arrow batch to derive
/// columns from but the output still needs the right number of
/// correctly-typed, zero-row columns to satisfy `ColumnarBatch`'s schema
/// invariant.
fn empty_array_for(data_type: DataType) -> ArrayRef {
    match data_type {
        DataType::Int64 => Arc::new(PrimitiveBuilder::<Int64Type>::with_capacity(0).finish()),
        DataType::Float64 => Arc::new(PrimitiveBuilder::<Float64Type>::with_capacity(0).finish()),
        DataType::Utf8 => Arc::new(StringBuilder::with_capacity(0, 0).finish()),
        DataType::Boolean => Arc::new(BooleanBuilder::with_capacity(0).finish()),
    }
}

/// Reads an entire Parquet file into a single [`ColumnarBatch`], with
/// optional column projection.
///
/// # Errors
/// Errors if the file can't be opened/parsed, or if it contains a column
/// type Basalt doesn't support (only `Int64`/`Float64`/`Utf8`/`Boolean`).
pub fn read_file(path: impl AsRef<Path>, projection: Option<&[usize]>) -> Result<ColumnarBatch> {
    let file = std::fs::File::open(path)?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(to_basalt_err)?;
    if let Some(proj) = projection {
        let mask = ProjectionMask::leaves(builder.parquet_schema(), proj.iter().copied());
        builder = builder.with_projection(mask);
    }
    let reader = builder.build().map_err(to_basalt_err)?;
    // `reader.schema()`, not the builder's — the builder's schema is the
    // *file's* full schema regardless of projection; the reader's reflects
    // whichever columns `with_projection` actually selected. Using the
    // builder's here was a real bug: `ColumnarBatch::try_new` below would
    // reject every projected read with a field-count mismatch.
    let basalt_schema = Arc::new(arrow_schema_to_basalt(reader.schema().as_ref())?);

    let mut batches = Vec::new();
    for batch in reader {
        let batch = batch.map_err(to_basalt_err)?;
        let columns = (0..batch.num_columns())
            .map(|i| arrow_array_to_basalt(batch.column(i).as_ref()))
            .collect::<Result<Vec<_>>>()?;
        batches.push(ColumnarBatch::try_new(basalt_schema.clone(), columns)?);
    }
    if batches.is_empty() {
        // No row groups at all: still need one correctly-typed empty
        // column per field, not zero columns.
        let columns = basalt_schema
            .fields()
            .iter()
            .map(|f| empty_array_for(f.data_type))
            .collect();
        return ColumnarBatch::try_new(basalt_schema, columns);
    }
    let num_columns = basalt_schema.len();
    let mut columns = Vec::with_capacity(num_columns);
    for col_idx in 0..num_columns {
        let parts: Vec<ArrayRef> = batches
            .iter()
            .map(|b| Arc::clone(b.column(col_idx).unwrap()))
            .collect();
        columns.push(crate::compute::concat::concat(&parts)?);
    }
    ColumnarBatch::try_new(basalt_schema, columns)
}

/// Writes a single [`ColumnarBatch`] to a new Parquet file, with statistics
/// enabled (needed for `row_group_may_match` to work on files this writes).
///
/// # Errors
/// Errors if the file can't be created, or if the batch's columns fail to
/// convert to their Arrow equivalents.
pub fn write_file(path: impl AsRef<Path>, batch: &ColumnarBatch) -> Result<()> {
    let arrow_schema = Arc::new(basalt_schema_to_arrow(batch.schema()));
    let arrow_columns = (0..batch.num_columns())
        .map(|i| basalt_array_to_arrow(batch.column(i).unwrap().as_ref()))
        .collect::<Result<Vec<_>>>()?;
    let arrow_batch =
        arrow::record_batch::RecordBatch::try_new(arrow_schema.clone(), arrow_columns)
            .map_err(to_basalt_err)?;

    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, arrow_schema, None).map_err(to_basalt_err)?;
    writer.write(&arrow_batch).map_err(to_basalt_err)?;
    writer.close().map_err(to_basalt_err)?;
    Ok(())
}

/// A Parquet-backed `TableSource`/scan configuration: path, schema, and the
/// pushdown Phase 2 supports (projection, one row-group-skipping predicate).
pub struct ParquetScanConfig {
    pub path: PathBuf,
    pub schema: SchemaRef,
    pub projection: Option<Vec<usize>>,
    pub predicate: Option<RowGroupPredicate>,
}

impl ParquetScanConfig {
    /// Reads the file's schema without decoding any row groups.
    ///
    /// # Errors
    /// Errors if the file can't be opened or its metadata parsed.
    pub fn discover_schema(path: impl AsRef<Path>) -> Result<SchemaRef> {
        let file = std::fs::File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(to_basalt_err)?;
        Ok(Arc::new(arrow_schema_to_basalt(builder.schema())?))
    }

    /// Executes the configured scan: opens the file, applies row-group
    /// skipping and projection, and returns every surviving row group as one
    /// concatenated batch (matching this crate's other pipeline-breaking
    /// operators' "one batch out" simplification — see `SortExec`'s doc
    /// comment for the same call made there).
    ///
    /// # Errors
    /// Errors if the file can't be read or a column's type isn't supported.
    pub fn execute(&self) -> Result<ColumnarBatch> {
        let file = std::fs::File::open(&self.path)?;
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(to_basalt_err)?;

        if let Some(pred) = &self.predicate {
            let metadata = builder.metadata().clone();
            let mut skip_row_groups = Vec::new();
            for i in 0..metadata.num_row_groups() {
                let column_meta = metadata.row_group(i).column(pred.column_index);
                if !row_group_may_match(column_meta.statistics(), pred) {
                    skip_row_groups.push(i);
                }
            }
            if !skip_row_groups.is_empty() {
                let keep: Vec<usize> = (0..metadata.num_row_groups())
                    .filter(|i| !skip_row_groups.contains(i))
                    .collect();
                builder = builder.with_row_groups(keep);
            }
        }
        if let Some(proj) = &self.projection {
            let mask = ProjectionMask::leaves(builder.parquet_schema(), proj.iter().copied());
            builder = builder.with_projection(mask);
        }

        let reader = builder.build().map_err(to_basalt_err)?;
        // See read_file's comment on the same point: the reader's schema
        // reflects projection and row-group filtering; the builder's does not.
        let basalt_schema = Arc::new(arrow_schema_to_basalt(reader.schema().as_ref())?);

        let mut batches = Vec::new();
        for batch in reader {
            let batch = batch.map_err(to_basalt_err)?;
            let columns = (0..batch.num_columns())
                .map(|i| arrow_array_to_basalt(batch.column(i).as_ref()))
                .collect::<Result<Vec<_>>>()?;
            batches.push(ColumnarBatch::try_new(basalt_schema.clone(), columns)?);
        }
        if batches.is_empty() {
            // Every row group was skipped (or the file has none): still need
            // one correctly-typed empty column per field, not zero columns —
            // this is exactly the `row_group_predicate_prunes_when_statistics_
            // prove_no_match` regression test below.
            let columns = basalt_schema
                .fields()
                .iter()
                .map(|f| empty_array_for(f.data_type))
                .collect();
            return ColumnarBatch::try_new(basalt_schema, columns);
        }
        let num_columns = basalt_schema.len();
        let mut columns = Vec::with_capacity(num_columns);
        for col_idx in 0..num_columns {
            let parts: Vec<ArrayRef> = batches
                .iter()
                .map(|b| Arc::clone(b.column(col_idx).unwrap()))
                .collect();
            columns.push(crate::compute::concat::concat(&parts)?);
        }
        ColumnarBatch::try_new(basalt_schema, columns)
    }
}

// `RowSelection` isn't used yet (page-level skipping, §9's tier 3) — kept
// imported to signal the seam is deliberately here for that follow-up.
#[allow(unused_imports)]
use RowSelection as _;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_primitive;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::types::schema::Field;

    fn sample_batch() -> ColumnarBatch {
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("score", DataType::Float64, true),
                Field::new("name", DataType::Utf8, true),
                Field::new("active", DataType::Boolean, false),
            ])
            .unwrap(),
        );
        let mut id = PrimitiveBuilder::<Int64Type>::with_capacity(3);
        id.append_value(1);
        id.append_value(2);
        id.append_value(3);
        let mut score = PrimitiveBuilder::<Float64Type>::with_capacity(3);
        score.append_value(1.5);
        score.append_null();
        score.append_value(3.5);
        let mut name = StringBuilder::with_capacity(3, 8);
        name.append_value("a").unwrap();
        name.append_null();
        name.append_value("c").unwrap();
        let mut active = BooleanBuilder::with_capacity(3);
        active.append_value(true);
        active.append_value(false);
        active.append_value(true);
        ColumnarBatch::try_new(
            schema,
            vec![
                Arc::new(id.finish()),
                Arc::new(score.finish()),
                Arc::new(name.finish()),
                Arc::new(active.finish()),
            ],
        )
        .unwrap()
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "basalt_parquet_test_{name}_{}.parquet",
            std::process::id()
        ))
    }

    #[test]
    fn round_trips_every_supported_type_with_and_without_nulls() {
        let path = temp_path("round_trip");
        let batch = sample_batch();
        write_file(&path, &batch).unwrap();
        let read_back = read_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(read_back.num_rows(), 3);
        let id = as_primitive::<Int64Type>(read_back.column(0).unwrap().as_ref()).unwrap();
        assert_eq!(id.value(0), 1);
        assert!(read_back.column(1).unwrap().is_null(1));
        assert!(read_back.column(2).unwrap().is_null(1));
    }

    #[test]
    fn projection_reads_only_requested_columns() {
        let path = temp_path("projection");
        write_file(&path, &sample_batch()).unwrap();
        let read_back = read_file(&path, Some(&[0, 3])).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(read_back.num_columns(), 2);
        assert_eq!(read_back.schema().field(0).unwrap().name, "id");
        assert_eq!(read_back.schema().field(1).unwrap().name, "active");
    }

    #[test]
    fn discover_schema_matches_written_schema() {
        let path = temp_path("schema");
        write_file(&path, &sample_batch()).unwrap();
        let schema = ParquetScanConfig::discover_schema(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(schema.len(), 4);
        assert_eq!(schema.field(0).unwrap().data_type, DataType::Int64);
    }

    #[test]
    fn row_group_predicate_prunes_when_statistics_prove_no_match() {
        let path = temp_path("pushdown");
        // ids 1..=3, so `id > 100` can never match this file's single row group.
        write_file(&path, &sample_batch()).unwrap();

        let config = ParquetScanConfig {
            path: path.clone(),
            schema: ParquetScanConfig::discover_schema(&path).unwrap(),
            projection: None,
            predicate: Some(RowGroupPredicate {
                column_index: 0,
                op: BinaryOp::Gt,
                value: ScalarValue::Int64(Some(100)),
            }),
        };
        let result = config.execute().unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn row_group_predicate_keeps_matching_row_group() {
        let path = temp_path("pushdown_keep");
        write_file(&path, &sample_batch()).unwrap();

        let config = ParquetScanConfig {
            path: path.clone(),
            schema: ParquetScanConfig::discover_schema(&path).unwrap(),
            projection: None,
            predicate: Some(RowGroupPredicate {
                column_index: 0,
                op: BinaryOp::GtEq,
                value: ScalarValue::Int64(Some(1)),
            }),
        };
        let result = config.execute().unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn missing_statistics_conservatively_scans() {
        // No statistics object at all: must return true (can't prove exclusion).
        assert!(row_group_may_match(
            None,
            &RowGroupPredicate {
                column_index: 0,
                op: BinaryOp::Gt,
                value: ScalarValue::Int64(Some(1))
            }
        ));
    }
}
