//! Statistics collection. See design-docs/basalt-phase3-lld.md §2.6.
//!
//! Two sources, and the second is free: `analyze` runs a full (or sampled)
//! scan to compute histograms/NDV/MCVs — real work, real I/O. Parquet
//! footers already carry per-row-group min/max/null-count per column;
//! reading them costs one small I/O and gives exact min/max/null counts and
//! row counts across the whole file with **no scan of the data at all**.
//! What footers *don't* give you: NDV, histograms, MCVs — those need
//! `analyze`.

use std::path::Path;
use std::sync::Arc;

use super::histogram::Histogram;
use super::hll::HyperLogLog;
use super::mcv::MostCommonValues;
use super::precision::Precision;
use super::stats::{ColumnStatistics, TableStatistics};
use crate::array::array::Array;
use crate::batch::ColumnarBatch;
use crate::compute::sort::{lexsort_to_indices, SortColumn, SortOptions};
use crate::compute::take::take;
use crate::error::Result;
use crate::logical_plan::TableSource;
use crate::physical_plan::scan::MemoryTableSource;
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;

pub trait StatisticsProvider {
    fn table_statistics(&self, table: &str) -> Result<TableStatistics>;
}

/// Full or sampled scan over an in-memory source, computing real
/// histograms, NDV (via HyperLogLog), and MCV lists per column.
///
/// `sample`, if given, is the fraction of rows to examine (a simple stride
/// sample — every `1/sample`-th row); the result is marked `Inexact`
/// whenever sampling was used, since it's derived from a subset rather than
/// counted exactly.
///
/// # Errors
/// Errors if `source` isn't backed by in-memory batches (Basalt's
/// `TableSource` trait doesn't yet expose a generic scan API — this reads
/// via `MemoryTableSource` specifically) or a column's type isn't
/// supported.
///
/// Only `Int64`/`Float64`/`Utf8`/`Boolean` are supported (this crate's
/// closed `DataType` lattice), and histograms only accept the two numeric
/// types today — a `Utf8`/`Boolean` column still gets NDV, min/max, and an
/// MCV list, just no histogram.
pub fn analyze(source: &dyn TableSource, sample: Option<f64>) -> Result<TableStatistics> {
    let memory_source = source
        .as_any()
        .downcast_ref::<MemoryTableSource>()
        .ok_or_else(|| {
            crate::error::BasaltError::Internal(
            "ANALYZE requires a MemoryTableSource in Phase 3 core (no generic TableSource scan \
             API exists yet)"
                .to_string(),
        )
        })?;

    let schema = source.schema();
    let num_columns = schema.fields().len();
    if memory_source.batches().is_empty() {
        return Ok(TableStatistics::unknown(num_columns));
    }

    let batch = concat_batches(memory_source.batches())?;
    let precision_kind = |exact: bool| {
        move |v| {
            if exact {
                Precision::Exact(v)
            } else {
                Precision::Inexact(v)
            }
        }
    };

    let mut column_statistics = Vec::with_capacity(num_columns);
    let mut total_rows = 0usize;
    for col_idx in 0..num_columns {
        let column = batch.column(col_idx).ok_or_else(|| {
            crate::error::BasaltError::Internal(format!("column index {col_idx} out of bounds"))
        })?;
        let (sampled_column, exact) = maybe_sample(column.as_ref(), sample)?;
        total_rows = total_rows.max(sampled_column.len());
        column_statistics.push(analyze_column(
            sampled_column.as_ref(),
            precision_kind(exact),
        )?);
    }

    Ok(TableStatistics {
        num_rows: Precision::Exact(batch.num_rows()),
        total_byte_size: Precision::Absent,
        column_statistics,
    })
}

fn maybe_sample(
    column: &dyn Array,
    sample: Option<f64>,
) -> Result<(std::sync::Arc<dyn Array>, bool)> {
    let Some(fraction) = sample else {
        return Ok((column.slice(0, column.len()), true));
    };
    let fraction = fraction.clamp(0.0, 1.0);
    if fraction >= 1.0 || column.is_empty() {
        return Ok((column.slice(0, column.len()), fraction >= 1.0));
    }
    let stride = (1.0 / fraction).round().max(1.0) as usize;
    let mut indices =
        crate::compute::index::UInt32Builder::with_capacity(column.len() / stride + 1);
    let mut i = 0usize;
    while i < column.len() {
        indices.append_value(i as u32);
        i += stride;
    }
    Ok((take(column, &indices.finish())?, false))
}

fn analyze_column(
    column: &dyn Array,
    precision: impl Fn(usize) -> Precision<usize>,
) -> Result<ColumnStatistics> {
    let null_count = column.null_count();
    let mut hll = HyperLogLog::new(12);
    for i in 0..column.len() {
        if column.is_null(i) {
            continue;
        }
        hash_value_into(column, i, &mut hll)?;
    }
    let ndv = hll.estimate();

    let (min_value, max_value) = min_max(column)?;
    let histogram = sorted_non_null_array(column)
        .and_then(|arr| Histogram::from_sorted(arr.as_ref(), 100).ok());
    let mcv = build_mcv(column, 10)?;

    Ok(ColumnStatistics {
        null_count: precision(null_count),
        distinct_count: precision(ndv),
        min_value: min_value.map_or(Precision::Absent, |v| precision(1).map(|_| v)),
        max_value: max_value.map_or(Precision::Absent, |v| precision(1).map(|_| v)),
        histogram,
        mcv,
    })
}

fn hash_value_into(column: &dyn Array, i: usize, hll: &mut HyperLogLog) -> Result<()> {
    use crate::array::array::{as_boolean, as_primitive, as_string};
    use crate::array::types::{Float64Type, Int64Type};
    match column.data_type() {
        DataType::Int64 => hll.add(&as_primitive::<Int64Type>(column)?.value(i)),
        DataType::Float64 => hll.add(&as_primitive::<Float64Type>(column)?.value(i).to_bits()),
        DataType::Utf8 => hll.add(&as_string(column)?.value(i)),
        DataType::Boolean => hll.add(&as_boolean(column)?.value(i)),
    }
    Ok(())
}

fn min_max(column: &dyn Array) -> Result<(Option<ScalarValue>, Option<ScalarValue>)> {
    use crate::array::array::{as_boolean, as_primitive, as_string};
    use crate::array::types::{Float64Type, Int64Type};

    let non_null: Vec<usize> = (0..column.len()).filter(|&i| !column.is_null(i)).collect();
    if non_null.is_empty() {
        return Ok((None, None));
    }

    Ok(match column.data_type() {
        DataType::Int64 => {
            let a = as_primitive::<Int64Type>(column)?;
            let (mut min, mut max) = (a.value(non_null[0]), a.value(non_null[0]));
            for &i in &non_null {
                let v = a.value(i);
                min = min.min(v);
                max = max.max(v);
            }
            (
                Some(ScalarValue::Int64(Some(min))),
                Some(ScalarValue::Int64(Some(max))),
            )
        }
        DataType::Float64 => {
            let a = as_primitive::<Float64Type>(column)?;
            let (mut min, mut max) = (a.value(non_null[0]), a.value(non_null[0]));
            for &i in &non_null {
                let v = a.value(i);
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            (
                Some(ScalarValue::Float64(Some(min))),
                Some(ScalarValue::Float64(Some(max))),
            )
        }
        DataType::Utf8 => {
            let a = as_string(column)?;
            let (mut min, mut max) = (a.value(non_null[0]), a.value(non_null[0]));
            for &i in &non_null {
                let v = a.value(i);
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            (
                Some(ScalarValue::Utf8(Some(min.to_string()))),
                Some(ScalarValue::Utf8(Some(max.to_string()))),
            )
        }
        DataType::Boolean => {
            let a = as_boolean(column)?;
            let (mut min, mut max) = (true, false);
            for &i in &non_null {
                let v = a.value(i);
                min &= v;
                max |= v;
            }
            (
                Some(ScalarValue::Boolean(Some(min))),
                Some(ScalarValue::Boolean(Some(max))),
            )
        }
    })
}

/// Sorts the non-null values of a numeric column, for histogram
/// construction. Returns `None` for non-numeric types (`Histogram` only
/// supports `Int64`/`Float64`) or an all-null column.
fn sorted_non_null_array(column: &dyn Array) -> Option<Arc<dyn Array>> {
    if !matches!(column.data_type(), DataType::Int64 | DataType::Float64) {
        return None;
    }
    let non_null: Vec<u32> = (0..column.len())
        .filter(|&i| !column.is_null(i))
        .map(|i| i as u32)
        .collect();
    if non_null.is_empty() {
        return None;
    }
    let mut idx_builder = crate::compute::index::UInt32Builder::with_capacity(non_null.len());
    for i in non_null {
        idx_builder.append_value(i);
    }
    let non_null_array = take(column, &idx_builder.finish()).ok()?;

    let sort_indices = lexsort_to_indices(&[SortColumn {
        values: Arc::clone(&non_null_array),
        options: SortOptions {
            descending: false,
            nulls_first: true,
        },
    }])
    .ok()?;
    take(non_null_array.as_ref(), &sort_indices).ok()
}

fn build_mcv(column: &dyn Array, top_n: usize) -> Result<Option<MostCommonValues>> {
    use std::collections::HashMap;
    let len = column.len();
    if len == 0 {
        return Ok(None);
    }
    let mut counts: HashMap<String, (ScalarValue, usize)> = HashMap::new();
    for i in 0..len {
        if column.is_null(i) {
            continue;
        }
        let v = scalar_at(column, i)?;
        let key = format!("{v:?}");
        counts.entry(key).or_insert_with(|| (v, 0)).1 += 1;
    }
    if counts.is_empty() {
        return Ok(None);
    }
    let mut pairs: Vec<(ScalarValue, usize)> = counts.into_values().collect();
    pairs.sort_by_key(|b| std::cmp::Reverse(b.1));
    pairs.truncate(top_n);

    let values = pairs.iter().map(|(v, _)| v.clone()).collect();
    let frequencies = pairs.iter().map(|(_, c)| *c as f64 / len as f64).collect();
    Ok(Some(MostCommonValues {
        values,
        frequencies,
    }))
}

fn scalar_at(column: &dyn Array, i: usize) -> Result<ScalarValue> {
    use crate::array::array::{as_boolean, as_primitive, as_string};
    use crate::array::types::{Float64Type, Int64Type};
    Ok(match column.data_type() {
        DataType::Int64 => ScalarValue::Int64(Some(as_primitive::<Int64Type>(column)?.value(i))),
        DataType::Float64 => {
            ScalarValue::Float64(Some(as_primitive::<Float64Type>(column)?.value(i)))
        }
        DataType::Utf8 => ScalarValue::Utf8(Some(as_string(column)?.value(i).to_string())),
        DataType::Boolean => ScalarValue::Boolean(Some(as_boolean(column)?.value(i))),
    })
}

fn concat_batches(batches: &[ColumnarBatch]) -> Result<ColumnarBatch> {
    if batches.len() == 1 {
        return Ok(batches[0].clone());
    }
    let schema = batches[0].schema();
    let num_columns = schema.fields().len();
    let mut columns = Vec::with_capacity(num_columns);
    for col_idx in 0..num_columns {
        let arrays: Vec<Arc<dyn Array>> = batches
            .iter()
            .map(|b| b.column(col_idx).cloned())
            .collect::<Option<_>>()
            .ok_or_else(|| {
                crate::error::BasaltError::Internal(format!("column index {col_idx} out of bounds"))
            })?;
        columns.push(crate::compute::concat::concat(&arrays)?);
    }
    ColumnarBatch::try_new(schema.clone(), columns)
}

/// Derives statistics from a Parquet file's footer metadata — **no data
/// scan at all**. Row counts and min/max/null-count per column come free;
/// NDV/histograms/MCVs are `Absent` (footers don't carry them; `analyze`
/// does).
///
/// # Errors
/// Errors if the file can't be opened or its metadata parsed.
pub fn statistics_from_parquet(path: impl AsRef<Path>) -> Result<TableStatistics> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| crate::error::BasaltError::Internal(e.to_string()))?;
    let metadata = builder.metadata();
    let num_columns = metadata.file_metadata().schema_descr().num_columns();

    let mut num_rows = 0usize;
    let mut null_counts = vec![0usize; num_columns];
    let mut mins: Vec<Option<ScalarValue>> = vec![None; num_columns];
    let mut maxes: Vec<Option<ScalarValue>> = vec![None; num_columns];

    for rg in 0..metadata.num_row_groups() {
        let row_group = metadata.row_group(rg);
        num_rows += row_group.num_rows() as usize;
        for col in 0..num_columns {
            let Some(stats) = row_group.column(col).statistics() else {
                continue;
            };
            null_counts[col] += stats.null_count_opt().unwrap_or(0) as usize;
            let (min, max) = parquet_stats_to_scalars(stats);
            merge_min(&mut mins[col], min);
            merge_max(&mut maxes[col], max);
        }
    }

    let column_statistics = (0..num_columns)
        .map(|col| ColumnStatistics {
            null_count: Precision::Exact(null_counts[col]),
            distinct_count: Precision::Absent,
            min_value: mins[col]
                .clone()
                .map_or(Precision::Absent, Precision::Exact),
            max_value: maxes[col]
                .clone()
                .map_or(Precision::Absent, Precision::Exact),
            histogram: None,
            mcv: None,
        })
        .collect();

    Ok(TableStatistics {
        num_rows: Precision::Exact(num_rows),
        total_byte_size: Precision::Absent,
        column_statistics,
    })
}

fn parquet_stats_to_scalars(
    stats: &parquet::file::statistics::Statistics,
) -> (Option<ScalarValue>, Option<ScalarValue>) {
    use parquet::file::statistics::Statistics as S;
    match stats {
        S::Int64(s) => (
            s.min_opt().map(|v| ScalarValue::Int64(Some(*v))),
            s.max_opt().map(|v| ScalarValue::Int64(Some(*v))),
        ),
        S::Double(s) => (
            s.min_opt().map(|v| ScalarValue::Float64(Some(*v))),
            s.max_opt().map(|v| ScalarValue::Float64(Some(*v))),
        ),
        S::ByteArray(s) => (
            s.min_opt()
                .and_then(|v| std::str::from_utf8(v.data()).ok())
                .map(|v| ScalarValue::Utf8(Some(v.to_string()))),
            s.max_opt()
                .and_then(|v| std::str::from_utf8(v.data()).ok())
                .map(|v| ScalarValue::Utf8(Some(v.to_string()))),
        ),
        S::Boolean(s) => (
            s.min_opt().map(|v| ScalarValue::Boolean(Some(*v))),
            s.max_opt().map(|v| ScalarValue::Boolean(Some(*v))),
        ),
        _ => (None, None),
    }
}

fn merge_min(acc: &mut Option<ScalarValue>, candidate: Option<ScalarValue>) {
    let Some(candidate) = candidate else { return };
    match acc {
        None => *acc = Some(candidate),
        Some(existing) => {
            if scalar_less_than(&candidate, existing) {
                *acc = Some(candidate);
            }
        }
    }
}

fn merge_max(acc: &mut Option<ScalarValue>, candidate: Option<ScalarValue>) {
    let Some(candidate) = candidate else { return };
    match acc {
        None => *acc = Some(candidate),
        Some(existing) => {
            if scalar_less_than(existing, &candidate) {
                *acc = Some(candidate);
            }
        }
    }
}

fn scalar_less_than(a: &ScalarValue, b: &ScalarValue) -> bool {
    match (a, b) {
        (ScalarValue::Int64(Some(a)), ScalarValue::Int64(Some(b))) => a < b,
        (ScalarValue::Float64(Some(a)), ScalarValue::Float64(Some(b))) => a < b,
        (ScalarValue::Utf8(Some(a)), ScalarValue::Utf8(Some(b))) => a < b,
        (ScalarValue::Boolean(Some(a)), ScalarValue::Boolean(Some(b))) => !a & b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::types::schema::{Field, Schema};

    fn source(values: &[i64]) -> MemoryTableSource {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap());
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            b.append_value(v);
        }
        let batch = ColumnarBatch::try_new(schema.clone(), vec![Arc::new(b.finish())]).unwrap();
        MemoryTableSource::new(schema, vec![batch])
    }

    #[test]
    fn analyze_computes_row_count_and_ndv() {
        let src = source(&(0..1000).collect::<Vec<_>>());
        let stats = analyze(&src, None).unwrap();
        assert_eq!(stats.num_rows, Precision::Exact(1000));
        let ndv = stats.column_statistics[0]
            .distinct_count
            .get_value()
            .copied();
        assert!(ndv.is_some());
        let ndv = ndv.unwrap();
        assert!(
            (900..=1100).contains(&ndv),
            "NDV estimate {ndv} too far from 1000"
        );
    }

    #[test]
    fn analyze_produces_min_max_and_histogram() {
        let src = source(&(0..500).collect::<Vec<_>>());
        let stats = analyze(&src, None).unwrap();
        let col = &stats.column_statistics[0];
        assert_eq!(
            col.min_value.get_value(),
            Some(&ScalarValue::Int64(Some(0)))
        );
        assert_eq!(
            col.max_value.get_value(),
            Some(&ScalarValue::Int64(Some(499)))
        );
        assert!(col.histogram.is_some());
    }

    #[test]
    fn analyze_on_empty_source_returns_unknown() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap());
        let src = MemoryTableSource::new(schema, vec![]);
        let stats = analyze(&src, None).unwrap();
        assert!(stats.num_rows.is_absent());
    }

    #[test]
    fn sampled_analyze_is_marked_inexact() {
        let src = source(&(0..1000).collect::<Vec<_>>());
        let stats = analyze(&src, Some(0.1)).unwrap();
        assert!(!stats.column_statistics[0].distinct_count.is_exact());
    }
}
