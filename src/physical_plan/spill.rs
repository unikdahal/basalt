//! External sort building blocks: memory accounting and spill-to-disk batch
//! serialization. See design-docs/basalt-phase2-lld.md §8.3.
//!
//! **What's here vs. what's wired up.** `MemoryReservation` and
//! `write_batch`/`read_batch` (a spill file is genuinely "dead simple,
//! length-prefixed serialized batches," per the LLD) are complete and
//! tested standalone. Splicing them into `SortExec` itself — accumulate
//! until the budget trips, sort and spill each run, k-way merge the runs —
//! is the next integration step, not done in this pass; `SortExec` today
//! always buffers its full input in memory, same as `AggregateExec` and
//! `HashJoinExec` already do. Building the tested primitive first and
//! wiring the pipeline second mirrors this codebase's `TopKExec`: the
//! bounded-heap algorithm is documented as a follow-up there for the same
//! reason — get the piece right in isolation before threading it through
//! a pipeline breaker.
//!
//! **Spill encoding reuses `aggregate::group_keys::GroupKeyEncoder`**
//! wholesale: a batch's rows, treated as one "group key" per row across all
//! of the batch's columns, is exactly the byte-with-null-flag-prefix
//! encoding a spill file needs. No second encoding scheme to write or trust.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::array::array::{as_boolean, as_primitive, as_string, ArrayRef};
use crate::array::boolean::BooleanBuilder;
use crate::array::primitive::PrimitiveBuilder;
use crate::array::string::StringBuilder;
use crate::array::types::{Float64Type, Int64Type};
use crate::batch::ColumnarBatch;
use crate::error::{BasaltError, Result};
use crate::physical_plan::aggregate::group_keys::GroupKeyEncoder;
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;
use crate::types::schema::SchemaRef;

/// A per-operator memory budget. A minimal reservation now; a proper
/// `MemoryPool` with cross-operator budgets and fair eviction is a Phase 5
/// concern per the LLD — this is deliberately the narrow thing.
pub struct MemoryReservation {
    limit: usize,
    used: AtomicUsize,
}

impl MemoryReservation {
    pub fn new(limit: usize) -> Self {
        MemoryReservation {
            limit,
            used: AtomicUsize::new(0),
        }
    }

    /// # Errors
    /// Errors (the caller's cue to spill) if granting `bytes` would exceed
    /// the budget.
    pub fn try_grow(&self, bytes: usize) -> Result<()> {
        let current = self.used.load(Ordering::Relaxed);
        let new_total = current
            .checked_add(bytes)
            .ok_or_else(|| BasaltError::Internal("memory reservation overflow".to_string()))?;
        if new_total > self.limit {
            return Err(BasaltError::Internal(format!(
                "memory reservation exceeded: {new_total} > limit {}",
                self.limit
            )));
        }
        self.used.store(new_total, Ordering::Relaxed);
        Ok(())
    }

    pub fn shrink(&self, bytes: usize) {
        self.used.fetch_sub(
            bytes.min(self.used.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
    }

    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    pub fn limit(&self) -> usize {
        self.limit
    }
}

/// Writes one batch to `w` in a dead-simple, length-prefixed format: this
/// crate controls both the writer and reader, so there's no need for a
/// portable or versioned format — see the module doc.
///
/// # Errors
/// Errors on I/O failure or if a column's values fail to encode.
pub fn write_batch(w: &mut impl Write, batch: &ColumnarBatch) -> Result<()> {
    let encoder = GroupKeyEncoder::new(
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.data_type)
            .collect(),
    );
    let columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| std::sync::Arc::clone(batch.column(i).unwrap()))
        .collect();

    let mut encoded = Vec::new();
    let mut offsets = Vec::new();
    encoder.encode(&columns, &mut encoded, &mut offsets)?;

    w.write_all(&(batch.num_rows() as u64).to_le_bytes())?;
    w.write_all(&(encoded.len() as u64).to_le_bytes())?;
    w.write_all(&encoded)?;
    for &offset in &offsets {
        w.write_all(&offset.to_le_bytes())?;
    }
    Ok(())
}

/// Reads one batch previously written by [`write_batch`].
///
/// # Errors
/// Errors on I/O failure, truncated/corrupt data, or if a decoded value
/// doesn't match `schema`'s declared column types.
pub fn read_batch(r: &mut impl Read, schema: SchemaRef) -> Result<ColumnarBatch> {
    let num_rows = read_u64(r)? as usize;
    let encoded_len = read_u64(r)? as usize;
    let mut encoded = vec![0u8; encoded_len];
    r.read_exact(&mut encoded)?;
    let mut offsets = vec![0u32; num_rows + 1];
    for offset in &mut offsets {
        *offset = read_u32(r)?;
    }

    let types: Vec<DataType> = schema.fields().iter().map(|f| f.data_type).collect();
    let encoder = GroupKeyEncoder::new(types.clone());

    // column-major accumulation of the row-major decoded values
    let mut columns: Vec<Vec<ScalarValue>> = vec![Vec::with_capacity(num_rows); types.len()];
    for row in 0..num_rows {
        let key = &encoded[offsets[row] as usize..offsets[row + 1] as usize];
        let values = encoder.decode(key)?;
        for (col, value) in values.into_iter().enumerate() {
            columns[col].push(value);
        }
    }

    let arrays = columns
        .into_iter()
        .zip(&types)
        .map(|(values, &data_type)| scalars_to_array(&values, data_type))
        .collect::<Result<Vec<_>>>()?;
    ColumnarBatch::try_new(schema, arrays)
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn scalars_to_array(values: &[ScalarValue], data_type: DataType) -> Result<ArrayRef> {
    use std::sync::Arc;
    Ok(match data_type {
        DataType::Int64 => {
            let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
            for v in values {
                match v {
                    ScalarValue::Int64(Some(x)) => b.append_value(*x),
                    ScalarValue::Int64(None) => b.append_null(),
                    other => {
                        return Err(BasaltError::Internal(format!(
                            "expected Int64, got {other:?}"
                        )))
                    }
                }
            }
            Arc::new(b.finish())
        }
        DataType::Float64 => {
            let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(values.len());
            for v in values {
                match v {
                    ScalarValue::Float64(Some(x)) => b.append_value(*x),
                    ScalarValue::Float64(None) => b.append_null(),
                    other => {
                        return Err(BasaltError::Internal(format!(
                            "expected Float64, got {other:?}"
                        )))
                    }
                }
            }
            Arc::new(b.finish())
        }
        DataType::Utf8 => {
            let mut b = StringBuilder::with_capacity(values.len(), 0);
            for v in values {
                match v {
                    ScalarValue::Utf8(Some(x)) => b.append_value(x)?,
                    ScalarValue::Utf8(None) => b.append_null(),
                    other => {
                        return Err(BasaltError::Internal(format!(
                            "expected Utf8, got {other:?}"
                        )))
                    }
                }
            }
            Arc::new(b.finish())
        }
        DataType::Boolean => {
            let mut b = BooleanBuilder::with_capacity(values.len());
            for v in values {
                match v {
                    ScalarValue::Boolean(Some(x)) => b.append_value(*x),
                    ScalarValue::Boolean(None) => b.append_null(),
                    other => {
                        return Err(BasaltError::Internal(format!(
                            "expected Boolean, got {other:?}"
                        )))
                    }
                }
            }
            Arc::new(b.finish())
        }
    })
}

// Referenced only to keep the downcast helpers' imports honest if this
// module grows type-specific spill paths later.
#[allow(unused_imports)]
use {as_boolean as _, as_primitive as _, as_string as _};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::{as_primitive, Array};
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::types::schema::{Field, Schema};
    use std::sync::Arc;

    fn schema() -> SchemaRef {
        Arc::new(
            Schema::new(vec![
                Field::new("a", DataType::Int64, true),
                Field::new("b", DataType::Utf8, true),
            ])
            .unwrap(),
        )
    }

    fn sample_batch() -> ColumnarBatch {
        let mut a = PrimitiveBuilder::<Int64Type>::with_capacity(3);
        a.append_value(1);
        a.append_null();
        a.append_value(3);
        let mut b = crate::array::string::StringBuilder::with_capacity(3, 8);
        b.append_value("x").unwrap();
        b.append_value("y").unwrap();
        b.append_null();
        ColumnarBatch::try_new(schema(), vec![Arc::new(a.finish()), Arc::new(b.finish())]).unwrap()
    }

    #[test]
    fn write_then_read_round_trips_values_and_nulls() {
        let batch = sample_batch();
        let mut buf = Vec::new();
        write_batch(&mut buf, &batch).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let read_back = read_batch(&mut cursor, schema()).unwrap();

        assert_eq!(read_back.num_rows(), 3);
        let a = as_primitive::<Int64Type>(read_back.column(0).unwrap().as_ref()).unwrap();
        assert_eq!(a.value(0), 1);
        assert!(a.is_null(1));
        assert_eq!(a.value(2), 3);
        assert!(read_back.column(1).unwrap().is_null(2));
    }

    #[test]
    fn multiple_batches_can_be_written_sequentially_and_read_back_in_order() {
        let mut buf = Vec::new();
        write_batch(&mut buf, &sample_batch()).unwrap();
        write_batch(&mut buf, &sample_batch()).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let first = read_batch(&mut cursor, schema()).unwrap();
        let second = read_batch(&mut cursor, schema()).unwrap();
        assert_eq!(first.num_rows(), 3);
        assert_eq!(second.num_rows(), 3);
    }

    #[test]
    fn empty_batch_round_trips() {
        let empty = ColumnarBatch::try_new(
            schema(),
            vec![
                Arc::new(PrimitiveBuilder::<Int64Type>::with_capacity(0).finish()) as ArrayRef,
                Arc::new(crate::array::string::StringBuilder::with_capacity(0, 0).finish())
                    as ArrayRef,
            ],
        )
        .unwrap();
        let mut buf = Vec::new();
        write_batch(&mut buf, &empty).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let read_back = read_batch(&mut cursor, schema()).unwrap();
        assert_eq!(read_back.num_rows(), 0);
    }

    #[test]
    fn memory_reservation_errors_when_budget_exceeded() {
        let res = MemoryReservation::new(100);
        res.try_grow(60).unwrap();
        assert!(res.try_grow(60).is_err()); // would be 120 > 100
        assert_eq!(res.used(), 60); // failed grow doesn't partially apply
    }

    #[test]
    fn memory_reservation_shrink_frees_budget_for_reuse() {
        let res = MemoryReservation::new(100);
        res.try_grow(80).unwrap();
        res.shrink(80);
        assert_eq!(res.used(), 0);
        assert!(res.try_grow(100).is_ok());
    }

    #[test]
    fn memory_reservation_shrink_does_not_underflow_below_zero() {
        let res = MemoryReservation::new(100);
        res.try_grow(10).unwrap();
        res.shrink(1000); // shrinking by more than used must not panic/underflow
        assert_eq!(res.used(), 0);
    }
}
