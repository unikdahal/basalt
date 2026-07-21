//! `Accumulator` — folds batches into running aggregate state. See
//! design-docs/basalt-phase2-lld.md §6.1.
//!
//! `state()`/`merge_batch()` exist even though Phase 2 is single-threaded:
//! this is the two-phase aggregation contract that lets Phase 4 distribute
//! aggregation across workers without redesigning anything (partial states
//! computed per-partition, shuffled by group key, merged in a final phase).
//! `AVG` is why the contract has to look like this rather than a single
//! running value: its state (`sum`, `count`) is richer than its output
//! (`sum / count`) — you cannot merge two averages by averaging them.
//!
//! Null semantics: aggregates skip nulls (`SUM`/`MIN`/`MAX`/`AVG` of an
//! all-null input is `NULL`, not `0`); `COUNT(*)` counts rows, `COUNT(expr)`
//! counts non-null values of `expr`.

use crate::array::array::{as_boolean, as_primitive, as_string, Array, ArrayRef};
use crate::array::types::{Float64Type, Int64Type};
use crate::error::{BasaltError, Result};
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;

pub trait Accumulator: std::fmt::Debug + Send + Sync {
    /// Fold a batch of input values into this accumulator.
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()>;

    /// Merge partial states produced by other accumulators of the same kind
    /// (two-phase aggregation).
    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()>;

    /// The intermediate state, for shipping to a final-phase aggregator.
    fn state(&self) -> Result<Vec<ScalarValue>>;

    /// The final output value.
    fn evaluate(&self) -> Result<ScalarValue>;

    /// Approximate bytes held, for memory accounting.
    fn size(&self) -> usize;
}

fn scalar_at(array: &dyn Array, i: usize) -> Result<Option<ScalarValue>> {
    if array.is_null(i) {
        return Ok(None);
    }
    Ok(Some(match array.data_type() {
        DataType::Int64 => ScalarValue::Int64(Some(as_primitive::<Int64Type>(array)?.value(i))),
        DataType::Float64 => {
            ScalarValue::Float64(Some(as_primitive::<Float64Type>(array)?.value(i)))
        }
        DataType::Utf8 => ScalarValue::Utf8(Some(as_string(array)?.value(i).to_string())),
        DataType::Boolean => ScalarValue::Boolean(Some(as_boolean(array)?.value(i))),
    }))
}

fn scalar_cmp(a: &ScalarValue, b: &ScalarValue) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (ScalarValue::Int64(Some(x)), ScalarValue::Int64(Some(y))) => Some(x.cmp(y)),
        (ScalarValue::Float64(Some(x)), ScalarValue::Float64(Some(y))) => x.partial_cmp(y),
        (ScalarValue::Utf8(Some(x)), ScalarValue::Utf8(Some(y))) => Some(x.cmp(y)),
        (ScalarValue::Boolean(Some(x)), ScalarValue::Boolean(Some(y))) => Some(x.cmp(y)),
        _ => None,
    }
}

/// `SUM(expr)`. Overflow uses the same checked-arithmetic-errors discipline
/// as `compute::arith` and Phase 1's `expr::eval` — a silently wrapped SUM
/// is a silently wrong answer.
#[derive(Debug)]
pub struct SumAccumulator {
    data_type: DataType,
    sum_int: i64,
    sum_float: f64,
    has_value: bool,
}

impl SumAccumulator {
    pub fn new(data_type: DataType) -> Self {
        SumAccumulator {
            data_type,
            sum_int: 0,
            sum_float: 0.0,
            has_value: false,
        }
    }
}

impl Accumulator for SumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = values[0].as_ref();
        match self.data_type {
            DataType::Int64 => {
                let arr = as_primitive::<Int64Type>(array)?;
                for i in 0..arr.len() {
                    if arr.is_null(i) {
                        continue;
                    }
                    self.sum_int = self
                        .sum_int
                        .checked_add(arr.value(i))
                        .ok_or(BasaltError::NumericOverflow)?;
                    self.has_value = true;
                }
            }
            DataType::Float64 => {
                let arr = as_primitive::<Float64Type>(array)?;
                for i in 0..arr.len() {
                    if arr.is_null(i) {
                        continue;
                    }
                    self.sum_float += arr.value(i);
                    self.has_value = true;
                }
            }
            other => {
                return Err(BasaltError::Type {
                    message: format!("SUM does not support {other}"),
                });
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // SUM's state is just its own output type, so merging partial sums
        // is the same operation as summing raw values.
        self.update_batch(states)
    }

    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        if !self.has_value {
            return Ok(match self.data_type {
                DataType::Int64 => ScalarValue::Int64(None),
                DataType::Float64 => ScalarValue::Float64(None),
                other => {
                    return Err(BasaltError::Type {
                        message: format!("SUM does not support {other}"),
                    })
                }
            });
        }
        Ok(match self.data_type {
            DataType::Int64 => ScalarValue::Int64(Some(self.sum_int)),
            DataType::Float64 => ScalarValue::Float64(Some(self.sum_float)),
            _ => unreachable!("validated in update_batch"),
        })
    }

    fn size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// `COUNT(*)` (counts rows, ignoring nulls) or `COUNT(expr)` (counts
/// non-null values of `expr`). Never `NULL`, even over zero rows — `0`.
#[derive(Debug)]
pub struct CountAccumulator {
    count: i64,
    star: bool,
}

impl CountAccumulator {
    pub fn star() -> Self {
        CountAccumulator {
            count: 0,
            star: true,
        }
    }

    pub fn expr() -> Self {
        CountAccumulator {
            count: 0,
            star: false,
        }
    }
}

impl Accumulator for CountAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = values[0].as_ref();
        if self.star {
            self.count = self
                .count
                .checked_add(array.len() as i64)
                .ok_or(BasaltError::NumericOverflow)?;
        } else {
            for i in 0..array.len() {
                if !array.is_null(i) {
                    self.count = self
                        .count
                        .checked_add(1)
                        .ok_or(BasaltError::NumericOverflow)?;
                }
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let arr = as_primitive::<Int64Type>(states[0].as_ref())?;
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                self.count = self
                    .count
                    .checked_add(arr.value(i))
                    .ok_or(BasaltError::NumericOverflow)?;
            }
        }
        Ok(())
    }

    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Int64(Some(self.count))])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(ScalarValue::Int64(Some(self.count)))
    }

    fn size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// `MIN(expr)`/`MAX(expr)`, generic over any comparable type via `ScalarValue`.
#[derive(Debug)]
pub struct MinMaxAccumulator {
    data_type: DataType,
    current: Option<ScalarValue>,
    is_min: bool,
}

impl MinMaxAccumulator {
    pub fn min(data_type: DataType) -> Self {
        MinMaxAccumulator {
            data_type,
            current: None,
            is_min: true,
        }
    }

    pub fn max(data_type: DataType) -> Self {
        MinMaxAccumulator {
            data_type,
            current: None,
            is_min: false,
        }
    }

    fn consider(&mut self, candidate: ScalarValue) {
        let better = match &self.current {
            None => true,
            Some(current) => match scalar_cmp(&candidate, current) {
                Some(std::cmp::Ordering::Less) => self.is_min,
                Some(std::cmp::Ordering::Greater) => !self.is_min,
                _ => false,
            },
        };
        if better {
            self.current = Some(candidate);
        }
    }
}

impl Accumulator for MinMaxAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = values[0].as_ref();
        for i in 0..array.len() {
            if let Some(v) = scalar_at(array, i)? {
                self.consider(v);
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.update_batch(states)
    }

    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(self.current.clone().unwrap_or(match self.data_type {
            DataType::Int64 => ScalarValue::Int64(None),
            DataType::Float64 => ScalarValue::Float64(None),
            DataType::Utf8 => ScalarValue::Utf8(None),
            DataType::Boolean => ScalarValue::Boolean(None),
        }))
    }

    fn size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// `AVG(expr)`. The canonical example that makes the `Accumulator` design
/// necessary: state is `(sum: f64, count: i64)`, output is `sum / count` —
/// richer state than output, so `state()` cannot just return `evaluate()`
/// the way `SUM`/`COUNT`/`MIN`/`MAX` do.
#[derive(Debug, Default)]
pub struct AvgAccumulator {
    sum: f64,
    count: i64,
}

impl AvgAccumulator {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Accumulator for AvgAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = values[0].as_ref();
        match array.data_type() {
            DataType::Int64 => {
                let arr = as_primitive::<Int64Type>(array)?;
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        self.sum += arr.value(i) as f64;
                        self.count += 1;
                    }
                }
            }
            DataType::Float64 => {
                let arr = as_primitive::<Float64Type>(array)?;
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        self.sum += arr.value(i);
                        self.count += 1;
                    }
                }
            }
            other => {
                return Err(BasaltError::Type {
                    message: format!("AVG does not support {other}"),
                });
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let sums = as_primitive::<Float64Type>(states[0].as_ref())?;
        let counts = as_primitive::<Int64Type>(states[1].as_ref())?;
        for i in 0..sums.len() {
            if !sums.is_null(i) {
                self.sum += sums.value(i);
            }
        }
        for i in 0..counts.len() {
            if !counts.is_null(i) {
                self.count += counts.value(i);
            }
        }
        Ok(())
    }

    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            ScalarValue::Float64(Some(self.sum)),
            ScalarValue::Int64(Some(self.count)),
        ])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        if self.count == 0 {
            return Ok(ScalarValue::Float64(None));
        }
        Ok(ScalarValue::Float64(Some(self.sum / self.count as f64)))
    }

    fn size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive::PrimitiveBuilder;
    use std::sync::Arc;

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
    fn sum_skips_nulls_and_returns_null_for_all_null_input() {
        let mut acc = SumAccumulator::new(DataType::Int64);
        acc.update_batch(&[int_array(&[None, None])]).unwrap();
        assert_eq!(acc.evaluate().unwrap(), ScalarValue::Int64(None));

        let mut acc = SumAccumulator::new(DataType::Int64);
        acc.update_batch(&[int_array(&[Some(1), None, Some(2)])])
            .unwrap();
        assert_eq!(acc.evaluate().unwrap(), ScalarValue::Int64(Some(3)));
    }

    #[test]
    fn sum_overflow_errors() {
        let mut acc = SumAccumulator::new(DataType::Int64);
        acc.update_batch(&[int_array(&[Some(i64::MAX), Some(1)])])
            .unwrap_err();
    }

    #[test]
    fn sum_two_phase_merge_matches_single_phase_sum() {
        let mut partial_a = SumAccumulator::new(DataType::Int64);
        partial_a
            .update_batch(&[int_array(&[Some(1), Some(2)])])
            .unwrap();
        let mut partial_b = SumAccumulator::new(DataType::Int64);
        partial_b
            .update_batch(&[int_array(&[Some(3), Some(4)])])
            .unwrap();

        let mut final_acc = SumAccumulator::new(DataType::Int64);
        let state_a = partial_a.state().unwrap();
        let state_b = partial_b.state().unwrap();
        let combined = int_array(&[
            match state_a[0] {
                ScalarValue::Int64(v) => v,
                _ => panic!(),
            },
            match state_b[0] {
                ScalarValue::Int64(v) => v,
                _ => panic!(),
            },
        ]);
        final_acc.merge_batch(&[combined]).unwrap();

        let mut whole = SumAccumulator::new(DataType::Int64);
        whole
            .update_batch(&[int_array(&[Some(1), Some(2), Some(3), Some(4)])])
            .unwrap();

        assert_eq!(final_acc.evaluate().unwrap(), whole.evaluate().unwrap());
    }

    #[test]
    fn count_star_counts_rows_including_nulls() {
        let mut acc = CountAccumulator::star();
        acc.update_batch(&[int_array(&[Some(1), None, Some(2)])])
            .unwrap();
        assert_eq!(acc.evaluate().unwrap(), ScalarValue::Int64(Some(3)));
    }

    #[test]
    fn count_expr_skips_nulls() {
        let mut acc = CountAccumulator::expr();
        acc.update_batch(&[int_array(&[Some(1), None, Some(2)])])
            .unwrap();
        assert_eq!(acc.evaluate().unwrap(), ScalarValue::Int64(Some(2)));
    }

    #[test]
    fn count_over_empty_input_is_zero_not_null() {
        let acc = CountAccumulator::expr();
        assert_eq!(acc.evaluate().unwrap(), ScalarValue::Int64(Some(0)));
    }

    #[test]
    fn min_max_skip_nulls() {
        let mut min_acc = MinMaxAccumulator::min(DataType::Int64);
        min_acc
            .update_batch(&[int_array(&[Some(5), None, Some(2), Some(8)])])
            .unwrap();
        assert_eq!(min_acc.evaluate().unwrap(), ScalarValue::Int64(Some(2)));

        let mut max_acc = MinMaxAccumulator::max(DataType::Int64);
        max_acc
            .update_batch(&[int_array(&[Some(5), None, Some(2), Some(8)])])
            .unwrap();
        assert_eq!(max_acc.evaluate().unwrap(), ScalarValue::Int64(Some(8)));
    }

    #[test]
    fn min_over_all_null_input_is_null() {
        let mut acc = MinMaxAccumulator::min(DataType::Int64);
        acc.update_batch(&[int_array(&[None, None])]).unwrap();
        assert_eq!(acc.evaluate().unwrap(), ScalarValue::Int64(None));
    }

    /// The two-phase contract's reason for existing: averages cannot be
    /// merged by averaging them, only by summing sums and counts.
    #[test]
    fn avg_state_is_richer_than_output_and_merges_correctly() {
        let mut partial_a = AvgAccumulator::new();
        partial_a
            .update_batch(&[int_array(&[Some(10), Some(20)])])
            .unwrap(); // avg 15
        let mut partial_b = AvgAccumulator::new();
        partial_b.update_batch(&[int_array(&[Some(100)])]).unwrap(); // avg 100

        // Naively averaging 15 and 100 would give 57.5 — wrong. The correct
        // merged average of [10, 20, 100] is 130/3.
        let mut merged = AvgAccumulator::new();
        let sums = {
            let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(2);
            for state in [partial_a.state().unwrap(), partial_b.state().unwrap()] {
                match state[0] {
                    ScalarValue::Float64(Some(v)) => b.append_value(v),
                    _ => panic!(),
                }
            }
            Arc::new(b.finish()) as ArrayRef
        };
        let counts = {
            let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(2);
            for state in [partial_a.state().unwrap(), partial_b.state().unwrap()] {
                match state[1] {
                    ScalarValue::Int64(Some(v)) => b.append_value(v),
                    _ => panic!(),
                }
            }
            Arc::new(b.finish()) as ArrayRef
        };
        merged.merge_batch(&[sums, counts]).unwrap();

        assert_eq!(
            merged.evaluate().unwrap(),
            ScalarValue::Float64(Some(130.0 / 3.0))
        );
    }

    #[test]
    fn avg_over_all_null_input_is_null() {
        let acc = AvgAccumulator::new();
        assert_eq!(acc.evaluate().unwrap(), ScalarValue::Float64(None));
    }
}
