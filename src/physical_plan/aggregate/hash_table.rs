//! `GroupedHashAggregator` — the group-by hash table. See
//! design-docs/basalt-phase2-lld.md §6.3.
//!
//! **Indirection through a group index.** The hash map stores only
//! `key -> usize`; accumulator state lives in `Vec`s indexed by that
//! `usize`. This keeps the map itself small (better cache behavior on
//! probes) and keeps accumulator state columnar and contiguous, rather than
//! interleaved with the keys.
//!
//! Per aggregate, per batch, rows are grouped by which group they belong to
//! and gathered via `compute::take` before calling `Accumulator::update_batch`
//! once per group — the same "index arrays, then take" primitive joins and
//! sort use, rather than one `update_batch` call per row.

use std::collections::HashMap;

use super::accumulator::Accumulator;
use super::group_keys::GroupKeyEncoder;
use crate::array::array::ArrayRef;
use crate::compute::index::UInt32Builder;
use crate::compute::take::take;
use crate::error::Result;
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;

/// `(one row per group of decoded group-by values, one row per group per
/// aggregate of its final output)`.
type FinishedGroups = (Vec<Vec<ScalarValue>>, Vec<Vec<ScalarValue>>);

pub struct GroupedHashAggregator {
    key_encoder: GroupKeyEncoder,
    group_indices: HashMap<Box<[u8]>, usize>,
    group_keys: Vec<Box<[u8]>>,
    /// `[aggregate_idx][group_idx]`.
    accumulators: Vec<Vec<Box<dyn Accumulator>>>,
    factories: Vec<Box<dyn Fn() -> Box<dyn Accumulator> + Send + Sync>>,
}

impl GroupedHashAggregator {
    pub fn new(
        group_types: Vec<DataType>,
        factories: Vec<Box<dyn Fn() -> Box<dyn Accumulator> + Send + Sync>>,
    ) -> Self {
        let accumulators = factories.iter().map(|_| Vec::new()).collect();
        GroupedHashAggregator {
            key_encoder: GroupKeyEncoder::new(group_types),
            group_indices: HashMap::new(),
            group_keys: Vec::new(),
            accumulators,
            factories,
        }
    }

    pub fn num_groups(&self) -> usize {
        self.group_keys.len()
    }

    /// # Errors
    /// Errors if the group or aggregate-input arrays can't be encoded or
    /// gathered (e.g. a type the key encoder or an accumulator rejects).
    pub fn update_batch(
        &mut self,
        group_arrays: &[ArrayRef],
        agg_arrays: &[ArrayRef],
    ) -> Result<()> {
        let num_rows = group_arrays.first().map_or(0, |a| a.len());
        let mut encoded = Vec::new();
        let mut offsets = Vec::new();
        self.key_encoder
            .encode(group_arrays, &mut encoded, &mut offsets)?;

        let mut group_of_row = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            let key = &encoded[offsets[row] as usize..offsets[row + 1] as usize];
            let group_idx = match self.group_indices.get(key) {
                Some(&idx) => idx,
                None => {
                    let idx = self.group_keys.len();
                    self.group_keys.push(key.into());
                    self.group_indices.insert(key.into(), idx);
                    for (agg_idx, factory) in self.factories.iter().enumerate() {
                        self.accumulators[agg_idx].push(factory());
                    }
                    idx
                }
            };
            group_of_row.push(group_idx);
        }

        for (agg_idx, agg_array) in agg_arrays.iter().enumerate() {
            let mut rows_per_group: Vec<Vec<u32>> = vec![Vec::new(); self.group_keys.len()];
            for (row, &g) in group_of_row.iter().enumerate() {
                rows_per_group[g].push(row as u32);
            }
            for (g, rows) in rows_per_group.into_iter().enumerate() {
                if rows.is_empty() {
                    continue; // group existed before this batch but got no rows in it
                }
                let mut idx_builder = UInt32Builder::with_capacity(rows.len());
                for r in rows {
                    idx_builder.append_value(r);
                }
                let gathered = take(agg_array.as_ref(), &idx_builder.finish())?;
                self.accumulators[agg_idx][g].update_batch(&[gathered])?;
            }
        }
        Ok(())
    }

    /// Consumes the aggregator, producing one row per group: the decoded
    /// group-by key values, and each aggregate's final output.
    ///
    /// # Errors
    /// Errors if a group key fails to decode or an accumulator fails to
    /// finalize.
    pub fn finish(self) -> Result<FinishedGroups> {
        let mut group_rows = Vec::with_capacity(self.group_keys.len());
        for key in &self.group_keys {
            group_rows.push(self.key_encoder.decode(key)?);
        }
        let mut agg_outputs: Vec<Vec<ScalarValue>> = (0..self.accumulators.len())
            .map(|_| Vec::with_capacity(self.group_keys.len()))
            .collect();
        for (agg_idx, accs) in self.accumulators.into_iter().enumerate() {
            for acc in accs {
                agg_outputs[agg_idx].push(acc.evaluate()?);
            }
        }
        Ok((group_rows, agg_outputs))
    }
}

#[cfg(test)]
mod tests {
    use super::super::accumulator::{CountAccumulator, SumAccumulator};
    use super::*;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use std::sync::Arc;

    fn int_array(values: &[i64]) -> ArrayRef {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            b.append_value(v);
        }
        Arc::new(b.finish())
    }

    #[test]
    fn groups_rows_and_sums_per_group() {
        // group_key: [1, 2, 1, 2] ; value: [10, 20, 30, 40]
        // expect group 1 -> sum 40, group 2 -> sum 60
        let mut agg = GroupedHashAggregator::new(
            vec![DataType::Int64],
            vec![Box::new(|| {
                Box::new(SumAccumulator::new(DataType::Int64)) as Box<dyn Accumulator>
            })],
        );
        agg.update_batch(&[int_array(&[1, 2, 1, 2])], &[int_array(&[10, 20, 30, 40])])
            .unwrap();
        assert_eq!(agg.num_groups(), 2);

        let (group_rows, agg_outputs) = agg.finish().unwrap();
        let mut by_key: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
        for (row, sum) in group_rows.iter().zip(&agg_outputs[0]) {
            let ScalarValue::Int64(Some(key)) = row[0] else {
                panic!()
            };
            let ScalarValue::Int64(Some(s)) = sum else {
                panic!()
            };
            by_key.insert(key, *s);
        }
        assert_eq!(by_key.get(&1), Some(&40));
        assert_eq!(by_key.get(&2), Some(&60));
    }

    #[test]
    fn new_groups_across_multiple_batches_accumulate_correctly() {
        let mut agg = GroupedHashAggregator::new(
            vec![DataType::Int64],
            vec![Box::new(|| {
                Box::new(CountAccumulator::star()) as Box<dyn Accumulator>
            })],
        );
        agg.update_batch(&[int_array(&[1, 1])], &[int_array(&[0, 0])])
            .unwrap();
        agg.update_batch(&[int_array(&[1, 2])], &[int_array(&[0, 0])])
            .unwrap();
        assert_eq!(agg.num_groups(), 2);

        let (group_rows, agg_outputs) = agg.finish().unwrap();
        let mut by_key: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
        for (row, count) in group_rows.iter().zip(&agg_outputs[0]) {
            let ScalarValue::Int64(Some(key)) = row[0] else {
                panic!()
            };
            let ScalarValue::Int64(Some(c)) = count else {
                panic!()
            };
            by_key.insert(key, *c);
        }
        assert_eq!(by_key.get(&1), Some(&3));
        assert_eq!(by_key.get(&2), Some(&1));
    }

    #[test]
    fn empty_input_yields_zero_groups() {
        let agg = GroupedHashAggregator::new(
            vec![DataType::Int64],
            vec![Box::new(|| {
                Box::new(SumAccumulator::new(DataType::Int64)) as Box<dyn Accumulator>
            })],
        );
        assert_eq!(agg.num_groups(), 0);
        let (group_rows, agg_outputs) = agg.finish().unwrap();
        assert!(group_rows.is_empty());
        assert!(agg_outputs[0].is_empty());
    }
}
