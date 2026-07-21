//! Equi-depth histograms. See design-docs/basalt-phase3-lld.md §2.3.
//!
//! Equi-depth (each bucket holds ~the same number of rows) rather than
//! equi-width (each bucket spans ~the same value range): equi-width
//! collapses on skewed data — if 90% of values fall in one range, one bucket
//! holds 90% of the rows and every estimate inside it is uniform garbage.
//! Equi-depth spends resolution where the data actually is, matching
//! Postgres/Oracle/SQL Server.

use crate::array::array::Array;
use crate::error::{BasaltError, Result};
use crate::scalar::ScalarValue;

#[derive(Clone, Debug)]
pub struct Bucket {
    /// Inclusive.
    pub lower: ScalarValue,
    /// Exclusive, except the final bucket (inclusive, so the max value is
    /// covered).
    pub upper: ScalarValue,
    /// Rows in `[lower, upper]`.
    pub count: usize,
    /// Distinct values within the bucket — stored per-bucket, not just per
    /// column, because equality selectivity inside a bucket
    /// (`count / distinct_count`) needs the *local* NDV. Using the
    /// column-wide NDV instead assumes distinct values are spread uniformly
    /// across buckets, exactly the assumption skewed data violates.
    pub distinct_count: usize,
}

#[derive(Clone, Debug)]
pub struct Histogram {
    pub buckets: Vec<Bucket>,
    pub total_count: usize,
    pub null_count: usize,
}

fn scalar_to_f64(v: &ScalarValue) -> Result<f64> {
    match v {
        ScalarValue::Int64(Some(i)) => Ok(*i as f64),
        ScalarValue::Float64(Some(f)) => Ok(*f),
        _ => Err(BasaltError::Internal(format!(
            "histogram interpolation requires a numeric, non-null scalar, got {v:?}"
        ))),
    }
}

fn cmp_scalar(a: &ScalarValue, b: &ScalarValue) -> Result<std::cmp::Ordering> {
    match (a, b) {
        (ScalarValue::Int64(Some(a)), ScalarValue::Int64(Some(b))) => Ok(a.cmp(b)),
        (ScalarValue::Float64(Some(a)), ScalarValue::Float64(Some(b))) => {
            a.partial_cmp(b).ok_or_else(|| {
                BasaltError::Internal("cannot order NaN in histogram comparison".to_string())
            })
        }
        (ScalarValue::Utf8(Some(a)), ScalarValue::Utf8(Some(b))) => Ok(a.cmp(b)),
        (ScalarValue::Boolean(Some(a)), ScalarValue::Boolean(Some(b))) => Ok(a.cmp(b)),
        _ => Err(BasaltError::Internal(format!(
            "cannot compare {a:?} and {b:?} for histogram bounds"
        ))),
    }
}

impl Histogram {
    /// Builds an equi-depth histogram from already-sorted, non-null values,
    /// by walking the array and emitting a bucket boundary roughly every
    /// `len / num_buckets` rows. Sampling before calling this (reservoir or
    /// block-level) is the caller's job for very large columns — a full
    /// sort of a billion-row column to build a histogram isn't acceptable;
    /// mark the result `Inexact` when sampled.
    ///
    /// # Errors
    /// Errors if `values` is empty, or contains non-comparable/non-numeric
    /// scalars this histogram implementation doesn't order.
    pub fn from_sorted(values: &dyn Array, num_buckets: usize) -> Result<Self> {
        let len = values.len();
        if len == 0 {
            return Err(BasaltError::Internal(
                "cannot build a histogram from an empty column".to_string(),
            ));
        }
        let num_buckets = num_buckets.max(1).min(len);
        let get = |i: usize| -> Result<ScalarValue> { array_scalar_at(values, i) };

        let rows_per_bucket = len.div_ceil(num_buckets);
        let mut buckets = Vec::with_capacity(num_buckets);
        let mut start = 0;
        while start < len {
            let end = (start + rows_per_bucket).min(len);
            let lower = get(start)?;
            let upper = get(end - 1)?;
            let mut distinct = 0usize;
            let mut prev: Option<ScalarValue> = None;
            for i in start..end {
                let v = get(i)?;
                if prev.as_ref() != Some(&v) {
                    distinct += 1;
                    prev = Some(v);
                }
            }
            buckets.push(Bucket {
                lower,
                upper,
                count: end - start,
                distinct_count: distinct,
            });
            start = end;
        }

        Ok(Histogram {
            buckets,
            total_count: len,
            null_count: 0,
        })
    }

    /// Estimated fraction of rows with value < `v`.
    ///
    /// # Errors
    /// Errors if `v` can't be ordered against this histogram's bounds.
    pub fn less_than(&self, v: &ScalarValue) -> Result<f64> {
        if self.total_count == 0 {
            return Ok(0.0);
        }
        let mut rows_below = 0.0;
        for bucket in &self.buckets {
            if cmp_scalar(v, &bucket.lower)? != std::cmp::Ordering::Greater {
                break;
            }
            if cmp_scalar(v, &bucket.upper)? == std::cmp::Ordering::Greater {
                rows_below += bucket.count as f64;
                continue;
            }
            // `v` straddles this bucket: linearly interpolate its position
            // within the bucket's numeric range.
            rows_below += interpolate_fraction(v, bucket)? * bucket.count as f64;
            break;
        }
        Ok((rows_below / self.total_count as f64).clamp(0.0, 1.0))
    }

    /// Estimated fraction of rows in `[lo, hi]`.
    ///
    /// # Errors
    /// Errors if `lo`/`hi` can't be ordered against this histogram's bounds.
    pub fn range(&self, lo: &ScalarValue, hi: &ScalarValue) -> Result<f64> {
        Ok((self.less_than(hi)? - self.less_than(lo)?).clamp(0.0, 1.0))
    }

    /// Estimated fraction of rows equal to `v`, using the bucket's local
    /// distinct count (`bucket.count / bucket.distinct_count`) rather than
    /// the column-wide NDV.
    ///
    /// # Errors
    /// Errors if `v` can't be ordered against this histogram's bounds.
    pub fn equals(&self, v: &ScalarValue) -> Result<f64> {
        if self.total_count == 0 {
            return Ok(0.0);
        }
        // A value can span several equi-depth buckets (e.g. one value
        // dominant enough to fill multiple bucket-widths of rows), so this
        // sums every matching bucket's local estimate rather than
        // returning on the first match.
        let mut matched_rows = 0.0;
        for bucket in &self.buckets {
            if cmp_scalar(v, &bucket.lower)? != std::cmp::Ordering::Less
                && cmp_scalar(v, &bucket.upper)? != std::cmp::Ordering::Greater
            {
                let distinct = bucket.distinct_count.max(1);
                matched_rows += bucket.count as f64 / distinct as f64;
            }
        }
        Ok((matched_rows / self.total_count as f64).clamp(0.0, 1.0))
    }
}

fn interpolate_fraction(v: &ScalarValue, bucket: &Bucket) -> Result<f64> {
    let (lo, hi, x) = (
        scalar_to_f64(&bucket.lower)?,
        scalar_to_f64(&bucket.upper)?,
        scalar_to_f64(v)?,
    );
    if (hi - lo).abs() < f64::EPSILON {
        return Ok(0.5);
    }
    Ok(((x - lo) / (hi - lo)).clamp(0.0, 1.0))
}

fn array_scalar_at(array: &dyn Array, i: usize) -> Result<ScalarValue> {
    use crate::array::array::{as_boolean, as_primitive, as_string};
    use crate::array::types::{Float64Type, Int64Type};
    use crate::types::data_type::DataType;

    if array.is_null(i) {
        return Err(BasaltError::Internal(
            "histogram construction requires non-null values".to_string(),
        ));
    }
    Ok(match array.data_type() {
        DataType::Int64 => ScalarValue::Int64(Some(as_primitive::<Int64Type>(array)?.value(i))),
        DataType::Float64 => {
            ScalarValue::Float64(Some(as_primitive::<Float64Type>(array)?.value(i)))
        }
        DataType::Utf8 => ScalarValue::Utf8(Some(as_string(array)?.value(i).to_string())),
        DataType::Boolean => ScalarValue::Boolean(Some(as_boolean(array)?.value(i))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;

    fn sorted_int_array(values: &[i64]) -> impl Array {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in values {
            b.append_value(v);
        }
        b.finish()
    }

    #[test]
    fn equi_depth_splits_rows_evenly_not_by_value_range() {
        // Skewed data: 90 rows at value 1, 10 rows spread 2..=11. Equi-width
        // would put almost everything in one bucket; equi-depth splits by
        // row count regardless of the value skew.
        let mut values: Vec<i64> = vec![1; 90];
        values.extend(2..=11);
        let arr = sorted_int_array(&values);
        let hist = Histogram::from_sorted(&arr, 10).unwrap();
        assert_eq!(hist.total_count, 100);
        for bucket in &hist.buckets {
            assert!(
                bucket.count <= 10,
                "bucket count {} exceeds 10",
                bucket.count
            );
        }
    }

    #[test]
    fn range_and_less_than_are_monotonic_and_bounded() {
        let arr = sorted_int_array(&(0..100).collect::<Vec<_>>());
        let hist = Histogram::from_sorted(&arr, 10).unwrap();
        let lt_0 = hist.less_than(&ScalarValue::Int64(Some(0))).unwrap();
        let lt_50 = hist.less_than(&ScalarValue::Int64(Some(50))).unwrap();
        let lt_100 = hist.less_than(&ScalarValue::Int64(Some(100))).unwrap();
        assert!(lt_0 <= lt_50);
        assert!(lt_50 <= lt_100);
        assert!((0.0..=1.0).contains(&lt_50));
        assert!(lt_100 > 0.9);
    }

    #[test]
    fn equals_uses_bucket_local_distinct_count() {
        // A single repeated value should estimate close to 1.0 within its
        // bucket, not smeared thin by a large column-wide NDV.
        let mut values = vec![7i64; 50];
        values.extend(8..=57);
        let arr = sorted_int_array(&values);
        let hist = Histogram::from_sorted(&arr, 10).unwrap();
        let sel = hist.equals(&ScalarValue::Int64(Some(7))).unwrap();
        assert!(
            sel > 0.3,
            "expected high selectivity for a dominant value, got {sel}"
        );
    }

    #[test]
    fn empty_column_errors_instead_of_panicking() {
        let arr = sorted_int_array(&[]);
        assert!(Histogram::from_sorted(&arr, 10).is_err());
    }
}
