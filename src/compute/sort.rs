//! `lexsort_to_indices` — produce a permutation that sorts several columns
//! lexicographically. See design-docs/basalt-phase2-lld.md §8.1.
//!
//! **Sorts indices, then `take`s** — the same "index arrays + take" pattern
//! `filter`/joins use. Moving a `u32` per comparison is cheap; moving whole
//! rows (`String`s included) on every swap is not.
//!
//! **Column-wise comparator**, not the LLD's normalized-row-key alternative
//! (encode all sort columns into one `memcmp`-able byte string, 2–5× faster
//! per the LLD). Correctness first at a second bug-dense area of this
//! phase; the row-key path can reuse `aggregate::group_keys::GroupKeyEncoder`
//! almost as-is (that encoder's order-preserving numeric encoding was built
//! with exactly this reuse in mind) as a benchmarked follow-up.
//!
//! Uses a **stable sort** (`sort_by`, not `sort_unstable_by`) so multi-key
//! ordering and repeated sorts compose correctly.
//!
//! `SortOptions` here is a separate type from `logical_plan::SortOptions`
//! despite having the same shape — `compute` sits below `logical_plan` in
//! the module layering (`buffer -> array -> batch -> compute ->
//! physical_expr -> physical_plan`, with `logical_plan` sitting above
//! `batch`), so `compute` depending on it would be an upward dependency.
//! `physical_plan::sort` converts between the two at the boundary.

use crate::array::array::{as_boolean, as_primitive, as_string, ArrayRef};
use crate::array::types::{Float64Type, Int64Type};
use crate::buffer::bit_at;
use crate::compute::index::{UInt32Array, UInt32Builder};
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SortOptions {
    pub descending: bool,
    pub nulls_first: bool,
}

pub struct SortColumn {
    pub values: ArrayRef,
    pub options: SortOptions,
}

/// # Errors
/// Errors if the columns have differing lengths.
pub fn lexsort_to_indices(columns: &[SortColumn]) -> Result<UInt32Array> {
    let num_rows = columns.first().map_or(0, |c| c.values.len());
    for c in columns {
        if c.values.len() != num_rows {
            return Err(BasaltError::Internal(
                "sort columns have differing lengths".to_string(),
            ));
        }
    }

    // Downcasting and deriving each column's values/validity slice happens
    // here, once, rather than inside the `O(n log n)` comparator below —
    // `compare_at` used to re-downcast (`as_primitive`/`as_string`/
    // `as_boolean`) and re-derive `value(i)`/`value(j)` (each walking
    // `Buffer::as_slice()` from scratch) on *every* comparison call, the
    // same class of bug `compute::arith`'s `value(i)` had (see that
    // module's doc comment).
    let hoisted: Vec<HoistedColumn> = columns
        .iter()
        .map(HoistedColumn::new)
        .collect::<Result<_>>()?;

    let mut indices: Vec<u32> = (0..num_rows as u32).collect();
    indices.sort_by(|&i, &j| {
        for column in &hoisted {
            let ord = column.compare(i as usize, j as usize);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });

    let mut builder = UInt32Builder::with_capacity(indices.len());
    for i in indices {
        builder.append_value(i);
    }
    Ok(builder.finish())
}

/// Per-type hoisted access, built once per column before the sort's inner
/// loop runs. `Utf8` still calls `StringArray::value(i)` per comparison
/// (that type's own internal buffer re-derivation is a separate, un-fixed
/// cost — out of scope here), but at least pays the downcast once instead
/// of on every comparison.
enum HoistedValues<'a> {
    Int64(&'a [i64]),
    Float64(&'a [f64]),
    Utf8(&'a crate::array::string::StringArray),
    Boolean { bytes: &'a [u8], bit_offset: usize },
}

struct HoistedColumn<'a> {
    values: HoistedValues<'a>,
    validity: Option<(&'a [u8], usize)>,
    options: SortOptions,
}

impl<'a> HoistedColumn<'a> {
    fn new(column: &'a SortColumn) -> Result<Self> {
        let array = column.values.as_ref();
        let validity = array.validity().map(|v| (v.as_bytes(), v.bit_offset()));
        let values = match array.data_type() {
            DataType::Int64 => HoistedValues::Int64(as_primitive::<Int64Type>(array)?.values()),
            DataType::Float64 => {
                HoistedValues::Float64(as_primitive::<Float64Type>(array)?.values())
            }
            DataType::Utf8 => HoistedValues::Utf8(as_string(array)?),
            DataType::Boolean => {
                let b = as_boolean(array)?;
                HoistedValues::Boolean {
                    bytes: b.values().as_bytes(),
                    bit_offset: b.values().bit_offset(),
                }
            }
        };
        Ok(HoistedColumn {
            values,
            validity,
            options: column.options,
        })
    }

    fn is_null(&self, i: usize) -> bool {
        self.validity
            .is_some_and(|(bytes, offset)| !bit_at(bytes, offset, i))
    }

    fn compare(&self, i: usize, j: usize) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        let (i_null, j_null) = (self.is_null(i), self.is_null(j));
        if i_null || j_null {
            return match (i_null, j_null) {
                (true, true) => Equal,
                (true, false) => {
                    if self.options.nulls_first {
                        Less
                    } else {
                        Greater
                    }
                }
                (false, true) => {
                    if self.options.nulls_first {
                        Greater
                    } else {
                        Less
                    }
                }
                (false, false) => unreachable!(),
            };
        }

        let ord = match &self.values {
            HoistedValues::Int64(values) => values[i].cmp(&values[j]),
            HoistedValues::Float64(values) => {
                // NaN policy: sorts as greater than everything, including
                // itself compared to itself (Equal) — matching
                // `exec::ops::compare_values`'s documented Phase 1 policy so
                // sort behavior is consistent across both engine generations.
                let (x, y) = (values[i], values[j]);
                match (x.is_nan(), y.is_nan()) {
                    (true, true) => Equal,
                    (true, false) => Greater,
                    (false, true) => Less,
                    (false, false) => x.partial_cmp(&y).unwrap_or(Equal),
                }
            }
            HoistedValues::Utf8(a) => a.value(i).cmp(a.value(j)),
            HoistedValues::Boolean { bytes, bit_offset } => {
                bit_at(bytes, *bit_offset, i).cmp(&bit_at(bytes, *bit_offset, j))
            }
        };
        if self.options.descending {
            ord.reverse()
        } else {
            ord
        }
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

    fn indices_of(arr: &UInt32Array) -> Vec<u32> {
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    #[test]
    fn sorts_ascending_by_default() {
        let col = SortColumn {
            values: int_array(&[Some(3), Some(1), Some(2)]),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        };
        let result = lexsort_to_indices(&[col]).unwrap();
        assert_eq!(indices_of(&result), vec![1, 2, 0]); // values 1,2,3
    }

    #[test]
    fn descending_reverses_order() {
        let col = SortColumn {
            values: int_array(&[Some(3), Some(1), Some(2)]),
            options: SortOptions {
                descending: true,
                nulls_first: true,
            },
        };
        let result = lexsort_to_indices(&[col]).unwrap();
        assert_eq!(indices_of(&result), vec![0, 2, 1]); // values 3,2,1
    }

    #[test]
    fn nulls_first_and_nulls_last_policies() {
        let values = || int_array(&[Some(1), None, Some(2)]);

        let first = lexsort_to_indices(&[SortColumn {
            values: values(),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        }])
        .unwrap();
        assert_eq!(indices_of(&first)[0], 1); // the null comes first

        let last = lexsort_to_indices(&[SortColumn {
            values: values(),
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        }])
        .unwrap();
        assert_eq!(indices_of(&last)[2], 1); // the null comes last
    }

    #[test]
    fn multi_key_sort_breaks_ties_with_second_column() {
        // (1, 20), (1, 10), (0, 5)  sorted by col0 asc, col1 asc
        // -> (0,5), (1,10), (1,20)
        let col0 = SortColumn {
            values: int_array(&[Some(1), Some(1), Some(0)]),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        };
        let col1 = SortColumn {
            values: int_array(&[Some(20), Some(10), Some(5)]),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        };
        let result = lexsort_to_indices(&[col0, col1]).unwrap();
        assert_eq!(indices_of(&result), vec![2, 1, 0]);
    }

    #[test]
    fn nan_sorts_as_greater_than_everything() {
        let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(3);
        b.append_value(1.0);
        b.append_value(f64::NAN);
        b.append_value(-1.0);
        let col = SortColumn {
            values: Arc::new(b.finish()),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        };
        let result = lexsort_to_indices(&[col]).unwrap();
        assert_eq!(indices_of(&result), vec![2, 0, 1]); // -1, 1, NaN
    }

    #[test]
    fn stable_sort_preserves_relative_order_of_equal_keys() {
        // Two rows with the same key; a stable sort must keep their
        // original relative order (index 0 before index 2).
        let col = SortColumn {
            values: int_array(&[Some(5), Some(1), Some(5)]),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        };
        let result = lexsort_to_indices(&[col]).unwrap();
        assert_eq!(indices_of(&result), vec![1, 0, 2]);
    }

    #[test]
    fn mismatched_lengths_error() {
        let col0 = SortColumn {
            values: int_array(&[Some(1)]),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        };
        let col1 = SortColumn {
            values: int_array(&[Some(1), Some(2)]),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        };
        assert!(lexsort_to_indices(&[col0, col1]).is_err());
    }
}
