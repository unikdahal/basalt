//! `concat` — combine several arrays of the same type into one.
//!
//! Not named as its own module in the LLD, but implied by "collect all
//! batches" in both `SortExec` (§8) and hash join's build phase (§7.1):
//! a pipeline breaker accumulates several batches, and its column-wise
//! algorithms (sort, hash-table probing) need one contiguous array per
//! column, not several. This is that seam.

use std::sync::Arc;

use crate::array::array::{as_boolean, as_primitive, as_string, Array, ArrayRef};
use crate::array::boolean::BooleanBuilder;
use crate::array::primitive::PrimitiveBuilder;
use crate::array::string::StringBuilder;
use crate::array::types::{Float64Type, Int64Type};
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;

/// # Errors
/// Errors if `arrays` is empty, or if they don't all share the same
/// `DataType`.
pub fn concat(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    let Some(first) = arrays.first() else {
        return Err(BasaltError::Internal(
            "concat requires at least one array".to_string(),
        ));
    };
    let data_type = first.data_type();
    for a in arrays {
        if a.data_type() != data_type {
            return Err(BasaltError::Internal(format!(
                "concat requires matching types, found {} and {data_type}",
                a.data_type()
            )));
        }
    }
    if arrays.len() == 1 {
        return Ok(Arc::clone(first));
    }

    let total_len: usize = arrays.iter().map(|a| a.len()).sum();
    Ok(match data_type {
        DataType::Int64 => {
            let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(total_len);
            for a in arrays {
                let a = as_primitive::<Int64Type>(a.as_ref())?;
                for i in 0..a.len() {
                    if a.is_null(i) {
                        b.append_null();
                    } else {
                        b.append_value(a.value(i));
                    }
                }
            }
            Arc::new(b.finish())
        }
        DataType::Float64 => {
            let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(total_len);
            for a in arrays {
                let a = as_primitive::<Float64Type>(a.as_ref())?;
                for i in 0..a.len() {
                    if a.is_null(i) {
                        b.append_null();
                    } else {
                        b.append_value(a.value(i));
                    }
                }
            }
            Arc::new(b.finish())
        }
        DataType::Utf8 => {
            let mut b = StringBuilder::with_capacity(total_len, 0);
            for a in arrays {
                let a = as_string(a.as_ref())?;
                for i in 0..a.len() {
                    if a.is_null(i) {
                        b.append_null();
                    } else {
                        b.append_value(a.value(i))?;
                    }
                }
            }
            Arc::new(b.finish())
        }
        DataType::Boolean => {
            let mut b = BooleanBuilder::with_capacity(total_len);
            for a in arrays {
                let a = as_boolean(a.as_ref())?;
                for i in 0..a.len() {
                    if a.is_null(i) {
                        b.append_null();
                    } else {
                        b.append_value(a.value(i));
                    }
                }
            }
            Arc::new(b.finish())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_primitive;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;

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
    fn concat_preserves_order_and_nulls() {
        let result = concat(&[int_array(&[Some(1), None]), int_array(&[Some(3)])]).unwrap();
        let result = as_primitive::<Int64Type>(result.as_ref()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.value(0), 1);
        assert!(result.is_null(1));
        assert_eq!(result.value(2), 3);
    }

    #[test]
    fn single_array_is_returned_without_copying() {
        let arr = int_array(&[Some(1), Some(2)]);
        let result = concat(std::slice::from_ref(&arr)).unwrap();
        assert!(Arc::ptr_eq(&arr, &result));
    }

    #[test]
    fn empty_input_errors() {
        assert!(concat(&[]).is_err());
    }

    #[test]
    fn mismatched_types_error() {
        let mut b = crate::array::primitive::PrimitiveBuilder::<Float64Type>::with_capacity(1);
        b.append_value(1.0);
        let float_arr: ArrayRef = Arc::new(b.finish());
        assert!(concat(&[int_array(&[Some(1)]), float_arr]).is_err());
    }
}
