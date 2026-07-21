//! Vectorized compute kernels operating on whole batches. See
//! design-docs/basalt-phase2-lld.md §4.

pub mod arith;
pub mod boolean;
pub mod cast;
pub mod comparison;
pub mod filter;
pub mod index;
pub mod take;

use std::sync::Arc;

use crate::array::array::ArrayRef;
use crate::array::boolean::BooleanBuilder;
use crate::array::primitive::PrimitiveBuilder;
use crate::array::string::StringBuilder;
use crate::array::types::{Float64Type, Int64Type};
use crate::error::Result;
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;

/// The result of evaluating an expression over a batch: either a full array,
/// or a single scalar that applies to every row.
///
/// **Keeping scalars unmaterialized is the point.** In `col_a + 5`, the
/// literal `5` must not become a full-length array — kernels read the
/// scalar once and loop over the array operand, which saves an allocation
/// and is faster (no second memory stream, better cache behavior). This
/// pass's kernels (see `arith.rs`'s module doc) don't yet fully exploit
/// that for every op — `into_array` is called more than the LLD's ideal —
/// but the type itself preserves the distinction so that optimization can
/// land later without an API change.
#[derive(Clone, Debug)]
pub enum ColumnarValue {
    Array(ArrayRef),
    Scalar(ScalarValue),
}

impl ColumnarValue {
    pub fn data_type(&self) -> DataType {
        match self {
            ColumnarValue::Array(a) => a.data_type(),
            ColumnarValue::Scalar(s) => s.data_type(),
        }
    }

    /// Materialize a scalar into an array of `num_rows`, repeating the
    /// value; an existing array is returned as-is (an `Arc` clone, not a
    /// copy). Use only when unavoidable — see the module-level note above.
    pub fn into_array(self, num_rows: usize) -> Result<ArrayRef> {
        match self {
            ColumnarValue::Array(a) => Ok(a),
            ColumnarValue::Scalar(s) => scalar_to_array(&s, num_rows),
        }
    }
}

// Manual `PartialEq`: `Arc<dyn Array>` has no general equality (comparing
// two arrays for value-equality is itself a kernel, not a language
// primitive), so array/array is identity comparison and any Array/Scalar
// mix is never equal. Good enough for tests; a real value-equality kernel
// is a `compute::comparison` concern, not this type's job.
impl PartialEq for ColumnarValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ColumnarValue::Scalar(a), ColumnarValue::Scalar(b)) => a == b,
            (ColumnarValue::Array(a), ColumnarValue::Array(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

fn scalar_to_array(scalar: &ScalarValue, num_rows: usize) -> Result<ArrayRef> {
    Ok(match scalar {
        ScalarValue::Int64(v) => {
            let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(num_rows);
            for _ in 0..num_rows {
                match v {
                    Some(x) => b.append_value(*x),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ScalarValue::Float64(v) => {
            let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(num_rows);
            for _ in 0..num_rows {
                match v {
                    Some(x) => b.append_value(*x),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ScalarValue::Utf8(v) => {
            let mut b = StringBuilder::with_capacity(num_rows, 0);
            for _ in 0..num_rows {
                match v {
                    Some(x) => b.append_value(x)?,
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ScalarValue::Boolean(v) => {
            let mut b = BooleanBuilder::with_capacity(num_rows);
            for _ in 0..num_rows {
                match v {
                    Some(x) => b.append_value(*x),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::{as_primitive, Array};

    #[test]
    fn scalar_into_array_repeats_the_value() {
        let cv = ColumnarValue::Scalar(ScalarValue::Int64(Some(7)));
        let arr = cv.into_array(3).unwrap();
        let arr = as_primitive::<Int64Type>(arr.as_ref()).unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value(0), 7);
        assert_eq!(arr.value(2), 7);
    }

    #[test]
    fn null_scalar_into_array_is_all_null() {
        let cv = ColumnarValue::Scalar(ScalarValue::Utf8(None));
        let arr = cv.into_array(2).unwrap();
        assert!(arr.is_null(0));
        assert!(arr.is_null(1));
    }

    #[test]
    fn array_into_array_is_a_no_op() {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(1);
        b.append_value(1);
        let original: ArrayRef = Arc::new(b.finish());
        let cv = ColumnarValue::Array(Arc::clone(&original));
        let result = cv.into_array(1).unwrap();
        assert!(Arc::ptr_eq(&original, &result));
    }

    #[test]
    fn data_type_reflects_the_underlying_value() {
        assert_eq!(
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))).data_type(),
            DataType::Boolean
        );
    }
}
