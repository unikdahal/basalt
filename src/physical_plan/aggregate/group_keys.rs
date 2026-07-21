//! `GroupKeyEncoder` — encodes group-by values into flat, hashable,
//! comparable byte keys. See design-docs/basalt-phase2-lld.md §6.2.
//!
//! **Why a packed byte row, not `Vec<ScalarValue>`.** A `Vec<ScalarValue>`
//! key means a heap allocation per row and a `Vec` comparison per hash-table
//! probe — the single biggest performance decision this module makes is to
//! avoid that. A byte key makes comparison a `memcmp`.
//!
//! **Numeric encodings are order-preserving** (big-endian with the sign bit
//! handled so negative values sort before positive ones) specifically so
//! the same encoder can serve sort keys in module 2.6 — that reuse is why
//! it's done this way rather than any encoding that merely round-trips.
//!
//! **The `Utf8` encoding is *not* order-preserving across different-length
//! strings** (it's a length-prefix + raw bytes scheme) — good enough for
//! this module's actual need, which is hashing/equality for grouping, not
//! ordering. A truly order-preserving variable-length encoding needs an
//! escape scheme for embedded length markers; that's a real gap if this
//! encoder is later reused for sorting a `Utf8` column, and is called out
//! here rather than silently assumed to work.

use crate::array::array::{as_boolean, as_primitive, as_string, Array, ArrayRef};
use crate::array::types::{Float64Type, Int64Type};
use crate::error::{BasaltError, Result};
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;

pub struct GroupKeyEncoder {
    types: Vec<DataType>,
}

impl GroupKeyEncoder {
    pub fn new(types: Vec<DataType>) -> Self {
        GroupKeyEncoder { types }
    }

    /// Append each row's encoded key into `out`, with `offsets` delimiting
    /// rows (`offsets` gets `arrays[0].len() + 1` entries, like
    /// `StringArray`'s offsets — row `i` occupies `out[offsets[i]..offsets[i+1]]`).
    ///
    /// # Errors
    /// Errors if `arrays`' types don't match `self.types`, or if the arrays
    /// have differing lengths.
    pub fn encode(
        &self,
        arrays: &[ArrayRef],
        out: &mut Vec<u8>,
        offsets: &mut Vec<u32>,
    ) -> Result<()> {
        let num_rows = arrays.first().map_or(0, |a| a.len());
        for a in arrays {
            if a.len() != num_rows {
                return Err(BasaltError::Internal(
                    "group key arrays have differing lengths".to_string(),
                ));
            }
        }
        offsets.push(out.len() as u32);
        for row in 0..num_rows {
            for (col, array) in arrays.iter().enumerate() {
                encode_value(array.as_ref(), row, self.types[col], out)?;
            }
            offsets.push(out.len() as u32);
        }
        Ok(())
    }

    /// # Errors
    /// Errors if `key`'s length doesn't match what `self.types` implies.
    pub fn decode(&self, mut key: &[u8]) -> Result<Vec<ScalarValue>> {
        let mut values = Vec::with_capacity(self.types.len());
        for &data_type in &self.types {
            let (value, rest) = decode_value(key, data_type)?;
            values.push(value);
            key = rest;
        }
        Ok(values)
    }
}

fn encode_value(
    array: &dyn Array,
    row: usize,
    data_type: DataType,
    out: &mut Vec<u8>,
) -> Result<()> {
    if array.is_null(row) {
        out.push(0); // null flag: absent
        return Ok(());
    }
    out.push(1); // null flag: present
    match data_type {
        DataType::Int64 => {
            let v = as_primitive::<Int64Type>(array)?.value(row);
            // Flipping the sign bit makes the big-endian byte order match
            // numeric order across the full i64 range, negatives included.
            let encoded = (v as u64) ^ 0x8000_0000_0000_0000;
            out.extend_from_slice(&encoded.to_be_bytes());
        }
        DataType::Float64 => {
            let v = as_primitive::<Float64Type>(array)?.value(row);
            out.extend_from_slice(&encode_f64_order_preserving(v).to_be_bytes());
        }
        DataType::Boolean => {
            out.push(as_boolean(array)?.value(row) as u8);
        }
        DataType::Utf8 => {
            let s = as_string(array)?.value(row);
            let len = u32::try_from(s.len())
                .map_err(|_| BasaltError::Internal("group key string too long".to_string()))?;
            out.extend_from_slice(&len.to_be_bytes());
            out.extend_from_slice(s.as_bytes());
        }
    }
    Ok(())
}

fn decode_value(key: &[u8], data_type: DataType) -> Result<(ScalarValue, &[u8])> {
    let (&null_flag, rest) = key
        .split_first()
        .ok_or_else(|| BasaltError::Internal("group key truncated".to_string()))?;
    if null_flag == 0 {
        return Ok((null_scalar(data_type), rest));
    }
    match data_type {
        DataType::Int64 => {
            let (bytes, rest) = split_at(rest, 8)?;
            let encoded = u64::from_be_bytes(bytes.try_into().unwrap());
            let v = (encoded ^ 0x8000_0000_0000_0000) as i64;
            Ok((ScalarValue::Int64(Some(v)), rest))
        }
        DataType::Float64 => {
            let (bytes, rest) = split_at(rest, 8)?;
            let encoded = u64::from_be_bytes(bytes.try_into().unwrap());
            Ok((
                ScalarValue::Float64(Some(decode_f64_order_preserving(encoded))),
                rest,
            ))
        }
        DataType::Boolean => {
            let (&b, rest) = rest
                .split_first()
                .ok_or_else(|| BasaltError::Internal("group key truncated".to_string()))?;
            Ok((ScalarValue::Boolean(Some(b != 0)), rest))
        }
        DataType::Utf8 => {
            let (len_bytes, rest) = split_at(rest, 4)?;
            let len = u32::from_be_bytes(len_bytes.try_into().unwrap()) as usize;
            let (str_bytes, rest) = split_at(rest, len)?;
            let s = std::str::from_utf8(str_bytes).map_err(|e| {
                BasaltError::Internal(format!("group key string is not valid UTF-8: {e}"))
            })?;
            Ok((ScalarValue::Utf8(Some(s.to_string())), rest))
        }
    }
}

fn split_at(bytes: &[u8], n: usize) -> Result<(&[u8], &[u8])> {
    if bytes.len() < n {
        return Err(BasaltError::Internal("group key truncated".to_string()));
    }
    Ok(bytes.split_at(n))
}

fn null_scalar(data_type: DataType) -> ScalarValue {
    match data_type {
        DataType::Int64 => ScalarValue::Int64(None),
        DataType::Float64 => ScalarValue::Float64(None),
        DataType::Utf8 => ScalarValue::Utf8(None),
        DataType::Boolean => ScalarValue::Boolean(None),
    }
}

/// Maps `f64` to a `u64` whose big-endian byte order matches IEEE-754 total
/// order (negatives sort before positives, `-0.0` before `0.0`, and this
/// deliberately does not special-case `NaN` — it lands wherever its bit
/// pattern happens to sort, consistent with this project's documented
/// "no special NaN handling at the encoding layer" stance elsewhere).
fn encode_f64_order_preserving(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits & (1u64 << 63) != 0 {
        !bits
    } else {
        bits | (1u64 << 63)
    }
}

fn decode_f64_order_preserving(encoded: u64) -> f64 {
    let bits = if encoded & (1u64 << 63) != 0 {
        encoded & !(1u64 << 63)
    } else {
        !encoded
    };
    f64::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::boolean::BooleanBuilder;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::string::StringBuilder;
    use std::sync::Arc;

    #[test]
    fn int64_round_trips_through_encode_decode() {
        let encoder = GroupKeyEncoder::new(vec![DataType::Int64]);
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(1);
        b.append_value(-42);
        let arrays: Vec<ArrayRef> = vec![Arc::new(b.finish())];
        let mut out = Vec::new();
        let mut offsets = Vec::new();
        encoder.encode(&arrays, &mut out, &mut offsets).unwrap();
        let key = &out[offsets[0] as usize..offsets[1] as usize];
        assert_eq!(
            encoder.decode(key).unwrap(),
            vec![ScalarValue::Int64(Some(-42))]
        );
    }

    #[test]
    fn int64_encoding_preserves_numeric_order_including_negatives() {
        let encoder = GroupKeyEncoder::new(vec![DataType::Int64]);
        let values = [i64::MIN, -100, -1, 0, 1, 100, i64::MAX];
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(values.len());
        for &v in &values {
            b.append_value(v);
        }
        let arrays: Vec<ArrayRef> = vec![Arc::new(b.finish())];
        let mut out = Vec::new();
        let mut offsets = Vec::new();
        encoder.encode(&arrays, &mut out, &mut offsets).unwrap();

        let keys: Vec<&[u8]> = (0..values.len())
            .map(|i| &out[offsets[i] as usize..offsets[i + 1] as usize])
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(
            keys, sorted,
            "encoded keys should already be in ascending order"
        );
    }

    #[test]
    fn float64_encoding_preserves_numeric_order() {
        let encoder = GroupKeyEncoder::new(vec![DataType::Float64]);
        let values = [f64::NEG_INFINITY, -1.5, -0.0, 0.0, 1.5, f64::INFINITY];
        let mut b = PrimitiveBuilder::<Float64Type>::with_capacity(values.len());
        for &v in &values {
            b.append_value(v);
        }
        let arrays: Vec<ArrayRef> = vec![Arc::new(b.finish())];
        let mut out = Vec::new();
        let mut offsets = Vec::new();
        encoder.encode(&arrays, &mut out, &mut offsets).unwrap();
        let keys: Vec<&[u8]> = (0..values.len())
            .map(|i| &out[offsets[i] as usize..offsets[i + 1] as usize])
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn null_values_encode_and_decode_distinctly() {
        let encoder = GroupKeyEncoder::new(vec![DataType::Int64]);
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(1);
        b.append_null();
        let arrays: Vec<ArrayRef> = vec![Arc::new(b.finish())];
        let mut out = Vec::new();
        let mut offsets = Vec::new();
        encoder.encode(&arrays, &mut out, &mut offsets).unwrap();
        let key = &out[offsets[0] as usize..offsets[1] as usize];
        assert_eq!(encoder.decode(key).unwrap(), vec![ScalarValue::Int64(None)]);
    }

    #[test]
    fn string_and_boolean_round_trip() {
        let encoder = GroupKeyEncoder::new(vec![DataType::Utf8, DataType::Boolean]);
        let mut sb = StringBuilder::with_capacity(1, 8);
        sb.append_value("hello").unwrap();
        let mut bb = BooleanBuilder::with_capacity(1);
        bb.append_value(true);
        let arrays: Vec<ArrayRef> = vec![Arc::new(sb.finish()), Arc::new(bb.finish())];
        let mut out = Vec::new();
        let mut offsets = Vec::new();
        encoder.encode(&arrays, &mut out, &mut offsets).unwrap();
        let key = &out[offsets[0] as usize..offsets[1] as usize];
        assert_eq!(
            encoder.decode(key).unwrap(),
            vec![
                ScalarValue::Utf8(Some("hello".to_string())),
                ScalarValue::Boolean(Some(true))
            ]
        );
    }

    #[test]
    fn multiple_rows_produce_distinct_keys_for_distinct_values() {
        let encoder = GroupKeyEncoder::new(vec![DataType::Int64]);
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(3);
        b.append_value(1);
        b.append_value(2);
        b.append_value(1);
        let arrays: Vec<ArrayRef> = vec![Arc::new(b.finish())];
        let mut out = Vec::new();
        let mut offsets = Vec::new();
        encoder.encode(&arrays, &mut out, &mut offsets).unwrap();
        let key0 = &out[offsets[0] as usize..offsets[1] as usize];
        let key1 = &out[offsets[1] as usize..offsets[2] as usize];
        let key2 = &out[offsets[2] as usize..offsets[3] as usize];
        assert_eq!(key0, key2, "same value must produce the same key");
        assert_ne!(key0, key1, "different values must produce different keys");
    }
}
