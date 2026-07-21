//! `NotExpr`/`NegExpr` — logical and arithmetic unary operators.

use std::any::Any;
use std::sync::Arc;

use super::expr::{PhysicalExpr, PhysicalExprRef};
use crate::array::array::{as_primitive, Array};
use crate::batch::ColumnarBatch;
use crate::compute::{boolean, ColumnarValue};
use crate::error::{BasaltError, Result};
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;
use crate::types::schema::Schema;

#[derive(Debug, Clone)]
pub struct NotExpr {
    pub expr: PhysicalExprRef,
}

impl NotExpr {
    pub fn new(expr: PhysicalExprRef) -> Self {
        NotExpr { expr }
    }
}

impl PhysicalExpr for NotExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        let t = self.expr.data_type(input_schema)?;
        if t != DataType::Boolean {
            return Err(BasaltError::Type {
                message: format!("cannot apply logical NOT to non-boolean type {t}"),
            });
        }
        Ok(DataType::Boolean)
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.expr.nullable(input_schema)
    }

    fn evaluate(&self, batch: &ColumnarBatch) -> Result<ColumnarValue> {
        match self.expr.evaluate(batch)? {
            ColumnarValue::Scalar(ScalarValue::Boolean(v)) => {
                Ok(ColumnarValue::Scalar(ScalarValue::Boolean(v.map(|b| !b))))
            }
            ColumnarValue::Scalar(other) => Err(BasaltError::Type {
                message: format!(
                    "cannot apply logical NOT to non-boolean type {}",
                    other.data_type()
                ),
            }),
            ColumnarValue::Array(a) => {
                let a = crate::array::array::as_boolean(a.as_ref())?;
                Ok(ColumnarValue::Array(Arc::new(boolean::not_kleene(a))))
            }
        }
    }

    fn children(&self) -> Vec<PhysicalExprRef> {
        vec![Arc::clone(&self.expr)]
    }
}

#[derive(Debug, Clone)]
pub struct NegExpr {
    pub expr: PhysicalExprRef,
}

impl NegExpr {
    pub fn new(expr: PhysicalExprRef) -> Self {
        NegExpr { expr }
    }
}

impl PhysicalExpr for NegExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        let t = self.expr.data_type(input_schema)?;
        if !t.is_numeric() {
            return Err(BasaltError::Type {
                message: format!("cannot apply unary negation (-) to non-numeric type {t}"),
            });
        }
        Ok(t)
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.expr.nullable(input_schema)
    }

    fn evaluate(&self, batch: &ColumnarBatch) -> Result<ColumnarValue> {
        match self.expr.evaluate(batch)? {
            ColumnarValue::Scalar(ScalarValue::Int64(v)) => {
                Ok(ColumnarValue::Scalar(ScalarValue::Int64(
                    v.map(|x| x.checked_neg().ok_or(BasaltError::NumericOverflow))
                        .transpose()?,
                )))
            }
            ColumnarValue::Scalar(ScalarValue::Float64(v)) => {
                Ok(ColumnarValue::Scalar(ScalarValue::Float64(v.map(|x| -x))))
            }
            ColumnarValue::Scalar(other) => Err(BasaltError::Type {
                message: format!("cannot negate non-numeric type {}", other.data_type()),
            }),
            ColumnarValue::Array(a) => match a.data_type() {
                DataType::Int64 => {
                    let src = as_primitive::<crate::array::types::Int64Type>(a.as_ref())?;
                    let mut builder = crate::array::primitive::PrimitiveBuilder::<
                        crate::array::types::Int64Type,
                    >::with_capacity(src.len());
                    for i in 0..src.len() {
                        if src.is_null(i) {
                            builder.append_null();
                        } else {
                            builder.append_value(
                                src.value(i)
                                    .checked_neg()
                                    .ok_or(BasaltError::NumericOverflow)?,
                            );
                        }
                    }
                    Ok(ColumnarValue::Array(Arc::new(builder.finish())))
                }
                DataType::Float64 => {
                    let src = as_primitive::<crate::array::types::Float64Type>(a.as_ref())?;
                    let mut builder = crate::array::primitive::PrimitiveBuilder::<
                        crate::array::types::Float64Type,
                    >::with_capacity(src.len());
                    for i in 0..src.len() {
                        if src.is_null(i) {
                            builder.append_null();
                        } else {
                            builder.append_value(-src.value(i));
                        }
                    }
                    Ok(ColumnarValue::Array(Arc::new(builder.finish())))
                }
                other => Err(BasaltError::Type {
                    message: format!("cannot apply unary negation (-) to non-numeric type {other}"),
                }),
            },
        }
    }

    fn children(&self) -> Vec<PhysicalExprRef> {
        vec![Arc::clone(&self.expr)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_primitive;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::physical_expr::column::ColumnExpr;
    use crate::physical_expr::literal::LiteralExpr;
    use crate::types::schema::Field;

    #[test]
    fn not_negates_boolean_array_and_preserves_null() {
        let schema = std::sync::Arc::new(
            Schema::new(vec![Field::new("a", DataType::Boolean, true)]).unwrap(),
        );
        let mut b = crate::array::boolean::BooleanBuilder::with_capacity(3);
        b.append_value(true);
        b.append_value(false);
        b.append_null();
        let batch = ColumnarBatch::try_new(schema, vec![Arc::new(b.finish())]).unwrap();

        let expr = NotExpr::new(Arc::new(ColumnExpr::new(0)));
        let result = expr.evaluate(&batch).unwrap();
        let result = match result {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let result = crate::array::array::as_boolean(result.as_ref()).unwrap();
        assert!(!result.value(0));
        assert!(result.value(1));
        assert!(result.is_null(2));
    }

    #[test]
    fn neg_negates_int_array_and_errors_on_min_overflow() {
        let schema = std::sync::Arc::new(
            Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap(),
        );
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(1);
        b.append_value(5);
        let batch = ColumnarBatch::try_new(schema, vec![Arc::new(b.finish())]).unwrap();

        let expr = NegExpr::new(Arc::new(ColumnExpr::new(0)));
        let result = expr.evaluate(&batch).unwrap();
        match result {
            ColumnarValue::Array(a) => {
                assert_eq!(as_primitive::<Int64Type>(a.as_ref()).unwrap().value(0), -5)
            }
            _ => panic!("expected array"),
        }

        let overflow_expr = NegExpr::new(Arc::new(LiteralExpr::new(ScalarValue::Int64(Some(
            i64::MIN,
        )))));
        assert!(overflow_expr.evaluate(&batch).is_err());
    }

    #[test]
    fn neg_on_non_numeric_type_errors() {
        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, false)]).unwrap();
        let expr = NegExpr::new(Arc::new(ColumnExpr::new(0)));
        assert!(expr.data_type(&schema).is_err());
    }
}
