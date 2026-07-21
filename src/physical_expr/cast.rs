//! `CastExpr` — explicit type conversion, materialized by the binder as a
//! `Cast` node (Phase 1's coercion discipline: casts are decided once, at
//! bind time, and only ever *executed* here — never decided again).

use std::any::Any;

use super::expr::{PhysicalExpr, PhysicalExprRef};
use crate::batch::ColumnarBatch;
use crate::compute::{cast, ColumnarValue};
use crate::error::Result;
use crate::types::data_type::DataType;
use crate::types::schema::Schema;

#[derive(Debug, Clone)]
pub struct CastExpr {
    pub expr: PhysicalExprRef,
    pub to: DataType,
}

impl CastExpr {
    pub fn new(expr: PhysicalExprRef, to: DataType) -> Self {
        CastExpr { expr, to }
    }
}

impl PhysicalExpr for CastExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.to)
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.expr.nullable(input_schema)
    }

    fn evaluate(&self, batch: &ColumnarBatch) -> Result<ColumnarValue> {
        match self.expr.evaluate(batch)? {
            ColumnarValue::Array(a) => Ok(ColumnarValue::Array(cast::cast(a.as_ref(), self.to)?)),
            ColumnarValue::Scalar(s) => Ok(ColumnarValue::Scalar(cast::cast_scalar(&s, self.to)?)),
        }
    }

    fn children(&self) -> Vec<PhysicalExprRef> {
        vec![std::sync::Arc::clone(&self.expr)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::as_primitive;
    use crate::physical_expr::literal::LiteralExpr;
    use crate::scalar::ScalarValue;
    use std::sync::Arc;

    #[test]
    fn casts_a_literal_scalar() {
        let schema = Schema::new(vec![]).unwrap();
        let batch = ColumnarBatch::try_new(Arc::new(schema.clone()), vec![]).unwrap();
        let expr = CastExpr::new(
            Arc::new(LiteralExpr::new(ScalarValue::Int64(Some(5)))),
            DataType::Float64,
        );
        assert_eq!(expr.data_type(&schema).unwrap(), DataType::Float64);
        match expr.evaluate(&batch).unwrap() {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(v))) => assert_eq!(v, 5.0),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn casts_an_array_column() {
        let mut b = crate::array::primitive::PrimitiveBuilder::<crate::array::types::Int64Type>::with_capacity(2);
        b.append_value(1);
        b.append_value(2);
        let schema = std::sync::Arc::new(
            Schema::new(vec![crate::types::schema::Field::new(
                "a",
                DataType::Int64,
                false,
            )])
            .unwrap(),
        );
        let batch = ColumnarBatch::try_new(schema, vec![Arc::new(b.finish())]).unwrap();
        let expr = CastExpr::new(
            Arc::new(crate::physical_expr::column::ColumnExpr::new(0)),
            DataType::Float64,
        );
        match expr.evaluate(&batch).unwrap() {
            ColumnarValue::Array(a) => {
                let a = as_primitive::<crate::array::types::Float64Type>(a.as_ref()).unwrap();
                assert_eq!(a.value(0), 1.0);
            }
            _ => panic!("expected array"),
        }
    }
}
