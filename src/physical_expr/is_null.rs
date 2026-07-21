//! `IsNullExpr`/`IsNotNullExpr` — the only way to test for null (Phase 1's
//! Rule 3: `= NULL` can't work, since equality of unknowns is unknown).
//! Always produce a real, non-null boolean — never `NULL` themselves.

use std::any::Any;
use std::sync::Arc;

use super::expr::{PhysicalExpr, PhysicalExprRef};
use crate::array::boolean::BooleanBuilder;
use crate::batch::ColumnarBatch;
use crate::compute::ColumnarValue;
use crate::error::Result;
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;
use crate::types::schema::Schema;

#[derive(Debug, Clone)]
pub struct IsNullExpr {
    pub expr: PhysicalExprRef,
    /// `false` for `IS NULL`, `true` for `IS NOT NULL`.
    pub negated: bool,
}

impl IsNullExpr {
    pub fn new(expr: PhysicalExprRef, negated: bool) -> Self {
        IsNullExpr { expr, negated }
    }
}

impl PhysicalExpr for IsNullExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &ColumnarBatch) -> Result<ColumnarValue> {
        let negated = self.negated;
        match self.expr.evaluate(batch)? {
            ColumnarValue::Scalar(s) => Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(
                s.is_null() != negated,
            )))),
            ColumnarValue::Array(a) => {
                let mut builder = BooleanBuilder::with_capacity(a.len());
                for i in 0..a.len() {
                    builder.append_value(a.is_null(i) != negated);
                }
                Ok(ColumnarValue::Array(Arc::new(builder.finish())))
            }
        }
    }

    fn children(&self) -> Vec<PhysicalExprRef> {
        vec![Arc::clone(&self.expr)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::array::{as_boolean, Array};
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::physical_expr::column::ColumnExpr;
    use crate::physical_expr::literal::LiteralExpr;
    use crate::types::schema::Field;

    #[test]
    fn is_null_and_is_not_null_never_produce_null_themselves() {
        let schema =
            std::sync::Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]).unwrap());
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(2);
        b.append_null();
        b.append_value(1);
        let batch = ColumnarBatch::try_new(schema, vec![Arc::new(b.finish())]).unwrap();

        let is_null = IsNullExpr::new(Arc::new(ColumnExpr::new(0)), false);
        let result = is_null.evaluate(&batch).unwrap();
        let result = match result {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let result = as_boolean(result.as_ref()).unwrap();
        assert!(result.value(0));
        assert!(!result.value(1));
        assert_eq!(result.null_count(), 0);

        let is_not_null = IsNullExpr::new(Arc::new(ColumnExpr::new(0)), true);
        let result = is_not_null.evaluate(&batch).unwrap();
        let result = match result {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let result = as_boolean(result.as_ref()).unwrap();
        assert!(!result.value(0));
        assert!(result.value(1));
    }

    #[test]
    fn scalar_literal_is_null_check() {
        let schema = Schema::new(vec![]).unwrap();
        let batch = ColumnarBatch::try_new(Arc::new(schema), vec![]).unwrap();
        let expr = IsNullExpr::new(Arc::new(LiteralExpr::new(ScalarValue::Int64(None))), false);
        match expr.evaluate(&batch).unwrap() {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(v))) => assert!(v),
            other => panic!("unexpected {other:?}"),
        }
    }
}
