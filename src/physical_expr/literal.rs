//! `LiteralExpr` — a constant, kept as a `ScalarValue` rather than
//! materialized into an array (see `compute::ColumnarValue`'s doc comment).

use std::any::Any;

use super::expr::{PhysicalExpr, PhysicalExprRef};
use crate::batch::ColumnarBatch;
use crate::compute::ColumnarValue;
use crate::error::Result;
use crate::scalar::ScalarValue;
use crate::types::data_type::DataType;
use crate::types::schema::Schema;

#[derive(Debug, Clone, PartialEq)]
pub struct LiteralExpr {
    pub value: ScalarValue,
}

impl LiteralExpr {
    pub fn new(value: ScalarValue) -> Self {
        LiteralExpr { value }
    }
}

impl PhysicalExpr for LiteralExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.value.data_type())
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(self.value.is_null())
    }

    fn evaluate(&self, _batch: &ColumnarBatch) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Scalar(self.value.clone()))
    }

    fn children(&self) -> Vec<PhysicalExprRef> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::schema::Field;

    #[test]
    fn evaluates_to_the_constant_scalar() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap();
        let expr = LiteralExpr::new(ScalarValue::Int64(Some(42)));
        assert_eq!(expr.data_type(&schema).unwrap(), DataType::Int64);
        assert!(!expr.nullable(&schema).unwrap());
    }

    #[test]
    fn null_literal_reports_nullable() {
        let schema = Schema::new(vec![]).unwrap();
        let expr = LiteralExpr::new(ScalarValue::Utf8(None));
        assert!(expr.nullable(&schema).unwrap());
    }
}
