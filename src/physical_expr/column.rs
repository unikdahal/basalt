//! `ColumnExpr` — a bare column reference, resolved to an ordinal at bind
//! time (same principle as Phase 1's `expr::expr::Expr::Column`).

use std::any::Any;

use super::expr::{PhysicalExpr, PhysicalExprRef};
use crate::batch::ColumnarBatch;
use crate::compute::ColumnarValue;
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;
use crate::types::schema::Schema;

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnExpr {
    pub index: usize,
}

impl ColumnExpr {
    pub fn new(index: usize) -> Self {
        ColumnExpr { index }
    }

    fn field<'a>(&self, input_schema: &'a Schema) -> Result<&'a crate::types::schema::Field> {
        input_schema.field(self.index).ok_or_else(|| {
            BasaltError::Internal(format!(
                "column index {} out of bounds for schema",
                self.index
            ))
        })
    }
}

impl PhysicalExpr for ColumnExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        Ok(self.field(input_schema)?.data_type)
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        Ok(self.field(input_schema)?.nullable)
    }

    fn evaluate(&self, batch: &ColumnarBatch) -> Result<ColumnarValue> {
        let col = batch.column(self.index).ok_or_else(|| {
            BasaltError::Internal(format!(
                "column index {} out of bounds for batch",
                self.index
            ))
        })?;
        Ok(ColumnarValue::Array(std::sync::Arc::clone(col)))
    }

    fn children(&self) -> Vec<PhysicalExprRef> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive::PrimitiveBuilder;
    use crate::array::types::Int64Type;
    use crate::types::data_type::DataType;
    use crate::types::schema::Field;
    use std::sync::Arc;

    fn test_batch() -> (ColumnarBatch, Schema) {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap();
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(2);
        b.append_value(1);
        b.append_value(2);
        let batch =
            ColumnarBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(b.finish())]).unwrap();
        (batch, schema)
    }

    #[test]
    fn evaluates_to_the_referenced_column() {
        let (batch, schema) = test_batch();
        let expr = ColumnExpr::new(0);
        assert_eq!(expr.data_type(&schema).unwrap(), DataType::Int64);
        assert!(!expr.nullable(&schema).unwrap());
        match expr.evaluate(&batch).unwrap() {
            ColumnarValue::Array(a) => assert_eq!(a.len(), 2),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn out_of_bounds_index_errors_not_panics() {
        let (batch, schema) = test_batch();
        let expr = ColumnExpr::new(5);
        assert!(expr.data_type(&schema).is_err());
        assert!(expr.evaluate(&batch).is_err());
    }
}
