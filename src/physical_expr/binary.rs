//! `BinaryExpr` — dispatches to the `compute` kernels for the ten binary
//! operators. See design-docs/basalt-phase2-lld.md §5.2.

use std::any::Any;
use std::sync::Arc;

use super::expr::{PhysicalExpr, PhysicalExprRef};
use crate::array::array::as_boolean;
use crate::array::boolean::BooleanArray;
use crate::batch::ColumnarBatch;
use crate::compute::{arith, boolean, comparison, ColumnarValue};
use crate::error::Result;
use crate::scalar::ScalarValue;
use crate::types::coercion::{coerce_binary, BinaryOp};
use crate::types::data_type::DataType;
use crate::types::schema::Schema;

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub left: PhysicalExprRef,
    pub op: BinaryOp,
    pub right: PhysicalExprRef,
}

impl BinaryExpr {
    pub fn new(left: PhysicalExprRef, op: BinaryOp, right: PhysicalExprRef) -> Self {
        BinaryExpr { left, op, right }
    }
}

impl PhysicalExpr for BinaryExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        let lhs = self.left.data_type(input_schema)?;
        let rhs = self.right.data_type(input_schema)?;
        Ok(coerce_binary(self.op, lhs, rhs)?.output)
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        Ok(self.left.nullable(input_schema)? || self.right.nullable(input_schema)?)
    }

    fn evaluate(&self, batch: &ColumnarBatch) -> Result<ColumnarValue> {
        let lhs = self.left.evaluate(batch)?;
        let rhs = self.right.evaluate(batch)?;

        use BinaryOp::*;
        match self.op {
            Add => arith::add(&lhs, &rhs),
            Sub => arith::sub(&lhs, &rhs),
            Mul => arith::mul(&lhs, &rhs),
            Div => arith::div(&lhs, &rhs),
            Mod => arith::rem(&lhs, &rhs),
            Eq => comparison::eq(&lhs, &rhs),
            NotEq => comparison::neq(&lhs, &rhs),
            Lt => comparison::lt(&lhs, &rhs),
            LtEq => comparison::lteq(&lhs, &rhs),
            Gt => comparison::gt(&lhs, &rhs),
            GtEq => comparison::gteq(&lhs, &rhs),
            And => eval_kleene(
                batch.num_rows(),
                &lhs,
                &rhs,
                boolean::and_kleene,
                |l, r| match (l, r) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                },
            ),
            Or => eval_kleene(
                batch.num_rows(),
                &lhs,
                &rhs,
                boolean::or_kleene,
                |l, r| match (l, r) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                },
            ),
        }
    }

    fn children(&self) -> Vec<PhysicalExprRef> {
        vec![Arc::clone(&self.left), Arc::clone(&self.right)]
    }
}

/// `AND`/`OR` need Kleene logic on `BooleanArray`, not the generic numeric
/// path `arith`/`comparison` use — this bridges `ColumnarValue` to that.
fn eval_kleene(
    num_rows: usize,
    lhs: &ColumnarValue,
    rhs: &ColumnarValue,
    array_op: impl Fn(&BooleanArray, &BooleanArray) -> Result<BooleanArray>,
    scalar_op: impl Fn(Option<bool>, Option<bool>) -> Option<bool>,
) -> Result<ColumnarValue> {
    if let (
        ColumnarValue::Scalar(ScalarValue::Boolean(l)),
        ColumnarValue::Scalar(ScalarValue::Boolean(r)),
    ) = (lhs, rhs)
    {
        return Ok(ColumnarValue::Scalar(ScalarValue::Boolean(scalar_op(
            *l, *r,
        ))));
    }
    let lhs_array = lhs.clone().into_array(num_rows)?;
    let rhs_array = rhs.clone().into_array(num_rows)?;
    let lhs_bool = as_boolean(lhs_array.as_ref())?;
    let rhs_bool = as_boolean(rhs_array.as_ref())?;
    Ok(ColumnarValue::Array(Arc::new(array_op(
        lhs_bool, rhs_bool,
    )?)))
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

    fn test_batch() -> ColumnarBatch {
        let schema = std::sync::Arc::new(
            Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap(),
        );
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(3);
        b.append_value(1);
        b.append_value(2);
        b.append_value(3);
        ColumnarBatch::try_new(schema, vec![Arc::new(b.finish())]).unwrap()
    }

    #[test]
    fn add_column_and_literal() {
        let batch = test_batch();
        let expr = BinaryExpr::new(
            Arc::new(ColumnExpr::new(0)),
            BinaryOp::Add,
            Arc::new(LiteralExpr::new(ScalarValue::Int64(Some(10)))),
        );
        let result = expr.evaluate(&batch).unwrap();
        match result {
            ColumnarValue::Array(a) => {
                let a = as_primitive::<Int64Type>(a.as_ref()).unwrap();
                assert_eq!(a.value(0), 11);
                assert_eq!(a.value(2), 13);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn and_short_circuit_semantics_via_kleene() {
        let schema = Schema::new(vec![]).unwrap();
        let expr = BinaryExpr::new(
            Arc::new(LiteralExpr::new(ScalarValue::Boolean(Some(false)))),
            BinaryOp::And,
            Arc::new(LiteralExpr::new(ScalarValue::Boolean(None))),
        );
        assert_eq!(expr.data_type(&schema).unwrap(), DataType::Boolean);
        // false AND NULL = false, not NULL.
        match expr
            .evaluate(&ColumnarBatch::try_new(Arc::new(schema), vec![]).unwrap())
            .unwrap()
        {
            ColumnarValue::Scalar(ScalarValue::Boolean(v)) => assert_eq!(v, Some(false)),
            _ => panic!("expected boolean scalar"),
        }
    }

    #[test]
    fn mismatched_types_error_via_coercion() {
        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, false)]).unwrap();
        let expr = BinaryExpr::new(
            Arc::new(ColumnExpr::new(0)),
            BinaryOp::Add,
            Arc::new(LiteralExpr::new(ScalarValue::Int64(Some(1)))),
        );
        assert!(expr.data_type(&schema).is_err());
    }
}
