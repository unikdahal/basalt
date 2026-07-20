//! `DataFrame` — relational operators and fluent pipeline API.
//!
//! Provides eager, row-at-a-time physical operators (filter, project, sort, limit)
//! and the execute pipeline running the entire query flow.

use crate::array::builder::ColumnBuilder;
use crate::batch::RecordBatch;
use crate::error::{BasaltError, Result};
use crate::expr::expr::Expr;
use crate::plan::binder::{BoundOrderBy, BoundProjection};
use crate::types::schema::{Field, Schema};
use crate::types::value::Value;

pub struct DataFrame {
    batch: RecordBatch,
}

impl DataFrame {
    /// Constructs a new DataFrame from a RecordBatch.
    pub fn new(batch: RecordBatch) -> Self {
        Self { batch }
    }

    pub fn schema(&self) -> &Schema {
        self.batch.schema()
    }

    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn into_batch(self) -> RecordBatch {
        self.batch
    }

    /// Evaluates a boolean expression on each row, keeping matching rows.
    pub fn filter(self, predicate: &Expr) -> Result<Self> {
        let indices = crate::expr::eval::eval_predicate(predicate, &self.batch)?;
        let batch = self.batch.take(&indices)?;
        Ok(Self::new(batch))
    }

    /// Evaluates projection expressions, computing a new batch schema.
    /// Fast-paths column projections by directly cloning the underlying Column.
    pub fn project(self, projections: &[BoundProjection]) -> Result<Self> {
        let num_rows = self.batch.num_rows();
        let mut columns = Vec::with_capacity(projections.len());
        let mut fields = Vec::with_capacity(projections.len());

        for p in projections {
            let dt = p.expr.data_type()?;
            let nullable = p.expr.nullable();
            fields.push(Field::new(p.output_name.clone(), dt, nullable));

            if let Expr::Column { index, .. } = &p.expr {
                // Optimization: directly clone the column if it's a bare column reference
                let col = self.batch.column(*index).ok_or_else(|| {
                    BasaltError::Internal(format!("column index {index} out of bounds"))
                })?;
                columns.push(col.clone());
            } else {
                let mut builder = ColumnBuilder::with_capacity(dt, num_rows);
                for r in 0..num_rows {
                    let val = crate::expr::eval::eval(&p.expr, &self.batch, r)?;
                    builder.append_value(val)?;
                }
                columns.push(builder.finish());
            }
        }

        let schema = Schema::new(fields)?;
        let batch = RecordBatch::try_new(schema, columns)?;
        Ok(Self::new(batch))
    }

    /// Sorts rows stably by comparing evaluated key values.
    /// NULL values are ordered first (SQLite style).
    pub fn sort(self, keys: &[BoundOrderBy]) -> Result<Self> {
        let num_rows = self.batch.num_rows();

        // Operator implementation (Stable index sorting):
        // Rather than moving entire rows around during the sort, we initialize a vector of indices
        // [0..num_rows] and sort it stably based on the evaluated multi-key sorting criteria.
        // We then use these indices to construct a new RecordBatch via a `take` operation.
        let mut indices: Vec<usize> = (0..num_rows).collect();

        let mut sort_err = None;
        indices.sort_by(|&i, &j| {
            if sort_err.is_some() {
                return std::cmp::Ordering::Equal;
            }

            for key in keys {
                let val_i = match crate::expr::eval::eval(&key.expr, &self.batch, i) {
                    Ok(v) => v,
                    Err(e) => {
                        sort_err = Some(e);
                        return std::cmp::Ordering::Equal;
                    }
                };
                let val_j = match crate::expr::eval::eval(&key.expr, &self.batch, j) {
                    Ok(v) => v,
                    Err(e) => {
                        sort_err = Some(e);
                        return std::cmp::Ordering::Equal;
                    }
                };

                // Null ordering policy: NULLs ordered first
                let ord = match (&val_i, &val_j) {
                    (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
                    (Value::Null, _) => {
                        if key.asc {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }
                    (_, Value::Null) => {
                        if key.asc {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    }
                    (l, r) => {
                        let ord = match crate::exec::ops::compare_values(l, r) {
                            Some(o) => o,
                            None => {
                                sort_err = Some(BasaltError::Type {
                                    message: format!("cannot compare values '{l}' and '{r}'"),
                                });
                                return std::cmp::Ordering::Equal;
                            }
                        };
                        if key.asc {
                            ord
                        } else {
                            ord.reverse()
                        }
                    }
                };

                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }

            std::cmp::Ordering::Equal
        });

        if let Some(e) = sort_err {
            return Err(e);
        }

        let batch = self.batch.take(&indices)?;
        Ok(Self::new(batch))
    }

    /// Limits the batch size to the first N rows.
    pub fn limit(self, n: usize) -> Result<Self> {
        let limit_rows = n.min(self.batch.num_rows());
        let indices: Vec<usize> = (0..limit_rows).collect();
        let batch = self.batch.take(&indices)?;
        Ok(Self::new(batch))
    }
}

/// The complete end-to-end execution pipeline:
/// SQL text -> Lexer -> Parser -> Binder -> Relational Operators -> RecordBatch
pub fn execute(sql: &str, input: RecordBatch) -> Result<RecordBatch> {
    // 1. Tokenize
    let mut lexer = crate::sql::lexer::Lexer::new(sql);
    let tokens = lexer.tokenize()?;

    // 2. Parse AST
    let mut parser = crate::sql::parser::Parser::new(tokens);
    let stmt = parser.parse_statement()?;

    // 3. Bind AST to BoundQuery (validating types, checking schemas, inserting Cast nodes)
    let binder = crate::plan::binder::Binder::new(input.schema());
    let query = binder.bind_statement(&stmt)?;

    // 4. Run operators in the optimal fixed order: scan -> filter -> sort -> limit -> project.
    // (We do manually here what cost-based optimizers will later plan dynamically).
    let mut df = DataFrame::new(input);

    if let Some(ref filter_expr) = query.filter {
        df = df.filter(filter_expr)?;
    }

    if !query.order_by.is_empty() {
        df = df.sort(&query.order_by)?;
    }

    if let Some(limit_val) = query.limit {
        df = df.limit(limit_val)?;
    }

    df = df.project(&query.projections)?;

    Ok(df.into_batch())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::column::{Column, ColumnData};
    use crate::types::data_type::DataType;
    use crate::types::schema::Field;

    fn sample_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("score", DataType::Float64, true),
        ])
        .unwrap();
        let cols = vec![
            Column::from_parts(ColumnData::Int64(vec![1, 2, 3]), None),
            Column::from_parts(
                ColumnData::Float64(vec![95.5, 88.0, f64::NAN]),
                Some(crate::array::validity::Validity::from_flags(vec![
                    true, true, true,
                ])),
            ),
        ];
        RecordBatch::try_new(schema, cols).unwrap()
    }

    #[test]
    fn test_dataframe_limit() {
        let df = DataFrame::new(sample_batch());
        let limited = df.limit(2).unwrap();
        assert_eq!(limited.num_rows(), 2);
    }

    #[test]
    fn test_execute_pipeline() {
        let input = sample_batch();
        let result = execute(
            "SELECT id, score FROM tbl WHERE id > 1 ORDER BY score ASC LIMIT 1",
            input,
        )
        .unwrap();
        assert_eq!(result.num_rows(), 1);
        assert_eq!(result.column(0).unwrap().get(0), Some(Value::Int64(2)));
    }

    #[test]
    fn test_dataframe_multi_key_sort() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("score", DataType::Float64, true),
        ])
        .unwrap();
        let cols = vec![
            Column::from_parts(ColumnData::Int64(vec![1, 1, 2, 2]), None),
            Column::from_parts(
                ColumnData::Float64(vec![90.0, 80.0, 95.0, 85.0]),
                Some(crate::array::validity::Validity::from_flags(vec![
                    true, true, true, true,
                ])),
            ),
        ];
        let batch = RecordBatch::try_new(schema, cols).unwrap();
        let df = DataFrame::new(batch);

        // Sort by id ASC, score DESC
        let keys = vec![
            BoundOrderBy {
                expr: Expr::Column {
                    index: 0,
                    data_type: DataType::Int64,
                    nullable: false,
                },
                asc: true,
            },
            BoundOrderBy {
                expr: Expr::Column {
                    index: 1,
                    data_type: DataType::Float64,
                    nullable: true,
                },
                asc: false,
            },
        ];

        let sorted = df.sort(&keys).unwrap();
        let id_col = sorted.into_batch().column(0).unwrap().clone();

        assert_eq!(id_col.get(0), Some(Value::Int64(1)));
        assert_eq!(id_col.get(2), Some(Value::Int64(2)));
    }

    #[test]
    fn test_dataframe_limit_validation() {
        let df = DataFrame::new(sample_batch());
        // Limit > num_rows should just return all rows
        let limited = df.limit(10).unwrap();
        assert_eq!(limited.num_rows(), 3);

        // Limit 0 should return empty batch
        let df2 = DataFrame::new(sample_batch());
        let empty = df2.limit(0).unwrap();
        assert_eq!(empty.num_rows(), 0);
    }
}
