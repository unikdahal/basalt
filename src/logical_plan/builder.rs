//! `LogicalPlanBuilder` — fluent construction of a `LogicalPlan` tree,
//! computing each node's output schema as it goes. See
//! design-docs/basalt-phase2-lld.md §5.1.

use std::sync::Arc;

use super::plan::{AggregateFunction, JoinType, LogicalPlan, SortExpr, TableSource};
use crate::error::Result;
use crate::expr::expr::Expr;
use crate::types::schema::{Field, Schema, SchemaRef};

pub struct LogicalPlanBuilder {
    plan: Arc<LogicalPlan>,
}

impl LogicalPlanBuilder {
    pub fn scan(table_name: impl Into<String>, source: Arc<dyn TableSource>) -> Self {
        let schema = source.schema();
        LogicalPlanBuilder {
            plan: Arc::new(LogicalPlan::TableScan {
                table_name: table_name.into(),
                source,
                projection: None,
                filters: vec![],
                schema,
            }),
        }
    }

    pub fn filter(self, predicate: Expr) -> Self {
        LogicalPlanBuilder {
            plan: Arc::new(LogicalPlan::Filter {
                input: self.plan,
                predicate,
            }),
        }
    }

    /// # Errors
    /// Errors if any projection expression's type can't be resolved against
    /// the input schema, or if the resulting field names collide.
    pub fn project(self, exprs: Vec<Expr>, output_names: Vec<String>) -> Result<Self> {
        let mut fields = Vec::with_capacity(exprs.len());
        for (expr, name) in exprs.iter().zip(&output_names) {
            fields.push(Field::new(name.clone(), expr.data_type()?, expr.nullable()));
        }
        let schema = Arc::new(Schema::new(fields)?);
        Ok(LogicalPlanBuilder {
            plan: Arc::new(LogicalPlan::Projection {
                input: self.plan,
                exprs,
                schema,
            }),
        })
    }

    /// # Errors
    /// Errors if any aggregate or group-by expression's type can't be
    /// resolved, or if the resulting field names collide.
    pub fn aggregate(
        self,
        group_expr: Vec<Expr>,
        aggr_expr: Vec<AggregateFunction>,
    ) -> Result<Self> {
        let mut fields = Vec::with_capacity(group_expr.len() + aggr_expr.len());
        for expr in &group_expr {
            fields.push(Field::new(
                expr_display_name(expr),
                expr.data_type()?,
                expr.nullable(),
            ));
        }
        for agg in &aggr_expr {
            let data_type = match &agg.arg {
                Some(e) => e.data_type()?,
                None => crate::types::data_type::DataType::Int64, // COUNT(*)
            };
            fields.push(Field::new(agg.output_name.clone(), data_type, false));
        }
        let schema = Arc::new(Schema::new(fields)?);
        Ok(LogicalPlanBuilder {
            plan: Arc::new(LogicalPlan::Aggregate {
                input: self.plan,
                group_expr,
                aggr_expr,
                schema,
            }),
        })
    }

    pub fn sort(self, exprs: Vec<SortExpr>) -> Self {
        LogicalPlanBuilder {
            plan: Arc::new(LogicalPlan::Sort {
                input: self.plan,
                exprs,
            }),
        }
    }

    pub fn limit(self, skip: usize, fetch: Option<usize>) -> Self {
        LogicalPlanBuilder {
            plan: Arc::new(LogicalPlan::Limit {
                input: self.plan,
                skip,
                fetch,
            }),
        }
    }

    /// Infallible today (the output schema is just both sides' fields
    /// concatenated), but kept `Result`-returning like the other builder
    /// methods since join-key type-checking is a natural, likely addition
    /// here and callers shouldn't need to change when it lands.
    pub fn join(
        self,
        right: Arc<LogicalPlan>,
        on: Vec<(Expr, Expr)>,
        filter: Option<Expr>,
        join_type: JoinType,
    ) -> Result<Self> {
        let mut fields: Vec<Field> = self.plan.schema().fields().to_vec();
        fields.extend(right.schema().fields().iter().cloned());
        // Duplicate column names across the two sides are ordinary SQL (both
        // tables having an `id` column), not a schema bug — see
        // `Schema::new_allow_duplicate_names`'s doc comment.
        let schema = Arc::new(Schema::new_allow_duplicate_names(fields));
        Ok(LogicalPlanBuilder {
            plan: Arc::new(LogicalPlan::Join {
                left: self.plan,
                right,
                on,
                filter,
                join_type,
                schema,
            }),
        })
    }

    pub fn build(self) -> Arc<LogicalPlan> {
        self.plan
    }

    pub fn schema(&self) -> &SchemaRef {
        self.plan.schema()
    }
}

fn expr_display_name(expr: &Expr) -> String {
    match expr {
        Expr::Column { index, .. } => format!("col_{index}"),
        _ => "expr".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::plan::SortOptions;
    use crate::types::data_type::DataType;
    use std::any::Any;

    #[derive(Debug)]
    struct StubSource(SchemaRef);
    impl TableSource for StubSource {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn schema(&self) -> SchemaRef {
            self.0.clone()
        }
    }

    fn source() -> Arc<dyn TableSource> {
        Arc::new(StubSource(Arc::new(
            Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("name", DataType::Utf8, false),
            ])
            .unwrap(),
        )))
    }

    #[test]
    fn scan_then_filter_then_limit_builds_a_tree() {
        let plan = LogicalPlanBuilder::scan("t", source())
            .filter(Expr::Column {
                index: 0,
                data_type: DataType::Int64,
                nullable: false,
            })
            .limit(0, Some(10))
            .build();
        match plan.as_ref() {
            LogicalPlan::Limit { input, .. } => match input.as_ref() {
                LogicalPlan::Filter { .. } => {}
                _ => panic!("expected Filter under Limit"),
            },
            _ => panic!("expected Limit at root"),
        }
    }

    #[test]
    fn project_computes_output_schema_from_expr_types() {
        let plan = LogicalPlanBuilder::scan("t", source())
            .project(
                vec![Expr::Column {
                    index: 0,
                    data_type: DataType::Int64,
                    nullable: false,
                }],
                vec!["id".to_string()],
            )
            .unwrap()
            .build();
        assert_eq!(plan.schema().field(0).unwrap().name, "id");
        assert_eq!(plan.schema().field(0).unwrap().data_type, DataType::Int64);
    }

    #[test]
    fn sort_preserves_input_schema() {
        let plan = LogicalPlanBuilder::scan("t", source())
            .sort(vec![SortExpr {
                expr: Expr::Column {
                    index: 0,
                    data_type: DataType::Int64,
                    nullable: false,
                },
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }])
            .build();
        assert_eq!(plan.schema().len(), 2);
    }

    #[test]
    fn join_concatenates_both_sides_schemas() {
        let left = LogicalPlanBuilder::scan("t1", source());
        let right = LogicalPlanBuilder::scan("t2", source()).build();
        let plan = left
            .join(right, vec![], None, JoinType::Inner)
            .unwrap()
            .build();
        assert_eq!(plan.schema().len(), 4);
    }

    /// Regression test: both sides here share column names ("id", "name") —
    /// an ordinary self-join-shaped query. A prior version routed the
    /// concatenated fields through `Schema::new`, which rejects duplicate
    /// names, so this construction unwrap-panicked before the join even ran.
    #[test]
    fn join_preserves_both_duplicate_named_columns_rather_than_erroring() {
        let left = LogicalPlanBuilder::scan("t1", source());
        let right = LogicalPlanBuilder::scan("t2", source()).build();
        let plan = left
            .join(right, vec![], None, JoinType::Inner)
            .unwrap()
            .build();
        assert_eq!(plan.schema().field(0).unwrap().name, "id");
        assert_eq!(plan.schema().field(2).unwrap().name, "id");
    }
}
