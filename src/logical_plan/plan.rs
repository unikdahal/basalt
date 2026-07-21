//! `LogicalPlan` — the query plan tree. See
//! design-docs/basalt-phase2-lld.md §5.1.
//!
//! An **enum**, deliberately the opposite choice from `PhysicalExpr`/
//! `ExecutionPlan` (both `dyn`). The logical plan is a *closed set* this
//! crate owns, and Phase 3's optimizer will pattern-match on it constantly
//! (`match plan { Filter { input: Filter { .. }, .. } => ... }`). Exhaustive
//! matching over an enum means adding a node type makes the compiler list
//! every rewrite rule that needs updating; with trait objects that
//! exhaustiveness guarantee is gone and every rule needs a `downcast_ref`
//! chain instead.
//!
//! Expressions reuse Phase 1's bound `expr::expr::Expr` directly rather than
//! a new logical expression type — it already carries resolved column
//! ordinals and types (exactly what a post-binding logical plan needs), and
//! duplicating it would just be the same tree with a different name.

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use crate::error::{BasaltError, Result};
use crate::expr::expr::Expr;
use crate::types::schema::SchemaRef;

/// Where a `TableScan` gets its rows. An open set (CSV, Parquet, in-memory,
/// eventually Iceberg) — `dyn`, not an enum, for the same reason
/// `PhysicalExpr` is `dyn`: sources are added over time, not enumerated once.
pub trait TableSource: Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn schema(&self) -> SchemaRef;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    LeftSemi,
    LeftAnti,
    RightSemi,
    RightAnti,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregateKind {
    Sum,
    Count,
    Min,
    Max,
    Avg,
}

/// One aggregate expression in a `GROUP BY` (or whole-input) aggregation.
/// `arg` is `None` for `COUNT(*)`, which counts rows rather than non-null
/// values of a particular expression.
#[derive(Clone, Debug)]
pub struct AggregateFunction {
    pub output_name: String,
    pub kind: AggregateKind,
    pub arg: Option<Expr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SortOptions {
    pub descending: bool,
    pub nulls_first: bool,
}

#[derive(Clone, Debug)]
pub struct SortExpr {
    pub expr: Expr,
    pub options: SortOptions,
}

#[derive(Clone, Debug)]
pub enum LogicalPlan {
    TableScan {
        table_name: String,
        source: Arc<dyn TableSource>,
        /// Pushdown candidate — populated by Phase 3's rules; Phase 2 always
        /// leaves this `None` and projects everything.
        projection: Option<Vec<usize>>,
        /// Pushdown candidates — populated by Phase 3's rules; Phase 2 never
        /// pushes filters into the scan itself.
        filters: Vec<Expr>,
        schema: SchemaRef,
    },
    Projection {
        input: Arc<LogicalPlan>,
        exprs: Vec<Expr>,
        schema: SchemaRef,
    },
    Filter {
        input: Arc<LogicalPlan>,
        predicate: Expr,
    },
    Aggregate {
        input: Arc<LogicalPlan>,
        group_expr: Vec<Expr>,
        aggr_expr: Vec<AggregateFunction>,
        schema: SchemaRef,
    },
    Join {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        on: Vec<(Expr, Expr)>,
        filter: Option<Expr>,
        join_type: JoinType,
        schema: SchemaRef,
    },
    Sort {
        input: Arc<LogicalPlan>,
        exprs: Vec<SortExpr>,
    },
    Limit {
        input: Arc<LogicalPlan>,
        skip: usize,
        fetch: Option<usize>,
    },
}

impl LogicalPlan {
    pub fn schema(&self) -> &SchemaRef {
        match self {
            LogicalPlan::TableScan { schema, .. }
            | LogicalPlan::Projection { schema, .. }
            | LogicalPlan::Aggregate { schema, .. }
            | LogicalPlan::Join { schema, .. } => schema,
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. } => input.schema(),
        }
    }

    pub fn inputs(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::TableScan { .. } => vec![],
            LogicalPlan::Projection { input, .. }
            | LogicalPlan::Filter { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. } => vec![input.as_ref()],
            LogicalPlan::Join { left, right, .. } => vec![left.as_ref(), right.as_ref()],
        }
    }

    /// The tree-rewriting primitive every optimizer rule builds on: recurse
    /// into children, then rebuild this node with the new children.
    ///
    /// # Errors
    /// Errors if `inputs`'s length doesn't match how many children this
    /// variant has (0 for `TableScan`, 1 for most, 2 for `Join`).
    pub fn with_new_inputs(&self, inputs: Vec<Arc<LogicalPlan>>) -> Result<LogicalPlan> {
        fn one(inputs: &[Arc<LogicalPlan>]) -> Result<Arc<LogicalPlan>> {
            match inputs {
                [single] => Ok(Arc::clone(single)),
                other => Err(BasaltError::Internal(format!(
                    "expected exactly 1 input, got {}",
                    other.len()
                ))),
            }
        }

        match self {
            LogicalPlan::TableScan { .. } => {
                if !inputs.is_empty() {
                    return Err(BasaltError::Internal(format!(
                        "TableScan takes 0 inputs, got {}",
                        inputs.len()
                    )));
                }
                Ok(self.clone())
            }
            LogicalPlan::Projection { exprs, schema, .. } => Ok(LogicalPlan::Projection {
                input: one(&inputs)?,
                exprs: exprs.clone(),
                schema: schema.clone(),
            }),
            LogicalPlan::Filter { predicate, .. } => Ok(LogicalPlan::Filter {
                input: one(&inputs)?,
                predicate: predicate.clone(),
            }),
            LogicalPlan::Aggregate {
                group_expr,
                aggr_expr,
                schema,
                ..
            } => Ok(LogicalPlan::Aggregate {
                input: one(&inputs)?,
                group_expr: group_expr.clone(),
                aggr_expr: aggr_expr.clone(),
                schema: schema.clone(),
            }),
            LogicalPlan::Join {
                on,
                filter,
                join_type,
                schema,
                ..
            } => match inputs.as_slice() {
                [left, right] => Ok(LogicalPlan::Join {
                    left: Arc::clone(left),
                    right: Arc::clone(right),
                    on: on.clone(),
                    filter: filter.clone(),
                    join_type: *join_type,
                    schema: schema.clone(),
                }),
                other => Err(BasaltError::Internal(format!(
                    "Join takes exactly 2 inputs, got {}",
                    other.len()
                ))),
            },
            LogicalPlan::Sort { exprs, .. } => Ok(LogicalPlan::Sort {
                input: one(&inputs)?,
                exprs: exprs.clone(),
            }),
            LogicalPlan::Limit { skip, fetch, .. } => Ok(LogicalPlan::Limit {
                input: one(&inputs)?,
                skip: *skip,
                fetch: *fetch,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema};

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

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap())
    }

    fn scan() -> LogicalPlan {
        LogicalPlan::TableScan {
            table_name: "t".to_string(),
            source: Arc::new(StubSource(schema())),
            projection: None,
            filters: vec![],
            schema: schema(),
        }
    }

    #[test]
    fn schema_delegates_through_single_input_nodes() {
        let plan = LogicalPlan::Limit {
            input: Arc::new(scan()),
            skip: 0,
            fetch: Some(10),
        };
        assert_eq!(plan.schema(), &schema());
    }

    #[test]
    fn inputs_reports_zero_for_scan_and_two_for_join() {
        let scan_plan = scan();
        assert!(scan_plan.inputs().is_empty());

        let join = LogicalPlan::Join {
            left: Arc::new(scan()),
            right: Arc::new(scan()),
            on: vec![],
            filter: None,
            join_type: JoinType::Inner,
            schema: schema(),
        };
        assert_eq!(join.inputs().len(), 2);
    }

    #[test]
    fn with_new_inputs_rebuilds_filter_with_new_child() {
        let filter = LogicalPlan::Filter {
            input: Arc::new(scan()),
            predicate: Expr::Literal(crate::types::value::Value::Boolean(true)),
        };
        let new_child = Arc::new(scan());
        let rebuilt = filter
            .with_new_inputs(vec![Arc::clone(&new_child)])
            .unwrap();
        match rebuilt {
            LogicalPlan::Filter { input, .. } => assert!(Arc::ptr_eq(&input, &new_child)),
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn with_new_inputs_rejects_wrong_arity() {
        let limit = LogicalPlan::Limit {
            input: Arc::new(scan()),
            skip: 0,
            fetch: None,
        };
        assert!(limit.with_new_inputs(vec![]).is_err());
        assert!(limit
            .with_new_inputs(vec![Arc::new(scan()), Arc::new(scan())])
            .is_err());
    }

    #[test]
    fn table_scan_with_new_inputs_requires_zero_inputs() {
        let scan_plan = scan();
        assert!(scan_plan.with_new_inputs(vec![]).is_ok());
        assert!(scan_plan.with_new_inputs(vec![Arc::new(scan())]).is_err());
    }
}
