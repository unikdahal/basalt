//! Cardinality estimation — the intellectual core of the optimizer. See
//! design-docs/basalt-phase3-lld.md §3.

pub mod join_card;
pub mod selectivity;

use crate::error::Result;
use crate::expr::expr::Expr;
use crate::logical_plan::plan::{AggregateKind, LogicalPlan};
use crate::optimizer::rule::OptimizerContext;
use crate::statistics::{ColumnStatistics, Precision, TableStatistics};

/// Derives statistics for a plan node, bottom-up. See
/// design-docs/basalt-phase3-lld.md §3.4: every node transforms its
/// child's statistics — `Filter` multiplies row count by selectivity and
/// degrades precision; `Projection` selects column stats; `Join` applies
/// §3.3's join cardinality; `Aggregate` outputs one row per estimated
/// group (itself an estimate); `Limit` caps.
///
/// # Errors
/// Errors if an expression references a column index out of range for its
/// node's statistics.
pub fn estimate_plan_statistics(
    plan: &LogicalPlan,
    ctx: &dyn OptimizerContext,
) -> Result<TableStatistics> {
    Ok(match plan {
        LogicalPlan::TableScan {
            table_name,
            projection,
            filters,
            schema,
            ..
        } => {
            let base = ctx
                .statistics_for(table_name)
                .map(|s| (*s).clone())
                .unwrap_or_else(|| TableStatistics::unknown(schema.fields().len()));
            let projected = match projection {
                Some(cols) => TableStatistics {
                    num_rows: base.num_rows,
                    total_byte_size: Precision::Absent,
                    column_statistics: cols
                        .iter()
                        .map(|&c| {
                            base.column_statistics
                                .get(c)
                                .cloned()
                                .unwrap_or_else(ColumnStatistics::unknown)
                        })
                        .collect(),
                },
                None => base,
            };
            apply_filters(projected, filters)?
        }
        LogicalPlan::Filter { input, predicate } => {
            let child = estimate_plan_statistics(input, ctx)?;
            apply_filters(child, std::slice::from_ref(predicate))?
        }
        LogicalPlan::Projection { input, exprs, .. } => {
            let child = estimate_plan_statistics(input, ctx)?;
            let column_statistics = exprs
                .iter()
                .map(|e| match e {
                    Expr::Column { index, .. } => child
                        .column_statistics
                        .get(*index)
                        .cloned()
                        .unwrap_or_else(ColumnStatistics::unknown),
                    _ => ColumnStatistics::unknown(),
                })
                .collect();
            TableStatistics {
                num_rows: child.num_rows,
                total_byte_size: Precision::Absent,
                column_statistics,
            }
        }
        LogicalPlan::Join {
            left,
            right,
            on,
            join_type,
            schema,
            ..
        } => {
            let left_stats = estimate_plan_statistics(left, ctx)?;
            let right_stats = estimate_plan_statistics(right, ctx)?;
            let on_indices: Vec<(usize, usize)> = on
                .iter()
                .filter_map(|(l, r)| match (l, r) {
                    (Expr::Column { index: li, .. }, Expr::Column { index: ri, .. }) => {
                        Some((*li, *ri))
                    }
                    _ => None,
                })
                .collect();
            let num_rows = if on_indices.is_empty() {
                join_card::cross_product_cardinality(&left_stats, &right_stats)
            } else {
                join_card::join_cardinality(&left_stats, &right_stats, &on_indices, *join_type)
            };
            let mut column_statistics = left_stats.column_statistics;
            column_statistics.extend(right_stats.column_statistics);
            // Every combined column degrades to Inexact — even a column
            // that was Exact pre-join no longer has a provably exact
            // count once join selectivity is estimated (Precision
            // degrades monotonically up the plan tree; see
            // `statistics::Precision`'s own doc comment).
            let column_statistics: Vec<ColumnStatistics> = column_statistics
                .into_iter()
                .take(schema.fields().len())
                .map(degrade_column)
                .collect();
            TableStatistics {
                num_rows,
                total_byte_size: Precision::Absent,
                column_statistics,
            }
        }
        LogicalPlan::Aggregate {
            input,
            group_expr,
            aggr_expr,
            schema,
        } => {
            let child = estimate_plan_statistics(input, ctx)?;
            let num_groups = estimate_group_count(group_expr, &child);
            let mut column_statistics = Vec::with_capacity(schema.fields().len());
            for e in group_expr {
                let stats = match e {
                    Expr::Column { index, .. } => child
                        .column_statistics
                        .get(*index)
                        .cloned()
                        .unwrap_or_else(ColumnStatistics::unknown),
                    _ => ColumnStatistics::unknown(),
                };
                column_statistics.push(degrade_column(stats));
            }
            for agg in aggr_expr {
                column_statistics.push(match agg.kind {
                    // COUNT is never null and bounded by the group's row
                    // count — still Absent here rather than guessing a
                    // number, but a real, identified refinement.
                    AggregateKind::Count => ColumnStatistics::unknown(),
                    _ => ColumnStatistics::unknown(),
                });
            }
            TableStatistics {
                num_rows: num_groups,
                total_byte_size: Precision::Absent,
                column_statistics,
            }
        }
        LogicalPlan::Sort { input, .. } => estimate_plan_statistics(input, ctx)?,
        LogicalPlan::Limit { input, skip, fetch } => {
            let child = estimate_plan_statistics(input, ctx)?;
            let capped = match (child.num_rows.get_value(), fetch) {
                (Some(&rows), Some(f)) => Precision::Inexact((*f).min(rows.saturating_sub(*skip))),
                (None, Some(f)) => Precision::Inexact(*f),
                _ => child.num_rows.to_inexact(),
            };
            TableStatistics {
                num_rows: capped,
                total_byte_size: Precision::Absent,
                ..child
            }
        }
        LogicalPlan::EmptyRelation { schema } => TableStatistics {
            num_rows: Precision::Exact(0),
            total_byte_size: Precision::Exact(0),
            column_statistics: (0..schema.fields().len())
                .map(|_| ColumnStatistics::unknown())
                .collect(),
        },
    })
}

/// Multiplies row count by each filter's estimated selectivity (combined
/// via `selectivity::combine_conjunction`'s independence assumption across
/// separate filter expressions, same as ANDing them) and degrades every
/// column's precision — a filtered batch's statistics are never `Exact`
/// again, even if the pre-filter statistics were.
fn apply_filters(stats: TableStatistics, filters: &[Expr]) -> Result<TableStatistics> {
    if filters.is_empty() {
        return Ok(stats);
    }
    let mut sel_value = 1.0;
    for f in filters {
        sel_value *= selectivity::selectivity(f, &stats)?.value;
    }
    let num_rows = stats
        .num_rows
        .map(|r| ((r as f64) * sel_value).round().max(1.0) as usize)
        .to_inexact();
    let column_statistics = stats
        .column_statistics
        .into_iter()
        .map(degrade_column)
        .collect();
    Ok(TableStatistics {
        num_rows,
        total_byte_size: Precision::Absent,
        column_statistics,
    })
}

fn degrade_column(mut col: ColumnStatistics) -> ColumnStatistics {
    col.null_count = col.null_count.to_inexact();
    col.distinct_count = col.distinct_count.to_inexact();
    col.min_value = col.min_value.to_inexact();
    col.max_value = col.max_value.to_inexact();
    col
}

/// Estimated group count for an `Aggregate`: the combined NDV of the
/// grouping columns (product of per-column NDVs — independence again, the
/// same simplifying assumption as multi-column join estimation), capped by
/// the input row count (there can never be more groups than input rows).
fn estimate_group_count(group_expr: &[Expr], child: &TableStatistics) -> Precision<usize> {
    if group_expr.is_empty() {
        return Precision::Exact(1); // A whole-input aggregate always produces exactly one row.
    }
    let mut ndv_product = 1usize;
    let mut any_absent = false;
    for e in group_expr {
        match e {
            Expr::Column { index, .. } => match child
                .column_statistics
                .get(*index)
                .and_then(|c| c.distinct_count.get_value())
            {
                Some(&ndv) => ndv_product = ndv_product.saturating_mul(ndv.max(1)),
                None => any_absent = true,
            },
            _ => any_absent = true,
        }
    }
    if any_absent {
        return Precision::Absent;
    }
    let capped = match child.num_rows.get_value() {
        Some(&rows) => ndv_product.min(rows.max(1)),
        None => ndv_product,
    };
    Precision::Inexact(capped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::optimizer::rule::NoStatistics;
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::types::coercion::BinaryOp;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema, SchemaRef};
    use crate::types::value::Value;
    use std::sync::Arc;

    struct FixedStats(std::collections::HashMap<String, Arc<TableStatistics>>);
    impl OptimizerContext for FixedStats {
        fn statistics_for(&self, table_name: &str) -> Option<Arc<TableStatistics>> {
            self.0.get(table_name).cloned()
        }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap())
    }

    fn col(i: usize) -> Expr {
        Expr::Column {
            index: i,
            data_type: DataType::Int64,
            nullable: false,
        }
    }

    #[test]
    fn table_scan_with_no_statistics_is_unknown() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let plan = LogicalPlanBuilder::scan("t", source).build();
        let stats = estimate_plan_statistics(&plan, &NoStatistics).unwrap();
        assert!(stats.num_rows.is_absent());
    }

    #[test]
    fn table_scan_uses_provided_statistics() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let plan = LogicalPlanBuilder::scan("t", source).build();
        let mut table_stats = TableStatistics::unknown(1);
        table_stats.num_rows = Precision::Exact(1000);
        let ctx = FixedStats(std::collections::HashMap::from([(
            "t".to_string(),
            Arc::new(table_stats),
        )]));
        let stats = estimate_plan_statistics(&plan, &ctx).unwrap();
        assert_eq!(stats.num_rows.get_value(), Some(&1000));
    }

    #[test]
    fn filter_degrades_row_count_precision_and_reduces_count() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let mut table_stats = TableStatistics::unknown(1);
        table_stats.num_rows = Precision::Exact(1000);
        table_stats.column_statistics[0].distinct_count = Precision::Exact(100);
        let ctx = FixedStats(std::collections::HashMap::from([(
            "t".to_string(),
            Arc::new(table_stats),
        )]));

        let plan = LogicalPlanBuilder::scan("t", source)
            .filter(Expr::Binary {
                left: Box::new(col(0)),
                op: BinaryOp::Eq,
                right: Box::new(Expr::Literal(Value::Int64(5))),
            })
            .build();
        let stats = estimate_plan_statistics(&plan, &ctx).unwrap();
        assert!(
            !stats.num_rows.is_exact(),
            "row count precision must degrade after a filter"
        );
        let rows = *stats.num_rows.get_value().unwrap();
        assert!(
            rows < 1000,
            "an equality filter with NDV=100 should reduce the estimated row count"
        );
    }

    #[test]
    fn limit_caps_row_count_at_fetch() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let mut table_stats = TableStatistics::unknown(1);
        table_stats.num_rows = Precision::Exact(1000);
        let ctx = FixedStats(std::collections::HashMap::from([(
            "t".to_string(),
            Arc::new(table_stats),
        )]));
        let plan = LogicalPlan::Limit {
            input: LogicalPlanBuilder::scan("t", source).build(),
            skip: 0,
            fetch: Some(10),
        };
        let stats = estimate_plan_statistics(&plan, &ctx).unwrap();
        assert_eq!(stats.num_rows.get_value(), Some(&10));
    }

    #[test]
    fn empty_relation_has_exactly_zero_rows() {
        let plan = LogicalPlan::EmptyRelation { schema: schema() };
        let stats = estimate_plan_statistics(&plan, &NoStatistics).unwrap();
        assert_eq!(stats.num_rows, Precision::Exact(0));
    }

    #[test]
    fn aggregate_with_no_group_expr_is_exactly_one_row() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let plan = LogicalPlan::Aggregate {
            input: LogicalPlanBuilder::scan("t", source).build(),
            group_expr: vec![],
            aggr_expr: vec![],
            schema: schema(),
        };
        let stats = estimate_plan_statistics(&plan, &NoStatistics).unwrap();
        assert_eq!(stats.num_rows, Precision::Exact(1));
    }
}
