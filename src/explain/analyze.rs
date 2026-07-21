//! `EXPLAIN ANALYZE` — execute the query, collect per-operator metrics,
//! and print them alongside the estimate. See
//! design-docs/basalt-phase3-lld.md §8.2.
//!
//! **Scope note.** The LLD's `OperatorMetrics` also lists `peak_memory`,
//! `spill_count`/`spill_bytes`, and scan-specific `files_scanned`/
//! `files_pruned`/`row_groups_pruned` — those need per-operator internal
//! instrumentation (each executor reporting its own numbers through the
//! `metrics()` hook `physical_plan::plan::ExecutionPlan` already declares,
//! stubbed to `None` everywhere since Phase 2). Wiring that through every
//! operator is real, identified follow-up work, not done here. What *is*
//! implemented and real: `output_rows` and `elapsed`, measured by actually
//! executing the plan — driven externally (a `Instant` around each node's
//! own `execute(0)` call), not self-reported, so it works today without
//! touching every executor.
//!
//! **A second, related scope note on `elapsed`**: because nothing here
//! calls each operator's *own* internal timer (there isn't one), the
//! measured `elapsed` at a node is **inclusive of its children's
//! execution time**, not exclusive — the same limitation self-reported
//! per-operator metrics would fix, and a real reason to eventually wire
//! `metrics()` through instead of measuring from outside.

use std::time::Instant;

use crate::error::Result;
use crate::physical_plan::plan::ExecutionPlan;

#[derive(Debug, Default, Clone, Copy)]
pub struct OperatorMetrics {
    pub output_rows: usize,
    pub elapsed: std::time::Duration,
}

pub struct AnalyzeNode {
    pub label: String,
    pub metrics: OperatorMetrics,
    pub children: Vec<AnalyzeNode>,
}

/// Executes `plan` (every node, via its own `execute(0)`) and renders an
/// indented tree annotated with `rows=N time=Xms` per operator.
///
/// # Errors
/// Errors if executing any node in the plan fails.
pub fn explain_analyze(plan: &dyn ExecutionPlan) -> Result<String> {
    let root = collect_metrics(plan)?;
    let mut out = String::new();
    render(&root, 0, &mut out);
    Ok(out)
}

fn collect_metrics(plan: &dyn ExecutionPlan) -> Result<AnalyzeNode> {
    let children = plan
        .children()
        .iter()
        .map(|c| collect_metrics(c.as_ref()))
        .collect::<Result<Vec<_>>>()?;

    let start = Instant::now();
    let stream = plan.execute(0)?;
    let mut rows = 0usize;
    for batch in stream {
        rows += batch?.num_rows();
    }
    let elapsed = start.elapsed();

    let label = format!("{plan:?}")
        .lines()
        .next()
        .unwrap_or_default()
        .to_string();
    Ok(AnalyzeNode {
        label,
        metrics: OperatorMetrics {
            output_rows: rows,
            elapsed,
        },
        children,
    })
}

fn render(node: &AnalyzeNode, depth: usize, out: &mut String) {
    let indent = "  ".repeat(depth);
    out.push_str(&format!(
        "{indent}{} (rows={} time={:.3}ms)\n",
        node.label,
        node.metrics.output_rows,
        node.metrics.elapsed.as_secs_f64() * 1000.0
    ));
    for child in &node.children {
        render(child, depth + 1, out);
    }
}

/// q-error: the standard metric in the cardinality-estimation literature
/// (Leis et al., VLDB 2015 — see `optimizer::cardinality`'s own doc
/// comment). Symmetric and multiplicative: a 2x overestimate and a 2x
/// underestimate are both q-error 2, and 1.0 means a perfect estimate —
/// the right metric because plan quality degrades with the *ratio*
/// between estimated and actual, not the raw difference.
///
/// # Panics
/// Panics if `estimated` or `actual` is 0 — q-error is undefined for a
/// zero on either side (a real estimate of "zero rows" that turns out
/// wrong is a `CanSkip`-vs-`MustScan` pruning bug, not a q-error data
/// point; callers computing q-error over a query workload should filter
/// zero-row cases out before calling this, not silently divide by them).
pub fn q_error(estimated: usize, actual: usize) -> f64 {
    assert!(
        estimated > 0 && actual > 0,
        "q_error is undefined when either estimated or actual is 0"
    );
    let (e, a) = (estimated as f64, actual as f64);
    (e / a).max(a / e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::physical_plan::planner::PhysicalPlanner;
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema, SchemaRef};
    use std::sync::Arc;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap())
    }

    #[test]
    fn explain_analyze_reports_actual_row_counts() {
        use crate::array::primitive::PrimitiveBuilder;
        use crate::array::types::Int64Type;
        use crate::batch::ColumnarBatch;

        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(5);
        for i in 0..5i64 {
            b.append_value(i);
        }
        let batch = ColumnarBatch::try_new(schema(), vec![Arc::new(b.finish())]).unwrap();
        let source = Arc::new(MemoryTableSource::new(schema(), vec![batch]));
        let logical = LogicalPlanBuilder::scan("t", source)
            .limit(0, Some(3))
            .build();
        let physical = PhysicalPlanner.create_physical_plan(&logical).unwrap();
        let rendered = explain_analyze(physical.as_ref()).unwrap();
        assert!(
            rendered.contains("rows=3"),
            "Limit(0,3) over 5 rows should report 3 actual rows: {rendered}"
        );
        assert!(
            rendered.contains("rows=5"),
            "the scan below it should report all 5 rows: {rendered}"
        );
    }

    #[test]
    fn q_error_is_one_for_a_perfect_estimate() {
        assert_eq!(q_error(100, 100), 1.0);
    }

    #[test]
    fn q_error_is_symmetric_for_over_and_under_estimates() {
        assert_eq!(q_error(200, 100), 2.0);
        assert_eq!(q_error(100, 200), 2.0);
    }

    #[test]
    #[should_panic(expected = "undefined when either")]
    fn q_error_panics_on_zero() {
        q_error(0, 100);
    }
}
