//! `EXPLAIN` plan rendering. See design-docs/basalt-phase3-lld.md §8.1.
//!
//! An indented tree, one operator per line, with per-node row/selectivity
//! estimates attached — reusing `LogicalPlan`'s own `Display` impl
//! (Phase 2's stub) for the tree shape and adding `optimizer::cardinality`'s
//! estimates on top, rather than a second, separate tree-printing
//! implementation.

use crate::error::Result;
use crate::logical_plan::plan::LogicalPlan;
use crate::optimizer::cardinality::estimate_plan_statistics;
use crate::optimizer::rule::OptimizerContext;
use crate::physical_plan::plan::ExecutionPlan;
use crate::statistics::Precision;

/// Renders `plan` as an indented tree with `est_rows=N` attached to every
/// node (from `optimizer::cardinality::estimate_plan_statistics`, run once
/// per node — cheap relative to actually planning or executing).
/// `verbose` additionally attaches whether the estimate is `Exact`/
/// `Inexact`/`Absent`.
///
/// # Errors
/// Errors if statistics estimation fails for any subtree (an out-of-range
/// column reference — a genuine plan-construction bug, not a normal
/// "stats unavailable" case, which is handled by `Precision::Absent`).
pub fn explain(plan: &LogicalPlan, ctx: &dyn OptimizerContext, verbose: bool) -> Result<String> {
    let mut out = String::new();
    write_node(plan, ctx, verbose, 0, &mut out)?;
    Ok(out)
}

fn write_node(
    plan: &LogicalPlan,
    ctx: &dyn OptimizerContext,
    verbose: bool,
    depth: usize,
    out: &mut String,
) -> Result<()> {
    let indent = "  ".repeat(depth);
    let stats = estimate_plan_statistics(plan, ctx)?;
    let est_rows = describe_precision(stats.num_rows, verbose);
    // `LogicalPlan`'s own `Display` renders the whole subtree indented
    // already; take only this node's own first line (before its
    // children's lines) and append the estimate to it.
    let full = plan.to_string();
    let this_line = full.lines().next().unwrap_or_default();
    out.push_str(&format!(
        "{indent}{} (est_rows={est_rows})\n",
        this_line.trim_start()
    ));

    for child in plan.inputs() {
        write_node(child, ctx, verbose, depth + 1, out)?;
    }
    Ok(())
}

fn describe_precision(p: Precision<usize>, verbose: bool) -> String {
    match (p, verbose) {
        (Precision::Exact(v), false) => format!("{v}"),
        (Precision::Inexact(v), false) => format!("{v}"),
        (Precision::Absent, false) => "unknown".to_string(),
        (Precision::Exact(v), true) => format!("{v} (exact)"),
        (Precision::Inexact(v), true) => format!("{v} (inexact)"),
        (Precision::Absent, true) => "unknown (absent)".to_string(),
    }
}

/// Renders a physical plan tree, one operator per line, indented by
/// depth — the physical-plan counterpart of `explain`, using each
/// operator's `Debug` representation (every `ExecutionPlan` implementor
/// already derives/implements it) as the node label, since `ExecutionPlan`
/// has no separate "display name" method to add.
pub fn explain_physical(plan: &dyn ExecutionPlan) -> String {
    let mut out = String::new();
    write_physical_node(plan, 0, &mut out);
    out
}

fn write_physical_node(plan: &dyn ExecutionPlan, depth: usize, out: &mut String) {
    let indent = "  ".repeat(depth);
    let label = format!("{plan:?}");
    let first_line = label.lines().next().unwrap_or_default();
    out.push_str(&format!("{indent}{first_line}\n"));
    for child in plan.children() {
        write_physical_node(child.as_ref(), depth + 1, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::optimizer::rule::NoStatistics;
    use crate::physical_plan::planner::PhysicalPlanner;
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema, SchemaRef};
    use std::sync::Arc;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap())
    }

    #[test]
    fn explain_renders_an_indented_tree_with_row_estimates() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let plan = LogicalPlanBuilder::scan("t", source)
            .limit(0, Some(5))
            .build();
        let rendered = explain(&plan, &NoStatistics, false).unwrap();
        assert!(rendered.contains("Limit"));
        assert!(rendered.contains("TableScan"));
        assert!(rendered.contains("est_rows="));
        let limit_line = rendered.lines().next().unwrap();
        let scan_line = rendered.lines().nth(1).unwrap();
        assert!(!limit_line.starts_with(' '));
        assert!(scan_line.starts_with("  "));
    }

    #[test]
    fn verbose_explain_shows_precision_kind() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let plan = LogicalPlanBuilder::scan("t", source).build();
        let rendered = explain(&plan, &NoStatistics, true).unwrap();
        assert!(
            rendered.contains("absent"),
            "with no statistics, the estimate must show as absent"
        );
    }

    #[test]
    fn explain_physical_renders_the_physical_tree() {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        let logical = LogicalPlanBuilder::scan("t", source).build();
        let physical = PhysicalPlanner.create_physical_plan(&logical).unwrap();
        let rendered = explain_physical(physical.as_ref());
        assert!(rendered.contains("MemoryScanExec") || rendered.contains("Scan"));
    }
}
