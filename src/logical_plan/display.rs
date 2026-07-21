//! Plan tree pretty-printing, stubbed for `EXPLAIN` — Phase 3 attaches
//! per-node statistics/cost and `EXPLAIN ANALYZE` metrics to this same tree
//! shape; see design-docs/basalt-phase2-lld.md §5.3, §14.

use super::plan::LogicalPlan;

impl std::fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_indented(self, f, 0)
    }
}

fn write_indented(
    plan: &LogicalPlan,
    f: &mut std::fmt::Formatter<'_>,
    depth: usize,
) -> std::fmt::Result {
    let indent = "  ".repeat(depth);
    match plan {
        LogicalPlan::TableScan {
            table_name,
            projection,
            filters,
            ..
        } => {
            writeln!(
                f,
                "{indent}TableScan: {table_name} projection={projection:?} filters={}",
                filters.len()
            )?;
        }
        LogicalPlan::Projection { exprs, .. } => {
            writeln!(f, "{indent}Projection: {} expr(s)", exprs.len())?;
        }
        LogicalPlan::Filter { predicate, .. } => {
            writeln!(f, "{indent}Filter: {predicate:?}")?;
        }
        LogicalPlan::Aggregate {
            group_expr,
            aggr_expr,
            ..
        } => {
            writeln!(
                f,
                "{indent}Aggregate: groupBy=[{} expr(s)], aggr=[{} expr(s)]",
                group_expr.len(),
                aggr_expr.len()
            )?;
        }
        LogicalPlan::Join { join_type, on, .. } => {
            writeln!(
                f,
                "{indent}Join: type={join_type:?} on={} pair(s)",
                on.len()
            )?;
        }
        LogicalPlan::Sort { exprs, .. } => {
            writeln!(f, "{indent}Sort: {} key(s)", exprs.len())?;
        }
        LogicalPlan::Limit { skip, fetch, .. } => {
            writeln!(f, "{indent}Limit: skip={skip} fetch={fetch:?}")?;
        }
    }
    for input in plan.inputs() {
        write_indented(input, f, depth + 1)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::plan::TableSource;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema, SchemaRef};
    use std::any::Any;
    use std::sync::Arc;

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

    #[test]
    fn display_renders_nested_indented_tree() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap());
        let scan = LogicalPlan::TableScan {
            table_name: "t".to_string(),
            source: Arc::new(StubSource(schema.clone())),
            projection: None,
            filters: vec![],
            schema: schema.clone(),
        };
        let limit = LogicalPlan::Limit {
            input: Arc::new(scan),
            skip: 0,
            fetch: Some(5),
        };
        let rendered = limit.to_string();
        assert!(rendered.contains("Limit"));
        assert!(rendered.contains("TableScan"));
        // The scan line is indented one level deeper than the limit line.
        let limit_line = rendered.lines().next().unwrap();
        let scan_line = rendered.lines().nth(1).unwrap();
        assert!(!limit_line.starts_with(' '));
        assert!(scan_line.starts_with("  "));
    }
}
