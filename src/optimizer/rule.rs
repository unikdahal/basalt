//! `OptimizerRule` and the fixed-point pass manager. See
//! design-docs/basalt-phase3-lld.md §5.2.

use std::sync::Arc;

use super::tree_node::Transformed;
use crate::error::Result;
use crate::logical_plan::plan::LogicalPlan;
use crate::statistics::TableStatistics;

/// Order a rule wants to be applied in its traversal. Most rules only care
/// about the node in front of them (`BottomUp`, applied via
/// `TreeNode::transform_up`); pushdown-style rules that need to see the
/// root before rewriting children want `TopDown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyOrder {
    BottomUp,
    TopDown,
}

/// Per-table statistics available to rules that need them (join ordering,
/// pruning). `None` means "no statistics known for this table" — rules
/// must still produce a correct plan, just possibly a less optimal one.
pub trait OptimizerContext {
    fn statistics_for(&self, table_name: &str) -> Option<Arc<TableStatistics>>;
}

/// A context carrying no statistics — the correct fallback everywhere else
/// (see `statistics::TableStatistics::unknown`).
pub struct NoStatistics;

impl OptimizerContext for NoStatistics {
    fn statistics_for(&self, _table_name: &str) -> Option<Arc<TableStatistics>> {
        None
    }
}

pub trait OptimizerRule: std::fmt::Debug {
    fn name(&self) -> &str;

    /// # Errors
    /// Errors if the rule can't validly rewrite this plan (a malformed
    /// input, not "the rule doesn't apply" — that's `Transformed::No`).
    fn apply(&self, plan: LogicalPlan, ctx: &dyn OptimizerContext) -> Result<Transformed<LogicalPlan>>;

    fn apply_order(&self) -> ApplyOrder {
        ApplyOrder::BottomUp
    }
}

pub struct Optimizer {
    pub rules: Vec<Arc<dyn OptimizerRule>>,
    /// 10 is plenty for any rule set that isn't oscillating; see below.
    pub max_iterations: usize,
}

impl Optimizer {
    pub fn new(rules: Vec<Arc<dyn OptimizerRule>>) -> Self {
        Optimizer {
            rules,
            max_iterations: 10,
        }
    }

    /// Fixed-point iteration: apply every rule in order; if any returned
    /// `Yes`, iterate again; stop at `max_iterations`.
    ///
    /// The cap exists because rules *can* oscillate (rule A rewrites X to
    /// Y, rule B rewrites Y back to X, forever) — that's a bug in the rule
    /// set, not a normal outcome, so hitting the cap logs loudly (via
    /// `eprintln!`; this crate has no logging framework dependency yet)
    /// rather than silently stopping.
    ///
    /// # Errors
    /// Propagates the first error any rule returns.
    pub fn optimize(&self, plan: LogicalPlan, ctx: &dyn OptimizerContext) -> Result<LogicalPlan> {
        let mut current = plan;
        for iteration in 0..self.max_iterations {
            let mut changed_this_pass = false;
            for rule in &self.rules {
                let transformed = apply_rule(rule.as_ref(), current, ctx)?;
                changed_this_pass |= transformed.is_yes();
                current = transformed.into_inner();
            }
            if !changed_this_pass {
                return Ok(current);
            }
            if iteration == self.max_iterations - 1 {
                eprintln!(
                    "basalt optimizer: hit max_iterations ({}) without reaching a fixed point — \
                     a rule is likely oscillating; this is a bug in the rule set, not expected \
                     behavior",
                    self.max_iterations
                );
            }
        }
        Ok(current)
    }
}

fn apply_rule(
    rule: &dyn OptimizerRule,
    plan: LogicalPlan,
    ctx: &dyn OptimizerContext,
) -> Result<Transformed<LogicalPlan>> {
    use super::tree_node::TreeNode;
    match rule.apply_order() {
        ApplyOrder::BottomUp => plan.transform_up(&mut |node| rule.apply(node, ctx)),
        ApplyOrder::TopDown => plan.transform_down(&mut |node| rule.apply(node, ctx)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema};

    fn schema() -> crate::types::schema::SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap())
    }

    fn scan_plan() -> LogicalPlan {
        let source = Arc::new(MemoryTableSource::new(schema(), vec![]));
        LogicalPlanBuilder::scan("t", source).build().as_ref().clone()
    }

    #[derive(Debug)]
    struct CountingRule {
        applications: std::sync::atomic::AtomicUsize,
        max_applications: usize,
    }

    impl OptimizerRule for CountingRule {
        fn name(&self) -> &str {
            "counting_rule"
        }
        fn apply(
            &self,
            plan: LogicalPlan,
            _ctx: &dyn OptimizerContext,
        ) -> Result<Transformed<LogicalPlan>> {
            let n = self
                .applications
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n < self.max_applications {
                Ok(Transformed::Yes(plan))
            } else {
                Ok(Transformed::No(plan))
            }
        }
    }

    #[test]
    fn stops_iterating_once_no_rule_reports_a_change() {
        let rule = Arc::new(CountingRule {
            applications: std::sync::atomic::AtomicUsize::new(0),
            max_applications: 3,
        });
        let optimizer = Optimizer::new(vec![rule.clone()]);
        optimizer.optimize(scan_plan(), &NoStatistics).unwrap();
        // 3 "Yes" iterations + 1 "No" iteration that stops the loop = 4.
        assert_eq!(
            rule.applications.load(std::sync::atomic::Ordering::SeqCst),
            4
        );
    }

    #[test]
    fn respects_max_iterations_cap_for_an_oscillating_rule() {
        #[derive(Debug)]
        struct AlwaysChanges;
        impl OptimizerRule for AlwaysChanges {
            fn name(&self) -> &str {
                "always_changes"
            }
            fn apply(
                &self,
                plan: LogicalPlan,
                _ctx: &dyn OptimizerContext,
            ) -> Result<Transformed<LogicalPlan>> {
                Ok(Transformed::Yes(plan))
            }
        }
        let mut optimizer = Optimizer::new(vec![Arc::new(AlwaysChanges)]);
        optimizer.max_iterations = 3;
        // Must terminate rather than loop forever.
        optimizer.optimize(scan_plan(), &NoStatistics).unwrap();
    }
}
