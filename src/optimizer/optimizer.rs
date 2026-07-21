//! Wires the standard rule set into an [`Optimizer`](super::Optimizer).
//!
//! `Optimizer`/`OptimizerRule`/`ApplyOrder` themselves live in `rule.rs`
//! (this file just assembles the default pipeline) — a small deviation from
//! the LLD's exact file split, since the pass-manager struct and the trait
//! it runs are one cohesive unit in practice.

use std::sync::Arc;

use super::rule::{Optimizer, OptimizerRule};
use super::rules::{
    common_subexpr::CommonSubexprEliminate, constant_folding::ConstantFolding,
    eliminate_cross_join::EliminateCrossJoin, eliminate_filter::EliminateFilter,
    limit_pushdown::LimitPushdown, merge_filters::MergeFilters,
    predicate_pushdown::PredicatePushdown, projection_pushdown::ProjectionPushdown,
    simplify_expressions::SimplifyExpressions,
};

/// The standard rule pipeline, in the order the LLD's rule catalogue
/// (§5.3) and pushdown module (§6) were written up. Order matters here:
/// `EliminateCrossJoin` must run before `PredicatePushdown`/join ordering
/// ever see a join edge for `FROM a, b WHERE a.id = b.id`-style queries,
/// and folding/simplification before pushdown means pushdown sees already
/// simplified predicates.
pub fn default_optimizer() -> Optimizer {
    let rules: Vec<Arc<dyn OptimizerRule>> = vec![
        Arc::new(ConstantFolding),
        Arc::new(SimplifyExpressions),
        Arc::new(EliminateCrossJoin),
        Arc::new(MergeFilters),
        Arc::new(EliminateFilter),
        Arc::new(CommonSubexprEliminate),
        Arc::new(PredicatePushdown),
        Arc::new(ProjectionPushdown),
        Arc::new(LimitPushdown),
    ];
    Optimizer::new(rules)
}
