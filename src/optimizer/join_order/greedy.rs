//! GOO (Greedy Operator Ordering) — the fallback above `DPsize`/`DPccp`'s
//! exact-DP relation-count threshold. See
//! design-docs/basalt-phase3-lld.md §7.4.
//!
//! Repeatedly joins the pair of current subsets whose estimated join
//! result is smallest, until one relation remains. `O(n^3)` (at each of
//! `n` steps, considers every pair among the shrinking set of current
//! subsets), with no optimality guarantee — but empirically decent, and
//! polynomial where exact DP is exponential.

use std::collections::HashMap;

use super::dp::{DpJoinOptimizer, PlanCandidate};
use super::graph::{singleton, JoinGraph, RelationSet};
use crate::error::{BasaltError, Result};
use crate::optimizer::cost::model::CostWeights;

/// # Errors
/// Errors if the join graph is disconnected (mirrors `DpJoinOptimizer::optimize`'s
/// same refusal to silently fall back to a cross product).
pub fn greedy_operator_ordering(graph: &JoinGraph, weights: &CostWeights) -> Result<PlanCandidate> {
    let n = graph.num_relations();
    if n == 0 {
        return Err(BasaltError::Internal(
            "cannot join zero relations".to_string(),
        ));
    }

    let opt = DpJoinOptimizer::new(graph, weights);
    let mut current: HashMap<RelationSet, PlanCandidate> = (0..n)
        .map(|i| (singleton(i), opt.base_candidate(i)))
        .collect();

    while current.len() > 1 {
        let sets: Vec<RelationSet> = current.keys().copied().collect();
        let mut best: Option<(RelationSet, RelationSet, PlanCandidate)> = None;

        for (i, &s1) in sets.iter().enumerate() {
            for &s2 in &sets[i + 1..] {
                if !graph.is_connected(s1, s2) {
                    continue;
                }
                let (c1, c2) = (&current[&s1], &current[&s2]);
                let Ok(candidate) = opt.build_join(c1, c2) else {
                    continue;
                };
                // GOO's own greedy metric: smallest *estimated result*, not
                // lowest cost — the LLD's framing ("repeatedly join the
                // pair whose estimated result is smallest"). Cardinality
                // rather than cost keeps this a genuinely different,
                // cheaper heuristic from a cost-based search, not DP
                // restricted to pairwise steps.
                let candidate_card = candidate
                    .cardinality
                    .get_value()
                    .copied()
                    .unwrap_or(usize::MAX);
                let better = best.as_ref().is_none_or(|(_, _, b)| {
                    candidate_card < b.cardinality.get_value().copied().unwrap_or(usize::MAX)
                });
                if better {
                    best = Some((s1, s2, candidate));
                }
            }
        }

        let Some((s1, s2, candidate)) = best else {
            return Err(BasaltError::Internal(
                "join graph is disconnected — no join order avoids a cross product".to_string(),
            ));
        };
        current.remove(&s1);
        current.remove(&s2);
        current.insert(s1 | s2, candidate);
    }

    current
        .into_values()
        .next()
        .ok_or_else(|| BasaltError::Internal("greedy join ordering produced no plan".to_string()))
}

/// Picks `DPccp` below a measured relation-count threshold, `GOO` above
/// it — exponential planning time must never exceed execution time. The
/// threshold should be measured (time the enumerator on increasing
/// relation counts and find where planning starts to rival execution),
/// not guessed; `DEFAULT_EXACT_THRESHOLD` is a reasonable starting point
/// pending that measurement, which is a `BENCHMARKS.md` follow-up rather
/// than a number this function invents authoritatively.
pub const DEFAULT_EXACT_THRESHOLD: usize = 12;

/// # Errors
/// Errors if the join graph is disconnected.
pub fn optimize_join_order(
    graph: &JoinGraph,
    weights: &CostWeights,
    threshold: usize,
) -> Result<PlanCandidate> {
    if graph.num_relations() <= threshold {
        DpJoinOptimizer::new(graph, weights).optimize()
    } else {
        greedy_operator_ordering(graph, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::logical_plan::plan::LogicalPlan;
    use crate::optimizer::join_order::graph::{JoinEdge, RelationNode};
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::statistics::{Precision, TableStatistics};
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema, SchemaRef};
    use std::sync::Arc;

    fn schema(name: &str) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]).unwrap())
    }

    fn relation(name: &str, num_rows: usize, ndv: usize) -> RelationNode {
        let plan =
            LogicalPlanBuilder::scan(name, Arc::new(MemoryTableSource::new(schema(name), vec![])))
                .build();
        let mut stats = TableStatistics::unknown(1);
        stats.num_rows = Precision::Exact(num_rows);
        stats.column_statistics[0].distinct_count = Precision::Exact(ndv);
        RelationNode {
            plan,
            stats: Arc::new(stats),
        }
    }

    fn chain_graph(n: usize) -> JoinGraph {
        let relations: Vec<RelationNode> = (0..n)
            .map(|i| relation(&format!("t{i}"), 100 + i, 50))
            .collect();
        let edges: Vec<JoinEdge> = (0..n - 1)
            .map(|i| JoinEdge {
                left_relation: i,
                left_column: 0,
                right_relation: i + 1,
                right_column: 0,
            })
            .collect();
        JoinGraph::new(relations, edges)
    }

    #[test]
    fn greedy_joins_every_relation_in_a_chain() {
        let graph = chain_graph(5);
        let weights = CostWeights::defaults();
        let result = greedy_operator_ordering(&graph, &weights).unwrap();
        assert_eq!(result.column_order.len(), 5);
        assert!(matches!(result.plan.as_ref(), LogicalPlan::Join { .. }));
    }

    #[test]
    fn greedy_errors_on_a_disconnected_graph() {
        let relations = vec![relation("a", 10, 1), relation("b", 10, 1)];
        let graph = JoinGraph::new(relations, vec![]);
        let weights = CostWeights::defaults();
        assert!(greedy_operator_ordering(&graph, &weights).is_err());
    }

    #[test]
    fn optimize_join_order_picks_dp_below_threshold_and_greedy_above() {
        let graph = chain_graph(4);
        let weights = CostWeights::defaults();
        // Both should still succeed and join everything, regardless of
        // which algorithm was used.
        let via_low_threshold = optimize_join_order(&graph, &weights, 2).unwrap();
        let via_high_threshold = optimize_join_order(&graph, &weights, 100).unwrap();
        assert_eq!(via_low_threshold.column_order.len(), 4);
        assert_eq!(via_high_threshold.column_order.len(), 4);
    }
}
