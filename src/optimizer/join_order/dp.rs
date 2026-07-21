//! `DPsize` — the Selinger-style bottom-up dynamic program for join order
//! enumeration. See design-docs/basalt-phase3-lld.md §7.2.
//!
//! ```text
//! for size in 2..=n:
//!     for each connected subset S of size `size`:
//!         for each split S = S1 ∪ S2 (S1, S2 disjoint, both connected, edge between them):
//!             cost = cost(best[S1]) + cost(best[S2]) + join_cost(best[S1], best[S2])
//!             if cost < best[S].cost:
//!                 best[S] = Join(best[S1], best[S2])
//! ```
//!
//! `best[full_set]` is the optimal plan. The optimal-substructure property
//! that makes this valid: the best plan for a set of relations contains
//! the best plans for its subsets — true because cost is additive and
//! cardinality depends only on the set, not the order it was built in
//! (that second part is an assumption, and it's why "interesting orders",
//! §7.5, complicate things — not implemented here, see that section's
//! note).
//!
//! `O(3^n)` for bushy plans (every subset times every split of it). Fine
//! to about 10-12 relations; `join_order::greedy` is the fallback above
//! that, and building `DPsize` first — even though `DPccp` (§7.3) is
//! strictly better — is deliberate: the DP *structure* is the lesson here,
//! and it's much easier to debug than `DPccp` once, before adding that
//! algorithm's own optimization on top.

use std::collections::HashMap;
use std::sync::Arc;

use super::graph::{full_set, singleton, JoinGraph, JoinEdge, RelationSet};
use crate::error::{BasaltError, Result};
use crate::expr::expr::Expr;
use crate::logical_plan::plan::{JoinType, LogicalPlan};
use crate::optimizer::cardinality::join_card::join_cardinality;
use crate::optimizer::cost::model::{Cost, CostWeights};
use crate::optimizer::cost::operators::hash_join_cost;
use crate::statistics::{ColumnStatistics, Precision, TableStatistics};
use crate::types::schema::Schema;

/// A best-known plan for one relation subset: its logical plan, estimated
/// cost and cardinality, and enough bookkeeping (`column_order`,
/// `column_statistics`) to build the *next* join on top of it correctly —
/// each subset's plan schema is the concatenation of its relations' own
/// schemas in whatever order they were joined, so building a further join
/// edge's `on` condition needs to know exactly where each column landed.
#[derive(Clone)]
pub struct PlanCandidate {
    pub plan: Arc<LogicalPlan>,
    pub cost: Cost,
    pub cardinality: Precision<usize>,
    /// `(relation_index, local_column_index)` for every output column, in
    /// this candidate's actual schema order.
    pub column_order: Vec<(usize, usize)>,
    /// Parallel to `column_order`.
    pub column_statistics: Vec<ColumnStatistics>,
}

pub struct DpJoinOptimizer<'a> {
    pub graph: &'a JoinGraph,
    pub weights: &'a CostWeights,
}

impl<'a> DpJoinOptimizer<'a> {
    pub fn new(graph: &'a JoinGraph, weights: &'a CostWeights) -> Self {
        DpJoinOptimizer { graph, weights }
    }

    /// # Errors
    /// Errors if the join graph is disconnected (no join order joins every
    /// relation without a cross product) — deliberately not silently
    /// falling back to a cross product, since that's a real, surprising
    /// cost cliff a caller should decide about explicitly (see
    /// `EliminateCrossJoin`'s doc comment for why cross products are
    /// avoided elsewhere in this pipeline too).
    pub fn optimize(&self) -> Result<PlanCandidate> {
        let n = self.graph.num_relations();
        if n == 0 {
            return Err(BasaltError::Internal("cannot join zero relations".to_string()));
        }
        let mut memo: HashMap<RelationSet, PlanCandidate> = HashMap::new();
        for i in 0..n {
            memo.insert(singleton(i), self.base_candidate(i));
        }

        for size in 2..=n {
            for s in all_subsets_of_size(n, size) {
                if let Some(candidate) = self.best_split(&memo, s) {
                    memo.insert(s, candidate);
                }
            }
        }

        memo.remove(&full_set(n)).ok_or_else(|| {
            BasaltError::Internal(
                "join graph is disconnected — no join order avoids a cross product".to_string(),
            )
        })
    }

    fn base_candidate(&self, i: usize) -> PlanCandidate {
        let relation = &self.graph.relations[i];
        let num_cols = relation.plan.schema().fields().len();
        PlanCandidate {
            plan: relation.plan.clone(),
            cost: Cost::ZERO,
            cardinality: relation.stats.num_rows,
            column_order: (0..num_cols).map(|c| (i, c)).collect(),
            column_statistics: relation.stats.column_statistics.clone(),
        }
    }

    fn best_split(&self, memo: &HashMap<RelationSet, PlanCandidate>, s: RelationSet) -> Option<PlanCandidate> {
        let mut best: Option<PlanCandidate> = None;
        for s1 in JoinGraph::sub_masks(s) {
            let s2 = s & !s1;
            if s1 >= s2 {
                continue; // Each unordered {S1, S2} split considered once.
            }
            if !self.graph.is_connected(s1, s2) {
                continue;
            }
            let (Some(c1), Some(c2)) = (memo.get(&s1), memo.get(&s2)) else {
                continue;
            };
            if let Ok(candidate) = self.build_join(c1, c2) {
                let better = best
                    .as_ref()
                    .is_none_or(|b| candidate.cost.total(self.weights) < b.cost.total(self.weights));
                if better {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    /// Builds the `PlanCandidate` for joining `left`'s and `right`'s
    /// subsets on every edge the graph has between them.
    fn build_join(&self, left: &PlanCandidate, right: &PlanCandidate) -> Result<PlanCandidate> {
        let left_set = left.column_order.iter().map(|&(r, _)| r).collect::<std::collections::HashSet<_>>();
        let right_set = right.column_order.iter().map(|&(r, _)| r).collect::<std::collections::HashSet<_>>();
        let left_mask = left_set.iter().fold(0u64, |acc, &r| acc | singleton(r));
        let right_mask = right_set.iter().fold(0u64, |acc, &r| acc | singleton(r));
        let edges = self.graph.edges_between(left_mask, right_mask);
        if edges.is_empty() {
            return Err(BasaltError::Internal("no join edge between these subsets".to_string()));
        }

        let on: Vec<(Expr, Expr)> = edges
            .iter()
            .map(|e| self.edge_to_columns(e, left, right))
            .collect::<Result<_>>()?;
        let on_local_indices: Vec<(usize, usize)> = on
            .iter()
            .map(|(l, r)| match (l, r) {
                (Expr::Column { index: li, .. }, Expr::Column { index: ri, .. }) => (*li, *ri),
                _ => unreachable!("edge_to_columns always returns Column exprs"),
            })
            .collect();

        let left_stats = TableStatistics {
            num_rows: left.cardinality,
            total_byte_size: Precision::Absent,
            column_statistics: left.column_statistics.clone(),
        };
        let right_stats = TableStatistics {
            num_rows: right.cardinality,
            total_byte_size: Precision::Absent,
            column_statistics: right.column_statistics.clone(),
        };
        let cardinality = join_cardinality(&left_stats, &right_stats, &on_local_indices, JoinType::Inner);

        // A byte-width-per-row estimate isn't tracked per column in this
        // engine yet; 8.0 bytes/column is a documented placeholder — cost
        // is for *ranking* plans (see `cost::model`'s doc comment), and
        // this constant applies uniformly to every candidate, so it
        // doesn't bias the comparison between them.
        const ASSUMED_COLUMN_WIDTH: f64 = 8.0;
        let build_rows = precision_as_f64(left.cardinality);
        let probe_rows = precision_as_f64(right.cardinality);
        let output_rows = precision_as_f64(cardinality);
        let row_width = left.column_order.len() as f64 * ASSUMED_COLUMN_WIDTH;
        let join_cost = hash_join_cost(build_rows, probe_rows, output_rows, row_width);
        let cost = left.cost.combine(&right.cost).combine(&join_cost);

        let mut column_order = left.column_order.clone();
        column_order.extend(right.column_order.iter().copied());
        let mut column_statistics = left.column_statistics.clone();
        column_statistics.extend(right.column_statistics.iter().cloned());

        let mut fields = Vec::with_capacity(column_order.len());
        for &(rel, col) in &column_order {
            fields.push(self.graph.relations[rel].plan.schema().fields()[col].clone());
        }
        let schema = Arc::new(Schema::new_allow_duplicate_names(fields));

        let plan = Arc::new(LogicalPlan::Join {
            left: left.plan.clone(),
            right: right.plan.clone(),
            on,
            filter: None,
            join_type: JoinType::Inner,
            schema,
        });

        Ok(PlanCandidate { plan, cost, cardinality, column_order, column_statistics })
    }

    /// Maps a graph edge's `(relation, column)` endpoints to `Expr::Column`
    /// nodes indexed relative to `left`'s and `right`'s own local schemas
    /// (what `LogicalPlan::Join.on` expects), pulling the real
    /// `DataType`/nullability from that relation's actual schema.
    fn edge_to_columns(&self, edge: &JoinEdge, left: &PlanCandidate, right: &PlanCandidate) -> Result<(Expr, Expr)> {
        let left_pos = left
            .column_order
            .iter()
            .position(|&(r, c)| r == edge.left_relation && c == edge.left_column)
            .ok_or_else(|| BasaltError::Internal("join edge references a column not in the left subset".to_string()))?;
        let right_pos = right
            .column_order
            .iter()
            .position(|&(r, c)| r == edge.right_relation && c == edge.right_column)
            .ok_or_else(|| BasaltError::Internal("join edge references a column not in the right subset".to_string()))?;

        let left_field = &self.graph.relations[edge.left_relation].plan.schema().fields()[edge.left_column];
        let right_field = &self.graph.relations[edge.right_relation].plan.schema().fields()[edge.right_column];
        Ok((
            Expr::Column { index: left_pos, data_type: left_field.data_type, nullable: left_field.nullable },
            Expr::Column { index: right_pos, data_type: right_field.data_type, nullable: right_field.nullable },
        ))
    }
}

fn precision_as_f64(p: Precision<usize>) -> f64 {
    // A documented fallback for ranking purposes when a subset's row count
    // is unknown (propagated `Absent` from missing base statistics) —
    // matches this module's "ranking, not prediction" cost philosophy.
    const UNKNOWN_ROWS_FALLBACK: f64 = 1000.0;
    p.get_value().map_or(UNKNOWN_ROWS_FALLBACK, |&v| v as f64)
}

fn all_subsets_of_size(n: usize, size: usize) -> impl Iterator<Item = RelationSet> {
    let full = full_set(n);
    (0..=full).filter(move |s| s.count_ones() as usize == size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::builder::LogicalPlanBuilder;
    use crate::optimizer::join_order::graph::RelationNode;
    use crate::physical_plan::scan::MemoryTableSource;
    use crate::types::data_type::DataType;
    use crate::types::schema::{Field, Schema, SchemaRef};

    fn schema(name: &str) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]).unwrap())
    }

    fn relation(name: &str, num_rows: usize, ndv: usize) -> RelationNode {
        let plan = LogicalPlanBuilder::scan(name, Arc::new(MemoryTableSource::new(schema(name), vec![])))
            .build();
        let mut stats = TableStatistics::unknown(1);
        stats.num_rows = Precision::Exact(num_rows);
        stats.column_statistics[0].distinct_count = Precision::Exact(ndv);
        RelationNode { plan, stats: Arc::new(stats) }
    }

    fn chain_graph() -> JoinGraph {
        // a -- b -- c, a chain of 3 relations.
        let relations = vec![relation("a", 1000, 100), relation("b", 100, 100), relation("c", 10000, 500)];
        let edges = vec![
            JoinEdge { left_relation: 0, left_column: 0, right_relation: 1, right_column: 0 },
            JoinEdge { left_relation: 1, left_column: 0, right_relation: 2, right_column: 0 },
        ];
        JoinGraph::new(relations, edges)
    }

    #[test]
    fn optimizes_a_chain_of_three_relations() {
        let graph = chain_graph();
        let weights = CostWeights::defaults();
        let result = DpJoinOptimizer::new(&graph, &weights).optimize().unwrap();
        assert_eq!(result.column_order.len(), 3);
        assert!(matches!(result.plan.as_ref(), LogicalPlan::Join { .. }));
    }

    #[test]
    fn disconnected_graph_errors_instead_of_defaulting_to_a_cross_product() {
        let relations = vec![relation("a", 10, 1), relation("b", 10, 1)];
        let graph = JoinGraph::new(relations, vec![]); // no edges at all
        let weights = CostWeights::defaults();
        assert!(DpJoinOptimizer::new(&graph, &weights).optimize().is_err());
    }

    #[test]
    fn single_relation_returns_it_unjoined() {
        let relations = vec![relation("a", 10, 1)];
        let graph = JoinGraph::new(relations, vec![]);
        let weights = CostWeights::defaults();
        let result = DpJoinOptimizer::new(&graph, &weights).optimize().unwrap();
        assert!(matches!(result.plan.as_ref(), LogicalPlan::TableScan { .. }));
    }

    /// The brute-force oracle: for a small graph, enumerate every valid
    /// join order (every full binary tree over the relations respecting
    /// connectivity) exhaustively and assert DP found the same minimum
    /// cost. The same trick Phase 2 used for its row-at-a-time execution
    /// oracle — for few enough relations there are few enough orderings to
    /// check by hand, so passing here means the DP can be trusted at
    /// larger relation counts too.
    #[test]
    fn dp_finds_the_same_optimum_as_brute_force_enumeration() {
        let graph = chain_graph();
        let weights = CostWeights::defaults();
        let dp_cost = DpJoinOptimizer::new(&graph, &weights).optimize().unwrap().cost.total(&weights);
        let brute_force_cost = brute_force_best_cost(&graph, &weights);
        assert!(
            (dp_cost - brute_force_cost).abs() < 1e-6,
            "DP cost {dp_cost} should match brute-force optimum {brute_force_cost}"
        );
    }

    /// Exhaustively tries every way to build up the full relation set via
    /// connected splits, recursively, without memoization — correct but
    /// exponential, fine for the 3-relation graphs this is tested against.
    fn brute_force_best_cost(graph: &JoinGraph, weights: &CostWeights) -> f64 {
        fn best_for(
            graph: &JoinGraph,
            weights: &CostWeights,
            opt: &DpJoinOptimizer,
            s: RelationSet,
            cache: &mut HashMap<RelationSet, PlanCandidate>,
        ) -> Option<PlanCandidate> {
            if s.count_ones() == 1 {
                let i = s.trailing_zeros() as usize;
                return Some(opt.base_candidate(i));
            }
            if let Some(c) = cache.get(&s) {
                return Some(c.clone());
            }
            let mut best: Option<PlanCandidate> = None;
            for s1 in JoinGraph::sub_masks(s) {
                let s2 = s & !s1;
                if s1 >= s2 {
                    continue;
                }
                if !graph.is_connected(s1, s2) {
                    continue;
                }
                let (Some(c1), Some(c2)) =
                    (best_for(graph, weights, opt, s1, cache), best_for(graph, weights, opt, s2, cache))
                else {
                    continue;
                };
                if let Ok(candidate) = opt.build_join(&c1, &c2) {
                    let better = best.as_ref().is_none_or(|b| candidate.cost.total(weights) < b.cost.total(weights));
                    if better {
                        best = Some(candidate);
                    }
                }
            }
            if let Some(b) = &best {
                cache.insert(s, b.clone());
            }
            best
        }

        let opt = DpJoinOptimizer::new(graph, weights);
        let mut cache = HashMap::new();
        best_for(graph, weights, &opt, full_set(graph.num_relations()), &mut cache)
            .expect("connected graph must have a valid join order")
            .cost
            .total(weights)
    }
}
