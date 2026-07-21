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

use super::graph::{full_set, singleton, JoinEdge, JoinGraph, RelationSet};
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
            return Err(BasaltError::Internal(
                "cannot join zero relations".to_string(),
            ));
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

    pub(crate) fn base_candidate(&self, i: usize) -> PlanCandidate {
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

    fn best_split(
        &self,
        memo: &HashMap<RelationSet, PlanCandidate>,
        s: RelationSet,
    ) -> Option<PlanCandidate> {
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
                let better = best.as_ref().is_none_or(|b| {
                    candidate.cost.total(self.weights) < b.cost.total(self.weights)
                });
                if better {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    /// Builds the `PlanCandidate` for joining `left`'s and `right`'s
    /// subsets on every edge the graph has between them.
    pub(crate) fn build_join(
        &self,
        left: &PlanCandidate,
        right: &PlanCandidate,
    ) -> Result<PlanCandidate> {
        let left_set = left
            .column_order
            .iter()
            .map(|&(r, _)| r)
            .collect::<std::collections::HashSet<_>>();
        let right_set = right
            .column_order
            .iter()
            .map(|&(r, _)| r)
            .collect::<std::collections::HashSet<_>>();
        let left_mask = left_set.iter().fold(0u64, |acc, &r| acc | singleton(r));
        let right_mask = right_set.iter().fold(0u64, |acc, &r| acc | singleton(r));
        let edges = self.graph.edges_between(left_mask, right_mask);
        if edges.is_empty() {
            return Err(BasaltError::Internal(
                "no join edge between these subsets".to_string(),
            ));
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
        let cardinality = join_cardinality(
            &left_stats,
            &right_stats,
            &on_local_indices,
            JoinType::Inner,
        );

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

        Ok(PlanCandidate {
            plan,
            cost,
            cardinality,
            column_order,
            column_statistics,
        })
    }

    /// Maps a graph edge's `(relation, column)` endpoints to `Expr::Column`
    /// nodes indexed relative to `left`'s and `right`'s own local schemas
    /// (what `LogicalPlan::Join.on` expects), pulling the real
    /// `DataType`/nullability from that relation's actual schema.
    fn edge_to_columns(
        &self,
        edge: &JoinEdge,
        left: &PlanCandidate,
        right: &PlanCandidate,
    ) -> Result<(Expr, Expr)> {
        let left_pos = left
            .column_order
            .iter()
            .position(|&(r, c)| r == edge.left_relation && c == edge.left_column)
            .ok_or_else(|| {
                BasaltError::Internal(
                    "join edge references a column not in the left subset".to_string(),
                )
            })?;
        let right_pos = right
            .column_order
            .iter()
            .position(|&(r, c)| r == edge.right_relation && c == edge.right_column)
            .ok_or_else(|| {
                BasaltError::Internal(
                    "join edge references a column not in the right subset".to_string(),
                )
            })?;

        let left_field = &self.graph.relations[edge.left_relation]
            .plan
            .schema()
            .fields()[edge.left_column];
        let right_field = &self.graph.relations[edge.right_relation]
            .plan
            .schema()
            .fields()[edge.right_column];
        Ok((
            Expr::Column {
                index: left_pos,
                data_type: left_field.data_type,
                nullable: left_field.nullable,
            },
            Expr::Column {
                index: right_pos,
                data_type: right_field.data_type,
                nullable: right_field.nullable,
            },
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

/// `DPccp` — Moerkotte & Neumann, *"Analysis of Two Existing and One New
/// Dynamic Programming Algorithm for the Generation of Optimal Bushy Join
/// Trees"* (VLDB 2006). See design-docs/basalt-phase3-lld.md §7.3.
///
/// `DPsize` wastes most of its time enumerating subsets that aren't
/// connected and splits that have no join edge — `all_subsets_of_size`
/// above walks every one of the `2^n` bitmasks of a given popcount,
/// regardless of whether the graph even has an edge that could connect
/// them. `DPccp`'s idea is to **only ever generate connected subgraphs
/// in the first place**, via neighborhood expansion from each relation
/// (`enumerate_connected_subsets` below): starting from `{v}`, repeatedly
/// grow by adding a nonempty subset of the current frontier's neighbors,
/// with a "forbidden" set of already-tried relations so the same
/// connected subgraph is never generated twice from two different
/// starting points. Work is proportional to the number of *connected*
/// subgraphs, not `3^n`.
///
/// This implementation enumerates connected subgraphs via that expansion
/// (the paper's `EnumerateCsg` idea) but, rather than the paper's
/// `EnumerateCmp`/forbidden-set machinery for generating each connected
/// subgraph's complements without redundant work, pairs connected
/// subgraphs directly from the same enumerated list (filtering for
/// disjointness and connectivity). That's a real, measured scope
/// narrowing: it still never touches a disconnected subset (`DPccp`'s
/// actual saving over `DPsize`), but the pairing step is `O(m)` per
/// connected subgraph (`m` = number of connected subgraphs) rather than
/// the paper's more tightly bounded complement enumeration. Verified
/// against `DPsize` for matching optimal cost on the same graphs (see
/// `dpccp_matches_dpsize_on_the_same_graph` below) — the LLD's own
/// suggested test.
pub fn enumerate_connected_subsets(graph: &JoinGraph) -> Vec<RelationSet> {
    let n = graph.num_relations();
    let mut result = Vec::new();
    for v in 0..n {
        let seed = singleton(v);
        result.push(seed);
        // Forbid this relation and every relation processed as a seed
        // before it, so a connected subgraph is only ever generated once,
        // from its lowest-indexed member.
        let forbidden = (0..=v).fold(0u64, |acc, i| acc | singleton(i));
        enumerate_csg_rec(graph, seed, forbidden, &mut result);
    }
    result
}

fn enumerate_csg_rec(
    graph: &JoinGraph,
    s: RelationSet,
    forbidden: RelationSet,
    result: &mut Vec<RelationSet>,
) {
    let frontier = graph.neighbors(s) & !forbidden;
    if frontier == 0 {
        return;
    }
    for expansion in JoinGraph::all_nonempty_submasks(frontier) {
        result.push(s | expansion);
    }
    let new_forbidden = forbidden | frontier;
    for expansion in JoinGraph::all_nonempty_submasks(frontier) {
        enumerate_csg_rec(graph, s | expansion, new_forbidden, result);
    }
}

pub struct DpccpJoinOptimizer<'a> {
    inner: DpJoinOptimizer<'a>,
}

impl<'a> DpccpJoinOptimizer<'a> {
    pub fn new(graph: &'a JoinGraph, weights: &'a CostWeights) -> Self {
        DpccpJoinOptimizer {
            inner: DpJoinOptimizer::new(graph, weights),
        }
    }

    /// # Errors
    /// Errors if the join graph is disconnected, exactly like
    /// `DpJoinOptimizer::optimize`.
    pub fn optimize(&self) -> Result<PlanCandidate> {
        let graph = self.inner.graph;
        let n = graph.num_relations();
        if n == 0 {
            return Err(BasaltError::Internal(
                "cannot join zero relations".to_string(),
            ));
        }

        let mut connected = enumerate_connected_subsets(graph);
        connected.sort_by_key(|s| s.count_ones());
        connected.dedup();

        let mut memo: HashMap<RelationSet, PlanCandidate> = HashMap::new();
        for i in 0..n {
            memo.insert(singleton(i), self.inner.base_candidate(i));
        }

        for &s in &connected {
            if s.count_ones() < 2 {
                continue; // Singletons are already the base case.
            }
            let mut best: Option<PlanCandidate> = None;
            // Only pair `s` with other *connected* subgraphs already in
            // the memo — this is the actual saving over DPsize: no
            // disconnected subset is ever considered as a split.
            for &s1 in &connected {
                if s1 == s || s1 & s != s1 || s1.count_ones() >= s.count_ones() {
                    continue; // s1 must be a proper subset of s.
                }
                let s2 = s & !s1;
                if !memo.contains_key(&s1) || !memo.contains_key(&s2) {
                    continue;
                }
                if !graph.is_connected(s1, s2) {
                    continue;
                }
                let (c1, c2) = (&memo[&s1], &memo[&s2]);
                if let Ok(candidate) = self.inner.build_join(c1, c2) {
                    let better = best.as_ref().is_none_or(|b| {
                        candidate.cost.total(self.inner.weights) < b.cost.total(self.inner.weights)
                    });
                    if better {
                        best = Some(candidate);
                    }
                }
            }
            if let Some(b) = best {
                memo.insert(s, b);
            }
        }

        memo.remove(&full_set(n)).ok_or_else(|| {
            BasaltError::Internal(
                "join graph is disconnected — no join order avoids a cross product".to_string(),
            )
        })
    }
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

    fn chain_graph() -> JoinGraph {
        // a -- b -- c, a chain of 3 relations.
        let relations = vec![
            relation("a", 1000, 100),
            relation("b", 100, 100),
            relation("c", 10000, 500),
        ];
        let edges = vec![
            JoinEdge {
                left_relation: 0,
                left_column: 0,
                right_relation: 1,
                right_column: 0,
            },
            JoinEdge {
                left_relation: 1,
                left_column: 0,
                right_relation: 2,
                right_column: 0,
            },
        ];
        JoinGraph::new(relations, edges)
    }

    /// Relation 0 connected to every other relation, none of the others
    /// connected to each other — a star.
    fn star_graph(n: usize) -> JoinGraph {
        let relations: Vec<_> = (0..n)
            .map(|i| relation(&format!("t{i}"), 100 * (i + 1), 20 + i))
            .collect();
        let edges: Vec<_> = (1..n)
            .map(|i| JoinEdge {
                left_relation: 0,
                left_column: 0,
                right_relation: i,
                right_column: 0,
            })
            .collect();
        JoinGraph::new(relations, edges)
    }

    /// Every relation connected to every other — a clique.
    fn clique_graph(n: usize) -> JoinGraph {
        let relations: Vec<_> = (0..n)
            .map(|i| relation(&format!("t{i}"), 50 * (i + 1), 10 + i))
            .collect();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push(JoinEdge {
                    left_relation: i,
                    left_column: 0,
                    right_relation: j,
                    right_column: 0,
                });
            }
        }
        JoinGraph::new(relations, edges)
    }

    /// Compares `cpu` (and cardinality, transitively, since `cpu` is
    /// derived from it) rather than `cost.total()`: on graphs with more
    /// than one structurally valid bushy shape (star, clique — a chain has
    /// only one shape worth considering), `DPsize` and `DPccp` can land on
    /// *different* tree shapes that both minimize `cpu`/`io` (the additive,
    /// truly DP-optimal-substructure-respecting dimensions) but differ in
    /// `memory`, because `Cost::combine` takes the **max** for memory, not
    /// the sum — a documented approximation (`cost::model::Cost::combine`'s
    /// own doc comment) that doesn't obey strict bottom-up optimal
    /// substructure the way an additive cost would. Found by this exact
    /// test failing on `total()` during development: both algorithms
    /// agreed on cardinality and `cpu` to the last digit, and differed
    /// only in which of several equally-cheap-in-cpu tree shapes each
    /// happened to pick, which then cascaded into a different `memory`
    /// figure. Not a correctness bug in either enumerator — a real,
    /// pre-existing property of the memory-max cost model this discovery
    /// is worth having on record.
    #[test]
    fn dpccp_matches_dpsize_on_a_star_graph() {
        let graph = star_graph(4);
        let weights = CostWeights::defaults();
        let dpsize = DpJoinOptimizer::new(&graph, &weights).optimize().unwrap();
        let dpccp = DpccpJoinOptimizer::new(&graph, &weights)
            .optimize()
            .unwrap();
        assert!(
            (dpsize.cost.cpu - dpccp.cost.cpu).abs() < 1e-6,
            "star graph cpu: DPccp {} vs DPsize {}",
            dpccp.cost.cpu,
            dpsize.cost.cpu
        );
        assert_eq!(dpsize.cardinality, dpccp.cardinality);
    }

    #[test]
    fn dpccp_matches_dpsize_on_a_clique_graph() {
        let graph = clique_graph(4);
        let weights = CostWeights::defaults();
        let dpsize = DpJoinOptimizer::new(&graph, &weights).optimize().unwrap();
        let dpccp = DpccpJoinOptimizer::new(&graph, &weights)
            .optimize()
            .unwrap();
        assert!(
            (dpsize.cost.cpu - dpccp.cost.cpu).abs() < 1e-6,
            "clique graph cpu: DPccp {} vs DPsize {}",
            dpccp.cost.cpu,
            dpsize.cost.cpu
        );
        assert_eq!(dpsize.cardinality, dpccp.cardinality);
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
        assert!(matches!(
            result.plan.as_ref(),
            LogicalPlan::TableScan { .. }
        ));
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
        let dp_cost = DpJoinOptimizer::new(&graph, &weights)
            .optimize()
            .unwrap()
            .cost
            .total(&weights);
        let brute_force_cost = brute_force_best_cost(&graph, &weights);
        assert!(
            (dp_cost - brute_force_cost).abs() < 1e-6,
            "DP cost {dp_cost} should match brute-force optimum {brute_force_cost}"
        );
    }

    #[test]
    fn enumerate_connected_subsets_never_includes_a_disconnected_set() {
        let graph = chain_graph(); // a -- b -- c
        let connected = enumerate_connected_subsets(&graph);
        // {a, c} (0b101) has no edge between a and c directly — it's only
        // reachable through b — so it must never be enumerated.
        assert!(!connected.contains(&0b101));
        // Every actually-connected subset must appear: {a}, {b}, {c},
        // {a,b}, {b,c}, {a,b,c}.
        for expected in [0b001, 0b010, 0b100, 0b011, 0b110, 0b111] {
            assert!(
                connected.contains(&expected),
                "missing connected subset {expected:03b}"
            );
        }
    }

    #[test]
    fn dpccp_matches_dpsize_on_the_same_graph() {
        let graph = chain_graph();
        let weights = CostWeights::defaults();
        let dpsize_cost = DpJoinOptimizer::new(&graph, &weights)
            .optimize()
            .unwrap()
            .cost
            .total(&weights);
        let dpccp_cost = DpccpJoinOptimizer::new(&graph, &weights)
            .optimize()
            .unwrap()
            .cost
            .total(&weights);
        assert!(
            (dpsize_cost - dpccp_cost).abs() < 1e-6,
            "DPccp cost {dpccp_cost} should match DPsize's optimum {dpsize_cost}"
        );
    }

    #[test]
    fn dpccp_errors_on_a_disconnected_graph() {
        let relations = vec![
            crate::optimizer::join_order::graph::RelationNode {
                plan: LogicalPlanBuilder::scan(
                    "a",
                    Arc::new(MemoryTableSource::new(schema("a"), vec![])),
                )
                .build(),
                stats: Arc::new({
                    let mut s = TableStatistics::unknown(1);
                    s.num_rows = Precision::Exact(10);
                    s
                }),
            },
            crate::optimizer::join_order::graph::RelationNode {
                plan: LogicalPlanBuilder::scan(
                    "b",
                    Arc::new(MemoryTableSource::new(schema("b"), vec![])),
                )
                .build(),
                stats: Arc::new({
                    let mut s = TableStatistics::unknown(1);
                    s.num_rows = Precision::Exact(10);
                    s
                }),
            },
        ];
        let graph = JoinGraph::new(relations, vec![]);
        let weights = CostWeights::defaults();
        assert!(DpccpJoinOptimizer::new(&graph, &weights)
            .optimize()
            .is_err());
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
                let (Some(c1), Some(c2)) = (
                    best_for(graph, weights, opt, s1, cache),
                    best_for(graph, weights, opt, s2, cache),
                ) else {
                    continue;
                };
                if let Ok(candidate) = opt.build_join(&c1, &c2) {
                    let better = best
                        .as_ref()
                        .is_none_or(|b| candidate.cost.total(weights) < b.cost.total(weights));
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
        best_for(
            graph,
            weights,
            &opt,
            full_set(graph.num_relations()),
            &mut cache,
        )
        .expect("connected graph must have a valid join order")
        .cost
        .total(weights)
    }
}
