//! The join graph: nodes are base relations, edges are join predicates
//! connecting them. See design-docs/basalt-phase3-lld.md §7.1.
//!
//! **A bitmask for the relation set is the whole reason DP is fast.**
//! Subset iteration, union, intersection, and connectivity checks all
//! become single instructions, and the DP table keys on a `u64`. Using a
//! `HashSet<usize>` here would cost an order of magnitude for no benefit.

use std::sync::Arc;

use crate::logical_plan::plan::LogicalPlan;
use crate::statistics::TableStatistics;

/// A subset of relations, as a bitmask. Supports up to 64 relations, which
/// is far beyond where exact DP is viable anyway (`DPsize` is `O(3^n)`).
pub type RelationSet = u64;

pub fn singleton(i: usize) -> RelationSet {
    1u64 << i
}

pub fn full_set(n: usize) -> RelationSet {
    if n >= 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1
    }
}

/// One base relation: its subplan (a `TableScan`, or a `Filter`/
/// `Projection` chain sitting directly above one — whatever `EliminateCrossJoin`
/// left as an enumerable join leaf) and its statistics.
pub struct RelationNode {
    pub plan: Arc<LogicalPlan>,
    pub stats: Arc<TableStatistics>,
}

/// One equi-join predicate, `left_relation.left_column = right_relation.right_column`,
/// with both column indices local to that relation's own schema.
#[derive(Clone, Copy, Debug)]
pub struct JoinEdge {
    pub left_relation: usize,
    pub left_column: usize,
    pub right_relation: usize,
    pub right_column: usize,
}

pub struct JoinGraph {
    pub relations: Vec<RelationNode>,
    pub edges: Vec<JoinEdge>,
}

impl JoinGraph {
    pub fn new(relations: Vec<RelationNode>, edges: Vec<JoinEdge>) -> Self {
        JoinGraph { relations, edges }
    }

    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }

    /// Whether any edge connects a relation in `s1` to a relation in `s2`
    /// (in either direction) — the check that keeps DP from ever
    /// considering a cross product, since a join order that produces one
    /// is almost never optimal and admitting them explodes the search
    /// space for nothing.
    pub fn is_connected(&self, s1: RelationSet, s2: RelationSet) -> bool {
        self.edges.iter().any(|e| {
            let (l, r) = (singleton(e.left_relation), singleton(e.right_relation));
            (l & s1 != 0 && r & s2 != 0) || (r & s1 != 0 && l & s2 != 0)
        })
    }

    /// Every edge connecting `s1` to `s2`, normalized so `left_relation`
    /// is always in `s1` and `right_relation` in `s2` — callers don't need
    /// to re-check which side of the original predicate landed where.
    pub fn edges_between(&self, s1: RelationSet, s2: RelationSet) -> Vec<JoinEdge> {
        self.edges
            .iter()
            .filter_map(|e| {
                let (l, r) = (singleton(e.left_relation), singleton(e.right_relation));
                if l & s1 != 0 && r & s2 != 0 {
                    Some(*e)
                } else if r & s1 != 0 && l & s2 != 0 {
                    Some(JoinEdge {
                        left_relation: e.right_relation,
                        left_column: e.right_column,
                        right_relation: e.left_relation,
                        right_column: e.left_column,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Iterates every nonempty proper submask of `s` — the split points a
    /// DP subset needs to consider.
    pub fn sub_masks(s: RelationSet) -> impl Iterator<Item = RelationSet> {
        let mut sub = (s.wrapping_sub(1)) & s;
        let mut done = false;
        std::iter::from_fn(move || {
            if done {
                return None;
            }
            if sub == 0 {
                done = true;
                return None;
            }
            let current = sub;
            sub = (sub.wrapping_sub(1)) & s;
            Some(current)
        })
    }

    /// Iterates every nonempty submask of `mask`, **including `mask`
    /// itself** — unlike `sub_masks`, which excludes the full set because
    /// it's enumerating *split points* of an already-known set. `DPccp`
    /// (§7.3) uses this to enumerate which neighbors to add when growing a
    /// connected subgraph.
    pub fn all_nonempty_submasks(mask: RelationSet) -> impl Iterator<Item = RelationSet> {
        let mut next = Some(mask);
        std::iter::from_fn(move || {
            let current = next?;
            if current == 0 {
                next = None;
                return None;
            }
            next = if current == 0 { None } else { Some((current.wrapping_sub(1)) & mask) };
            Some(current)
        })
    }

    /// Every relation directly connected to some relation in `s` via an
    /// edge, excluding `s` itself — the frontier `DPccp`'s connected-subgraph
    /// enumeration grows into.
    pub fn neighbors(&self, s: RelationSet) -> RelationSet {
        let mut result = 0u64;
        for e in &self.edges {
            let (l, r) = (singleton(e.left_relation), singleton(e.right_relation));
            if l & s != 0 {
                result |= r;
            }
            if r & s != 0 {
                result |= l;
            }
        }
        result & !s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_nonempty_submasks_includes_the_mask_itself_and_excludes_zero() {
        let subs: Vec<RelationSet> = JoinGraph::all_nonempty_submasks(0b101).collect();
        assert!(subs.contains(&0b101));
        assert!(!subs.contains(&0));
        assert_eq!(subs.len(), 3); // {01, 100, 101}
    }

    #[test]
    fn all_nonempty_submasks_of_a_single_bit_yields_only_itself() {
        let subs: Vec<RelationSet> = JoinGraph::all_nonempty_submasks(0b100).collect();
        assert_eq!(subs, vec![0b100]);
    }

    #[test]
    fn all_nonempty_submasks_of_zero_yields_nothing() {
        let subs: Vec<RelationSet> = JoinGraph::all_nonempty_submasks(0).collect();
        assert!(subs.is_empty());
    }

    #[test]
    fn neighbors_excludes_the_set_itself_and_finds_both_directions() {
        let edges = vec![
            JoinEdge { left_relation: 0, left_column: 0, right_relation: 1, right_column: 0 },
            JoinEdge { left_relation: 2, left_column: 0, right_relation: 1, right_column: 0 },
        ];
        let graph = JoinGraph { relations: vec![], edges };
        assert_eq!(graph.neighbors(singleton(1)), singleton(0) | singleton(2));
        assert_eq!(graph.neighbors(singleton(0)), singleton(1));
        assert_eq!(graph.neighbors(singleton(0) | singleton(1)), singleton(2));
    }

    #[test]
    fn singleton_and_full_set_are_correct_bitmasks() {
        assert_eq!(singleton(0), 0b1);
        assert_eq!(singleton(3), 0b1000);
        assert_eq!(full_set(3), 0b111);
        assert_eq!(full_set(0), 0);
    }

    #[test]
    fn sub_masks_enumerates_every_nonempty_proper_subset() {
        let s = 0b111u64; // {0,1,2}
        let subs: Vec<RelationSet> = JoinGraph::sub_masks(s).collect();
        // Every nonempty proper subset of {0,1,2}: 6 of them (2^3 - 2).
        assert_eq!(subs.len(), 6);
        assert!(subs.contains(&0b001));
        assert!(subs.contains(&0b110));
        assert!(!subs.contains(&0b111)); // not proper
        assert!(!subs.contains(&0)); // not nonempty
    }

    #[test]
    fn is_connected_finds_an_edge_in_either_direction() {
        let edges = vec![JoinEdge { left_relation: 0, left_column: 0, right_relation: 1, right_column: 0 }];
        let graph = JoinGraph { relations: vec![], edges };
        assert!(graph.is_connected(singleton(0), singleton(1)));
        assert!(graph.is_connected(singleton(1), singleton(0))); // order-independent
        assert!(!graph.is_connected(singleton(0), singleton(2)));
    }

    #[test]
    fn edges_between_normalizes_orientation_to_match_the_query_sides() {
        let edges = vec![JoinEdge { left_relation: 1, left_column: 5, right_relation: 0, right_column: 2 }];
        let graph = JoinGraph { relations: vec![], edges };
        // Query with s1={0}, s2={1}: the edge's actual left is relation 1
        // (in s2), so it must come back normalized with left_relation=0.
        let found = graph.edges_between(singleton(0), singleton(1));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].left_relation, 0);
        assert_eq!(found[0].left_column, 2);
        assert_eq!(found[0].right_relation, 1);
        assert_eq!(found[0].right_column, 5);
    }
}
