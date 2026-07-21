//! Join cardinality estimation. See design-docs/basalt-phase3-lld.md §3.3.
//!
//! The standard estimate for an equi-join, from Selinger onward:
//! `|R ⋈ S| = (|R| × |S|) / max(NDV(R.a), NDV(S.b))`. Assumes
//! **containment** (the smaller domain is a subset of the larger — true
//! for foreign keys, which is most joins) and **uniformity** within each
//! domain.

use crate::logical_plan::plan::JoinType;
use crate::statistics::{Precision, TableStatistics};

/// # Panics
/// Panics if `on` is empty — the caller must have at least one join
/// column pair (a genuine cross product has a separate, simpler estimate:
/// `|R| * |S|`, computed directly by callers rather than through this
/// function).
pub fn join_cardinality(
    left: &TableStatistics,
    right: &TableStatistics,
    on: &[(usize, usize)],
    join_type: JoinType,
) -> Precision<usize> {
    assert!(
        !on.is_empty(),
        "join_cardinality requires at least one join column pair"
    );

    let (Some(&left_rows), Some(&right_rows)) =
        (left.num_rows.get_value(), right.num_rows.get_value())
    else {
        return Precision::Absent;
    };

    // Multi-column joins: the combined selectivity is the product of each
    // column pair's max-NDV denominator — independence across columns,
    // the same simplifying assumption single-column estimation makes
    // within one column. A combined multi-column NDV (if collected) would
    // be a strictly better refinement; not implemented here (see the LLD's
    // own "refinements worth implementing, in order of payoff" — this is
    // listed above histogram-based join estimation precisely because it's
    // the cheaper win).
    let mut denom = 1.0f64;
    let mut any_absent = false;
    for &(l_col, r_col) in on {
        let (Some(l_stats), Some(r_stats)) = (
            left.column_statistics.get(l_col),
            right.column_statistics.get(r_col),
        ) else {
            any_absent = true;
            continue;
        };
        match (
            l_stats.distinct_count.get_value(),
            r_stats.distinct_count.get_value(),
        ) {
            (Some(&l_ndv), Some(&r_ndv)) if l_ndv > 0 && r_ndv > 0 => {
                denom *= l_ndv.max(r_ndv) as f64;
            }
            _ => any_absent = true,
        }
    }

    if any_absent {
        return Precision::Absent;
    }

    let inner_estimate = (left_rows as f64 * right_rows as f64 / denom).round() as usize;
    let clamped_inner = inner_estimate.clamp(1, left_rows.saturating_mul(right_rows).max(1));

    let estimate = match join_type {
        JoinType::Inner => clamped_inner,
        // Outer joins: inner estimate, plus the rows on the preserved
        // side that don't match. Floored at the preserved side's row
        // count — the naive inner-join formula can (and often does)
        // produce a value *below* |R|, which is nonsense for a LEFT JOIN:
        // every left row appears at least once, matched or null-padded.
        JoinType::Left => clamped_inner.max(left_rows),
        JoinType::Right => clamped_inner.max(right_rows),
        JoinType::Full => clamped_inner.max(left_rows).max(right_rows),
        // Semi/anti: bounded by the preserved side's row count (semi can
        // never exceed it; anti is its complement).
        JoinType::LeftSemi | JoinType::LeftAnti => clamped_inner.min(left_rows),
        JoinType::RightSemi | JoinType::RightAnti => clamped_inner.min(right_rows),
    };

    let max_possible = match join_type {
        JoinType::LeftSemi | JoinType::LeftAnti => left_rows,
        JoinType::RightSemi | JoinType::RightAnti => right_rows,
        _ => left_rows.saturating_mul(right_rows).max(1),
    };
    Precision::Inexact(estimate.clamp(1, max_possible.max(1)))
}

/// A genuine cross product (no join columns at all): `|R| * |S|`, with no
/// NDV involved since there's no join predicate to apply selectivity to.
pub fn cross_product_cardinality(
    left: &TableStatistics,
    right: &TableStatistics,
) -> Precision<usize> {
    left.num_rows.multiply(&right.num_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(num_rows: usize, ndv: usize) -> TableStatistics {
        let mut s = TableStatistics::unknown(1);
        s.num_rows = Precision::Exact(num_rows);
        s.column_statistics[0].distinct_count = Precision::Exact(ndv);
        s
    }

    #[test]
    fn inner_join_matches_the_selinger_formula() {
        let left = stats(1000, 100);
        let right = stats(500, 50);
        // (1000 * 500) / max(100, 50) = 5000
        let card = join_cardinality(&left, &right, &[(0, 0)], JoinType::Inner);
        assert_eq!(card.get_value(), Some(&5000));
    }

    #[test]
    fn left_join_is_floored_at_left_row_count() {
        // A tiny right-side NDV mismatch that would otherwise produce an
        // inner estimate below |left| must still floor at |left| — every
        // left row appears at least once in a LEFT JOIN.
        let left = stats(1000, 1);
        let right = stats(2, 1000);
        let card = join_cardinality(&left, &right, &[(0, 0)], JoinType::Left);
        let value = *card.get_value().unwrap();
        assert!(
            value >= 1000,
            "LEFT JOIN cardinality {value} must be >= left row count 1000"
        );
    }

    #[test]
    fn semi_join_never_exceeds_the_preserved_side() {
        let left = stats(100, 10);
        let right = stats(100_000, 5);
        let card = join_cardinality(&left, &right, &[(0, 0)], JoinType::LeftSemi);
        let value = *card.get_value().unwrap();
        assert!(
            value <= 100,
            "LEFT SEMI cardinality {value} must not exceed left row count 100"
        );
    }

    #[test]
    fn missing_statistics_yields_absent_not_a_guess() {
        let left = TableStatistics::unknown(1);
        let right = stats(500, 50);
        let card = join_cardinality(&left, &right, &[(0, 0)], JoinType::Inner);
        assert!(card.is_absent());
    }

    #[test]
    fn cross_product_multiplies_row_counts() {
        let left = stats(10, 1);
        let right = stats(20, 1);
        assert_eq!(
            cross_product_cardinality(&left, &right).get_value(),
            Some(&200)
        );
    }

    #[test]
    #[should_panic(expected = "at least one join column pair")]
    fn empty_on_list_panics() {
        let left = stats(10, 1);
        let right = stats(20, 1);
        let _ = join_cardinality(&left, &right, &[], JoinType::Inner);
    }
}
