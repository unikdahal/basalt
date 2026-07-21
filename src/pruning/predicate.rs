//! Statistics-based pruning: evaluate a predicate against min/max/null
//! statistics rather than data. See design-docs/basalt-phase3-lld.md §6.4.
//!
//! **The critical asymmetry: only skip when you can *prove* nothing
//! matches.** Uncertainty means scan. A false `CanSkip` silently drops
//! rows — the worst bug class in the engine. A false `MustScan` costs some
//! I/O. Those are not symmetric mistakes, and the implementation is
//! conservative everywhere it's unsure.
//!
//! This machinery gets reused three times (per the LLD): Parquet row-group
//! skipping (Phase 2's own hook, `io::parquet::row_group_may_match`, a
//! narrower special-cased version of the same idea), Iceberg partition
//! pruning and manifest-level file skipping (Phase 4), and here, at plan
//! time for partitioned in-memory sources. Built once, generically, over
//! `ColumnStatistics`.

use crate::expr::expr::Expr;
use crate::scalar::ScalarValue;
use crate::statistics::ColumnStatistics;
use crate::types::coercion::BinaryOp;
use crate::types::value::Value;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PruningResult {
    /// Proven: no row in this partition/file/row-group can match.
    CanSkip,
    /// Some row might match, or we can't tell.
    MustScan,
}

/// Evaluates `predicate` against `stats` (one `ColumnStatistics` per
/// column `predicate` might reference, indexed the same way the predicate
/// indexes columns) rather than data.
///
/// # Errors
/// Errors only if `predicate` references a column index outside `stats`'s
/// range — treated as a genuine bug (a mismatched schema/statistics pair)
/// rather than conservatively returning `MustScan`, since that shouldn't
/// happen for a well-formed plan.
pub fn prune(predicate: &Expr, stats: &[ColumnStatistics]) -> crate::error::Result<PruningResult> {
    Ok(match predicate {
        Expr::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            // Either side proving CanSkip is enough — if no row can
            // satisfy the left conjunct, no row satisfies the AND either.
            match (prune(left, stats)?, prune(right, stats)?) {
                (PruningResult::CanSkip, _) | (_, PruningResult::CanSkip) => PruningResult::CanSkip,
                _ => PruningResult::MustScan,
            }
        }
        Expr::Binary {
            left,
            op: BinaryOp::Or,
            right,
        } => {
            // Both sides must prove CanSkip — a surviving row could
            // satisfy either disjunct.
            match (prune(left, stats)?, prune(right, stats)?) {
                (PruningResult::CanSkip, PruningResult::CanSkip) => PruningResult::CanSkip,
                _ => PruningResult::MustScan,
            }
        }
        Expr::Binary { left, op, right } => prune_comparison(left, *op, right, stats)?,
        // `IS NULL` can only be proven false (CanSkip) with a plain
        // null_count, which this function has; `IS NOT NULL` would need
        // the total row count too (to prove "every row is null"), which
        // isn't part of this function's per-column-statistics interface —
        // conservative (`MustScan`) rather than threading an extra
        // parameter through for one direction of one predicate shape.
        Expr::IsNull(inner) => prune_is_null(inner, stats)?,
        // Anything else (a bare column, a literal, NOT, casts, IS NOT
        // NULL, ...): no structure this function knows how to reason
        // about — always conservative.
        _ => PruningResult::MustScan,
    })
}

fn column_stats_at<'a>(
    expr: &Expr,
    stats: &'a [ColumnStatistics],
) -> crate::error::Result<Option<&'a ColumnStatistics>> {
    match expr {
        Expr::Column { index, .. } => stats.get(*index).map(Some).ok_or_else(|| {
            crate::error::BasaltError::Internal(format!(
                "pruning predicate references column {index}, but only {} column(s) of statistics were given",
                stats.len()
            ))
        }),
        _ => Ok(None),
    }
}

fn prune_is_null(inner: &Expr, stats: &[ColumnStatistics]) -> crate::error::Result<PruningResult> {
    let Some(col) = column_stats_at(inner, stats)? else {
        return Ok(PruningResult::MustScan);
    };
    match col.null_count.get_value() {
        Some(&0) => Ok(PruningResult::CanSkip), // No nulls at all: IS NULL matches nothing here.
        _ => Ok(PruningResult::MustScan),
    }
}

fn prune_comparison(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    stats: &[ColumnStatistics],
) -> crate::error::Result<PruningResult> {
    let (col_expr, literal, op) = match (left, right) {
        (Expr::Column { .. }, Expr::Literal(v)) => (left, v, op),
        (Expr::Literal(v), Expr::Column { .. }) => (right, v, flip(op)),
        _ => return Ok(PruningResult::MustScan),
    };
    let Some(col) = column_stats_at(col_expr, stats)? else {
        return Ok(PruningResult::MustScan);
    };
    let Some(scalar) = value_to_scalar(literal) else {
        return Ok(PruningResult::MustScan); // A NULL literal: comparisons with NULL are never provably true.
    };
    let (Some(min), Some(max)) = (col.min_value.get_value(), col.max_value.get_value()) else {
        return Ok(PruningResult::MustScan);
    };
    let (Some(min), Some(max)) = (as_f64(min), as_f64(max)) else {
        return Ok(PruningResult::MustScan); // Non-numeric type this interval check doesn't handle.
    };
    let Some(value) = as_f64(&scalar) else {
        return Ok(PruningResult::MustScan);
    };

    // Rewrite `col OP value` into a question about `[min, max]`: skip only
    // if the interval provably can't contain a satisfying value.
    let can_skip = match op {
        BinaryOp::Gt => max <= value,
        BinaryOp::GtEq => max < value,
        BinaryOp::Lt => min >= value,
        BinaryOp::LtEq => min > value,
        BinaryOp::Eq => value < min || value > max,
        // NotEq can't be proven false by an interval alone — conservative.
        _ => return Ok(PruningResult::MustScan),
    };
    Ok(if can_skip {
        PruningResult::CanSkip
    } else {
        PruningResult::MustScan
    })
}

fn flip(op: BinaryOp) -> BinaryOp {
    match op {
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::LtEq => BinaryOp::GtEq,
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::GtEq => BinaryOp::LtEq,
        other => other,
    }
}

fn value_to_scalar(v: &Value) -> Option<ScalarValue> {
    Some(match v {
        Value::Int64(i) => ScalarValue::Int64(Some(*i)),
        Value::Float64(f) => ScalarValue::Float64(Some(*f)),
        Value::Utf8(s) => ScalarValue::Utf8(Some(s.clone())),
        Value::Boolean(b) => ScalarValue::Boolean(Some(*b)),
        Value::Null => return None,
    })
}

fn as_f64(v: &ScalarValue) -> Option<f64> {
    match v {
        ScalarValue::Int64(Some(i)) => Some(*i as f64),
        ScalarValue::Float64(Some(f)) => Some(*f),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::Precision;
    use crate::types::data_type::DataType;

    fn col(i: usize) -> Expr {
        Expr::Column {
            index: i,
            data_type: DataType::Int64,
            nullable: false,
        }
    }

    fn lit(v: i64) -> Expr {
        Expr::Literal(Value::Int64(v))
    }

    fn stats_with_range(min: i64, max: i64) -> Vec<ColumnStatistics> {
        let mut c = ColumnStatistics::unknown();
        c.min_value = Precision::Exact(ScalarValue::Int64(Some(min)));
        c.max_value = Precision::Exact(ScalarValue::Int64(Some(max)));
        vec![c]
    }

    #[test]
    fn skips_when_predicate_is_provably_outside_the_range() {
        // amount > 1000, but this partition's max is 500 — provably no match.
        let stats = stats_with_range(0, 500);
        let predicate = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Gt,
            right: Box::new(lit(1000)),
        };
        assert_eq!(prune(&predicate, &stats).unwrap(), PruningResult::CanSkip);
    }

    #[test]
    fn must_scan_when_the_range_could_contain_a_match() {
        let stats = stats_with_range(0, 2000);
        let predicate = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Gt,
            right: Box::new(lit(1000)),
        };
        assert_eq!(prune(&predicate, &stats).unwrap(), PruningResult::MustScan);
    }

    #[test]
    fn must_scan_without_statistics() {
        let stats = vec![ColumnStatistics::unknown()];
        let predicate = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Gt,
            right: Box::new(lit(1000)),
        };
        assert_eq!(prune(&predicate, &stats).unwrap(), PruningResult::MustScan);
    }

    #[test]
    fn equality_outside_range_can_skip() {
        let stats = stats_with_range(0, 100);
        let predicate = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Eq,
            right: Box::new(lit(500)),
        };
        assert_eq!(prune(&predicate, &stats).unwrap(), PruningResult::CanSkip);
    }

    #[test]
    fn and_skips_if_either_conjunct_proves_no_match() {
        let stats = stats_with_range(0, 100);
        let predicate = Expr::Binary {
            left: Box::new(Expr::Binary {
                left: Box::new(col(0)),
                op: BinaryOp::Gt,
                right: Box::new(lit(1000)),
            }),
            op: BinaryOp::And,
            right: Box::new(Expr::Binary {
                left: Box::new(col(0)),
                op: BinaryOp::Gt,
                right: Box::new(lit(-1000)),
            }),
        };
        assert_eq!(prune(&predicate, &stats).unwrap(), PruningResult::CanSkip);
    }

    #[test]
    fn or_requires_both_sides_to_prove_no_match() {
        let stats = stats_with_range(0, 100);
        // First disjunct provably fails; second doesn't — must scan.
        let predicate = Expr::Binary {
            left: Box::new(Expr::Binary {
                left: Box::new(col(0)),
                op: BinaryOp::Gt,
                right: Box::new(lit(1000)),
            }),
            op: BinaryOp::Or,
            right: Box::new(Expr::Binary {
                left: Box::new(col(0)),
                op: BinaryOp::Lt,
                right: Box::new(lit(50)),
            }),
        };
        assert_eq!(prune(&predicate, &stats).unwrap(), PruningResult::MustScan);
    }

    #[test]
    fn out_of_range_column_index_errors_instead_of_guessing() {
        let stats: Vec<ColumnStatistics> = vec![];
        let predicate = Expr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Gt,
            right: Box::new(lit(1000)),
        };
        assert!(prune(&predicate, &stats).is_err());
    }
}
