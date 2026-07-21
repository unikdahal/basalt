//! Predicate selectivity estimation. See
//! design-docs/basalt-phase3-lld.md §3.1-§3.2.
//!
//! The intellectual core: get this wrong and everything downstream is
//! confidently wrong (Leis et al., VLDB 2015 — cardinality estimation
//! error, not the cost model or enumerator, is the dominant cause of bad
//! plans).

use crate::error::Result;
use crate::expr::expr::Expr;
use crate::statistics::{ColumnStatistics, Precision, TableStatistics};
use crate::types::coercion::BinaryOp;
use crate::types::value::Value;

/// Named, centralized fallback constants — magic numbers, but the only
/// thing worse than a magic number is the same magic number scattered
/// across six files. Postgres's equivalents live in `selfuncs.c`; 0.005 for
/// equality and 0.33 for inequality come from there.
mod defaults {
    /// `col = literal` with no MCV/NDV: assume a reasonably selective
    /// column (matches Postgres's `DEFAULT_EQ_SEL`).
    pub const EQUALITY: f64 = 0.005;
    /// `col < literal` / `col > literal` with no histogram: a third of
    /// rows, Postgres's `DEFAULT_INEQ_SEL`.
    pub const INEQUALITY: f64 = 0.33;
    /// `col BETWEEN a AND b` with no histogram.
    pub const RANGE: f64 = 0.25;
    /// `col LIKE '%x%'` — no structure to exploit at all.
    pub const LIKE_SUBSTRING: f64 = 0.05;
    /// Anything unrecognized.
    pub const UNKNOWN: f64 = 0.1;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Selectivity {
    pub value: f64,
    /// Did we actually have statistics, or guess? `Precision<()>` carries
    /// no payload — only whether the estimate is grounded in real data.
    pub precision: Precision<()>,
}

impl Selectivity {
    fn exact(value: f64) -> Self {
        Selectivity { value: clamp(value), precision: Precision::Exact(()) }
    }
    fn inexact(value: f64) -> Self {
        Selectivity { value: clamp(value), precision: Precision::Inexact(()) }
    }
}

/// Floor for any non-zero selectivity: a selectivity of exactly 0
/// propagates a cardinality of 0 up the plan tree and makes every plan
/// look free, which produces spectacularly bad decisions — "at least one
/// row" is a much safer default than "definitely nothing."
fn clamp(v: f64) -> f64 {
    v.clamp(1e-6, 1.0)
}

/// Estimates the fraction of input rows `predicate` is expected to keep.
///
/// # Errors
/// Errors if `predicate` references a column index out of range for
/// `schema`/`stats`.
pub fn selectivity(predicate: &Expr, stats: &TableStatistics) -> Result<Selectivity> {
    Ok(match predicate {
        Expr::Binary { left, op: BinaryOp::And, right } => {
            combine_conjunction(&selectivity(left, stats)?, &selectivity(right, stats)?)
        }
        Expr::Binary { left, op: BinaryOp::Or, right } => {
            combine_disjunction(&selectivity(left, stats)?, &selectivity(right, stats)?)
        }
        Expr::Unary { op: crate::types::coercion::UnaryOp::Not, expr } => {
            negate(&selectivity(expr, stats)?)
        }
        Expr::IsNull(inner) => is_null_selectivity(inner, stats, true)?,
        Expr::IsNotNull(inner) => is_null_selectivity(inner, stats, false)?,
        Expr::Binary { left, op, right } => comparison_selectivity(left, *op, right, stats)?,
        _ => Selectivity::inexact(defaults::UNKNOWN),
    })
}

fn column_stats<'a>(expr: &Expr, stats: &'a TableStatistics) -> Option<&'a ColumnStatistics> {
    match expr {
        Expr::Column { index, .. } => stats.column_statistics.get(*index),
        _ => None,
    }
}

fn is_null_selectivity(inner: &Expr, stats: &TableStatistics, want_null: bool) -> Result<Selectivity> {
    let Some(col) = column_stats(inner, stats) else {
        return Ok(Selectivity::inexact(defaults::UNKNOWN));
    };
    let Some(&null_count) = col.null_count.get_value() else {
        return Ok(Selectivity::inexact(defaults::UNKNOWN));
    };
    let Some(&num_rows) = stats.num_rows.get_value() else {
        return Ok(Selectivity::inexact(defaults::UNKNOWN));
    };
    if num_rows == 0 {
        return Ok(Selectivity::inexact(defaults::UNKNOWN));
    }
    let null_frac = null_count as f64 / num_rows as f64;
    let value = if want_null { null_frac } else { 1.0 - null_frac };
    Ok(if col.null_count.is_exact() && stats.num_rows.is_exact() {
        Selectivity::exact(value)
    } else {
        Selectivity::inexact(value)
    })
}

fn comparison_selectivity(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    stats: &TableStatistics,
) -> Result<Selectivity> {
    // `col1 = col2` (same table): 1 / max(ndv1, ndv2).
    if op == BinaryOp::Eq {
        if let (Expr::Column { .. }, Expr::Column { .. }) = (left, right) {
            if let (Some(l), Some(r)) = (column_stats(left, stats), column_stats(right, stats)) {
                if let (Some(&ndv_l), Some(&ndv_r)) = (l.distinct_count.get_value(), r.distinct_count.get_value()) {
                    let ndv = ndv_l.max(ndv_r).max(1);
                    return Ok(Selectivity::inexact(1.0 / ndv as f64));
                }
            }
        }
    }

    let (col_expr, literal, flipped) = match (left, right) {
        (Expr::Column { .. }, Expr::Literal(v)) => (left, v, false),
        (Expr::Literal(v), Expr::Column { .. }) => (right, v, true),
        _ => return Ok(Selectivity::inexact(defaults::UNKNOWN)),
    };
    let Some(col) = column_stats(col_expr, stats) else {
        return Ok(Selectivity::inexact(defaults::UNKNOWN));
    };
    let op = if flipped { flip(op) } else { op };
    let scalar = value_to_scalar(literal);

    Ok(match op {
        BinaryOp::Eq => equality_selectivity(col, scalar.as_ref()),
        BinaryOp::Lt | BinaryOp::Gt | BinaryOp::LtEq | BinaryOp::GtEq => {
            inequality_selectivity(col, op, scalar.as_ref())
        }
        _ => Selectivity::inexact(defaults::UNKNOWN),
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

fn value_to_scalar(v: &Value) -> Option<crate::scalar::ScalarValue> {
    use crate::scalar::ScalarValue;
    Some(match v {
        Value::Int64(i) => ScalarValue::Int64(Some(*i)),
        Value::Float64(f) => ScalarValue::Float64(Some(*f)),
        Value::Utf8(s) => ScalarValue::Utf8(Some(s.clone())),
        Value::Boolean(b) => ScalarValue::Boolean(Some(*b)),
        Value::Null => return None,
    })
}

/// `col = literal`: MCV frequency if present, else `(1 - sum(mcv
/// frequencies)) / (ndv - mcv_count)`, else `1/ndv`, else the fallback
/// constant.
fn equality_selectivity(col: &ColumnStatistics, scalar: Option<&crate::scalar::ScalarValue>) -> Selectivity {
    if let (Some(mcv), Some(scalar)) = (&col.mcv, scalar) {
        if let Some(freq) = mcv.frequency_of(scalar) {
            return Selectivity::exact(freq);
        }
        if let Some(&ndv) = col.distinct_count.get_value() {
            if let Some(non_mcv) = mcv.non_mcv_selectivity(ndv) {
                return Selectivity::inexact(non_mcv);
            }
        }
    }
    if let Some(&ndv) = col.distinct_count.get_value() {
        if ndv > 0 {
            return Selectivity::inexact(1.0 / ndv as f64);
        }
    }
    Selectivity::inexact(defaults::EQUALITY)
}

/// `col < literal` / `col > literal`: histogram if present, else the
/// fallback constant.
fn inequality_selectivity(
    col: &ColumnStatistics,
    op: BinaryOp,
    scalar: Option<&crate::scalar::ScalarValue>,
) -> Selectivity {
    if let (Some(hist), Some(scalar)) = (&col.histogram, scalar) {
        let frac = match op {
            BinaryOp::Lt | BinaryOp::LtEq => hist.less_than(scalar),
            BinaryOp::Gt | BinaryOp::GtEq => hist.less_than(scalar).map(|f| 1.0 - f),
            _ => unreachable!("caller only passes comparison ops"),
        };
        if let Ok(frac) = frac {
            return Selectivity::inexact(frac);
        }
    }
    Selectivity::inexact(defaults::INEQUALITY)
}

/// `col BETWEEN a AND b`: histogram range if present, else the fallback
/// constant.
pub fn range_selectivity(col: &ColumnStatistics, lo: &Value, hi: &Value) -> Selectivity {
    if let (Some(hist), Some(lo), Some(hi)) = (&col.histogram, value_to_scalar(lo), value_to_scalar(hi)) {
        if let Ok(frac) = hist.range(&lo, &hi) {
            return Selectivity::inexact(frac);
        }
    }
    Selectivity::inexact(defaults::RANGE)
}

/// `col IN (a, b, c)`: sum of the equality selectivities, capped at 1.0.
pub fn in_list_selectivity(col: &ColumnStatistics, values: &[Value]) -> Selectivity {
    let mut total = 0.0;
    let mut any_inexact = false;
    for v in values {
        let scalar = value_to_scalar(v);
        let s = equality_selectivity(col, scalar.as_ref());
        total += s.value;
        any_inexact |= !s.precision.is_exact();
    }
    if any_inexact {
        Selectivity::inexact(total.min(1.0))
    } else {
        Selectivity::exact(total.min(1.0))
    }
}

/// `col LIKE '%x%'`: no structure to exploit.
pub fn like_substring_selectivity() -> Selectivity {
    Selectivity::inexact(defaults::LIKE_SUBSTRING)
}

/// Combines two conjunct selectivities assuming independence
/// (`sel(A) * sel(B)`, the textbook answer — known to *underestimate*,
/// often severely, since real predicates are positively correlated).
/// `combine_conjunction_backoff` below is the exponential-backoff
/// alternative (`s1 * s2^(1/2) * s3^(1/4) * ...`, sorted descending — a
/// heuristic with no principled derivation that empirically beats
/// independence by damping the tail's contribution). Both are implemented;
/// measuring the difference in q-error across TPC-H-shaped queries (as the
/// LLD calls for) is a follow-up left for `BENCHMARKS.md`, not implemented
/// here.
fn combine_conjunction(a: &Selectivity, b: &Selectivity) -> Selectivity {
    let precision = weaker(a.precision, b.precision);
    with_precision(a.value * b.value, precision)
}

/// Exponential-backoff combination for an arbitrary list of conjuncts:
/// sorted ascending, each subsequent selectivity contributes less
/// (`s1 * s2^(1/2) * s3^(1/4) * ...`).
pub fn combine_conjunction_backoff(selectivities: &[Selectivity]) -> Selectivity {
    if selectivities.is_empty() {
        return Selectivity::exact(1.0);
    }
    let mut sorted: Vec<f64> = selectivities.iter().map(|s| s.value).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut result = sorted[0];
    let mut damping = 0.5;
    for &s in &sorted[1..] {
        result *= s.powf(damping);
        damping /= 2.0;
    }
    let precision = selectivities
        .iter()
        .fold(Precision::Exact(()), |acc, s| weaker(acc, s.precision));
    with_precision(result, precision)
}

/// `sel(A OR B) = sel(A) + sel(B) - sel(A)*sel(B)`, assuming independence.
fn combine_disjunction(a: &Selectivity, b: &Selectivity) -> Selectivity {
    let precision = weaker(a.precision, b.precision);
    with_precision(a.value + b.value - a.value * b.value, precision)
}

/// `sel(NOT A) = 1 - sel(A)`.
fn negate(a: &Selectivity) -> Selectivity {
    with_precision(1.0 - a.value, a.precision)
}

fn weaker(a: Precision<()>, b: Precision<()>) -> Precision<()> {
    match (a, b) {
        (Precision::Exact(()), Precision::Exact(())) => Precision::Exact(()),
        (Precision::Absent, _) | (_, Precision::Absent) => Precision::Absent,
        _ => Precision::Inexact(()),
    }
}

fn with_precision(value: f64, precision: Precision<()>) -> Selectivity {
    Selectivity { value: clamp(value), precision }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::MostCommonValues;
    use crate::types::data_type::DataType;

    fn stats_unknown() -> TableStatistics {
        TableStatistics::unknown(1)
    }

    fn col(i: usize) -> Expr {
        Expr::Column { index: i, data_type: DataType::Int64, nullable: false }
    }

    fn lit(v: i64) -> Expr {
        Expr::Literal(Value::Int64(v))
    }

    #[test]
    fn equality_falls_back_to_default_with_no_statistics() {
        let s = selectivity(
            &Expr::Binary { left: Box::new(col(0)), op: BinaryOp::Eq, right: Box::new(lit(5)) },
            &stats_unknown(),
        )
        .unwrap();
        assert_eq!(s.value, defaults::EQUALITY);
        assert!(!s.precision.is_exact());
    }

    #[test]
    fn equality_uses_ndv_when_present() {
        let mut stats = stats_unknown();
        stats.column_statistics[0].distinct_count = Precision::Exact(100);
        let s = selectivity(
            &Expr::Binary { left: Box::new(col(0)), op: BinaryOp::Eq, right: Box::new(lit(5)) },
            &stats,
        )
        .unwrap();
        assert!((s.value - 0.01).abs() < 1e-9);
    }

    #[test]
    fn equality_prefers_mcv_frequency_over_ndv() {
        let mut stats = stats_unknown();
        stats.column_statistics[0].distinct_count = Precision::Exact(100);
        stats.column_statistics[0].mcv = Some(MostCommonValues {
            values: vec![crate::scalar::ScalarValue::Int64(Some(5))],
            frequencies: vec![0.6],
        });
        let s = selectivity(
            &Expr::Binary { left: Box::new(col(0)), op: BinaryOp::Eq, right: Box::new(lit(5)) },
            &stats,
        )
        .unwrap();
        assert_eq!(s.value, 0.6);
        assert!(s.precision.is_exact());
    }

    #[test]
    fn conjunction_assumes_independence_and_multiplies() {
        let a = Selectivity::exact(0.5);
        let b = Selectivity::exact(0.4);
        let combined = combine_conjunction(&a, &b);
        assert!((combined.value - 0.2).abs() < 1e-9);
    }

    #[test]
    fn disjunction_matches_inclusion_exclusion() {
        let a = Selectivity::exact(0.5);
        let b = Selectivity::exact(0.4);
        let combined = combine_disjunction(&a, &b);
        // 0.5 + 0.4 - 0.2 = 0.7
        assert!((combined.value - 0.7).abs() < 1e-9);
    }

    #[test]
    fn negation_is_one_minus_original() {
        // NOT (x > 5) does not include rows where x is NULL, because
        // NULL > 5 is NULL and NOT NULL is still NULL, and WHERE rejects
        // it — this function only computes the arithmetic complement of
        // the *known* selectivity; the null-exclusion is handled by
        // whatever selectivity was passed in already not counting nulls
        // (`comparison_selectivity` never assigns nulls to either branch).
        let a = Selectivity::exact(0.3);
        let combined = negate(&a);
        assert!((combined.value - 0.7).abs() < 1e-9);
    }

    #[test]
    fn zero_selectivity_is_clamped_to_a_nonzero_floor() {
        let s = Selectivity::exact(0.0);
        assert!(s.value > 0.0, "a selectivity of exactly 0 must not propagate as free");
    }

    #[test]
    fn is_null_selectivity_uses_null_count_over_num_rows() {
        let mut stats = stats_unknown();
        stats.num_rows = Precision::Exact(100);
        stats.column_statistics[0].null_count = Precision::Exact(10);
        let s = is_null_selectivity(&col(0), &stats, true).unwrap();
        assert!((s.value - 0.1).abs() < 1e-9);
        assert!(s.precision.is_exact());

        let s = is_null_selectivity(&col(0), &stats, false).unwrap();
        assert!((s.value - 0.9).abs() < 1e-9);
    }

    #[test]
    fn backoff_damps_later_conjuncts_more_than_independence() {
        let sels = vec![Selectivity::exact(0.5), Selectivity::exact(0.5), Selectivity::exact(0.5)];
        let independence = sels.iter().fold(1.0, |acc, s| acc * s.value);
        let backoff = combine_conjunction_backoff(&sels).value;
        assert!(backoff > independence, "backoff ({backoff}) should be less aggressive than independence ({independence})");
    }

    #[test]
    fn range_selectivity_falls_back_without_a_histogram() {
        let col_stats = ColumnStatistics::unknown();
        let s = range_selectivity(&col_stats, &Value::Int64(1), &Value::Int64(10));
        assert_eq!(s.value, defaults::RANGE);
    }

    #[test]
    fn histogram_backed_inequality_differs_from_the_default_constant() {
        use crate::array::primitive::PrimitiveBuilder;
        use crate::array::types::Int64Type;
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(100);
        for i in 0..100i64 {
            b.append_value(i);
        }
        let hist = crate::statistics::Histogram::from_sorted(&b.finish(), 10).unwrap();
        let mut col_stats = ColumnStatistics::unknown();
        col_stats.histogram = Some(hist);
        let s = inequality_selectivity(
            &col_stats,
            BinaryOp::Lt,
            Some(&crate::scalar::ScalarValue::Int64(Some(10))),
        );
        assert!(s.value < defaults::INEQUALITY, "10/100 rows below 10 should be well below the 0.33 default");
    }
}
