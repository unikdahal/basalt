//! `Precision<T>` — track confidence, not just presence. See
//! design-docs/basalt-phase3-lld.md §2.1.
//!
//! Three states rather than `Option<T>`, because `Option` conflates "I don't
//! know" with "I know, roughly." A row count read from a Parquet footer is
//! `Exact`; after a filter, it's `Inexact`. That distinction should drive
//! real optimizer behavior (e.g. preferring a more robust plan when
//! confidence is low), which a plain `Option<T>` can't express. Precision
//! degrades monotonically up the plan tree: combining two statistics is
//! `Exact` only if both inputs were, and any operation on an `Inexact` value
//! produces `Inexact` — never accidentally regains confidence.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision<T> {
    /// Known exactly (e.g. row count from a Parquet footer).
    Exact(T),
    /// Derived through an estimate somewhere in the chain.
    Inexact(T),
    /// No information at all.
    Absent,
}

impl<T: Clone> Precision<T> {
    /// The value, whatever its confidence — `None` only for `Absent`.
    pub fn get_value(&self) -> Option<&T> {
        match self {
            Precision::Exact(v) | Precision::Inexact(v) => Some(v),
            Precision::Absent => None,
        }
    }

    /// Degrades `Exact` to `Inexact`; `Inexact`/`Absent` are unchanged.
    /// Precision only ever moves in this direction once data starts moving
    /// through operators the estimator can't verify exactly.
    #[must_use]
    pub fn to_inexact(self) -> Self {
        match self {
            Precision::Exact(v) => Precision::Inexact(v),
            other => other,
        }
    }

    #[must_use]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Precision<U> {
        match self {
            Precision::Exact(v) => Precision::Exact(f(v)),
            Precision::Inexact(v) => Precision::Inexact(f(v)),
            Precision::Absent => Precision::Absent,
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(self, Precision::Exact(_))
    }

    pub fn is_absent(&self) -> bool {
        matches!(self, Precision::Absent)
    }
}

impl Precision<usize> {
    /// `Exact + Exact = Exact`. Anything else degrades — combining an
    /// `Absent` operand with anything yields `Absent` (no basis to add), and
    /// mixing `Exact`/`Inexact` yields `Inexact` (the sum inherits the
    /// weaker input's confidence).
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (Precision::Exact(a), Precision::Exact(b)) => Precision::Exact(a + b),
            (Precision::Absent, _) | (_, Precision::Absent) => Precision::Absent,
            (a, b) => {
                let (Some(a), Some(b)) = (a.get_value(), b.get_value()) else {
                    return Precision::Absent;
                };
                Precision::Inexact(a + b)
            }
        }
    }

    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        match (self, other) {
            (Precision::Exact(a), Precision::Exact(b)) => Precision::Exact(a * b),
            (Precision::Absent, _) | (_, Precision::Absent) => Precision::Absent,
            (a, b) => {
                let (Some(a), Some(b)) = (a.get_value(), b.get_value()) else {
                    return Precision::Absent;
                };
                Precision::Inexact(a * b)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_value_is_none_only_for_absent() {
        assert_eq!(Precision::Exact(5).get_value(), Some(&5));
        assert_eq!(Precision::Inexact(5).get_value(), Some(&5));
        assert_eq!(Precision::<i64>::Absent.get_value(), None);
    }

    #[test]
    fn to_inexact_degrades_exact_only() {
        assert_eq!(Precision::Exact(5).to_inexact(), Precision::Inexact(5));
        assert_eq!(Precision::Inexact(5).to_inexact(), Precision::Inexact(5));
        assert_eq!(Precision::<i64>::Absent.to_inexact(), Precision::Absent);
    }

    #[test]
    fn add_is_exact_only_if_both_are() {
        assert_eq!(
            Precision::Exact(2usize).add(&Precision::Exact(3)),
            Precision::Exact(5)
        );
        assert_eq!(
            Precision::Exact(2usize).add(&Precision::Inexact(3)),
            Precision::Inexact(5)
        );
        assert_eq!(
            Precision::Exact(2usize).add(&Precision::Absent),
            Precision::Absent
        );
    }

    #[test]
    fn multiply_is_exact_only_if_both_are() {
        assert_eq!(
            Precision::Exact(4usize).multiply(&Precision::Exact(5)),
            Precision::Exact(20)
        );
        assert_eq!(
            Precision::Inexact(4usize).multiply(&Precision::Exact(5)),
            Precision::Inexact(20)
        );
    }

    #[test]
    fn map_preserves_precision_kind() {
        assert_eq!(Precision::Exact(2).map(|v| v * 10), Precision::Exact(20));
        assert_eq!(
            Precision::Inexact(2).map(|v| v * 10),
            Precision::Inexact(20)
        );
        assert_eq!(Precision::<i64>::Absent.map(|v| v * 10), Precision::Absent);
    }
}
