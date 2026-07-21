//! Most-common-values lists. See design-docs/basalt-phase3-lld.md §2.5.
//!
//! Kept separately from the histogram, not folded into it: for a column
//! where one value is 60% of rows, a histogram gives a bucket-averaged
//! estimate that's badly wrong, while an MCV list gives the true frequency
//! directly. This is the single highest-leverage improvement to equality
//! selectivity on skewed, categorical data — Postgres does exactly this.

use crate::scalar::ScalarValue;

/// Top-N frequent values, kept separately from the histogram.
#[derive(Clone, Debug)]
pub struct MostCommonValues {
    pub values: Vec<ScalarValue>,
    /// Fraction of rows, parallel to `values`.
    pub frequencies: Vec<f64>,
}

impl MostCommonValues {
    /// The frequency of `v`, if it's tracked in this MCV list.
    pub fn frequency_of(&self, v: &ScalarValue) -> Option<f64> {
        self.values
            .iter()
            .position(|mcv| mcv == v)
            .map(|i| self.frequencies[i])
    }

    /// Estimated equality selectivity for a value *not* in this MCV list:
    /// spread the leftover probability mass — `1 - sum(frequencies)` — over
    /// the remaining distinct values (`ndv - mcv_count`). Getting this right
    /// is what stops the MCV list from making non-MCV equality estimates
    /// worse than not having one at all: without it, a naive `1/ndv`
    /// estimate for a non-MCV value ignores that the MCVs have already
    /// siphoned off a disproportionate share of the rows.
    ///
    /// Returns `None` if every distinct value is already accounted for by
    /// the MCV list (`ndv <= mcv_count`) — there's no "leftover" case to
    /// estimate.
    pub fn non_mcv_selectivity(&self, ndv: usize) -> Option<f64> {
        let mcv_count = self.values.len();
        if ndv <= mcv_count {
            return None;
        }
        let leftover_mass = (1.0 - self.frequencies.iter().sum::<f64>()).max(0.0);
        let leftover_distinct = (ndv - mcv_count) as f64;
        Some(leftover_mass / leftover_distinct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mcv() -> MostCommonValues {
        MostCommonValues {
            values: vec![
                ScalarValue::Utf8(Some("west".to_string())),
                ScalarValue::Utf8(Some("east".to_string())),
            ],
            frequencies: vec![0.6, 0.1],
        }
    }

    #[test]
    fn frequency_of_returns_the_tracked_fraction() {
        let mcv = mcv();
        assert_eq!(
            mcv.frequency_of(&ScalarValue::Utf8(Some("west".to_string()))),
            Some(0.6)
        );
        assert_eq!(
            mcv.frequency_of(&ScalarValue::Utf8(Some("north".to_string()))),
            None
        );
    }

    #[test]
    fn non_mcv_selectivity_spreads_leftover_mass_over_remaining_distinct_values() {
        let mcv = mcv();
        // 1.0 - 0.7 = 0.3 leftover mass over (10 - 2) = 8 remaining values.
        let sel = mcv.non_mcv_selectivity(10).unwrap();
        assert!((sel - 0.3 / 8.0).abs() < 1e-9);
    }

    #[test]
    fn non_mcv_selectivity_is_none_when_mcv_list_covers_every_distinct_value() {
        let mcv = mcv();
        assert_eq!(mcv.non_mcv_selectivity(2), None);
        assert_eq!(mcv.non_mcv_selectivity(1), None);
    }
}
