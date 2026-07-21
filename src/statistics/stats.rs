//! `TableStatistics`/`ColumnStatistics`. See design-docs/basalt-phase3-lld.md
//! §2.2.

use super::histogram::Histogram;
use super::mcv::MostCommonValues;
use super::precision::Precision;
use crate::scalar::ScalarValue;

#[derive(Clone, Debug)]
pub struct TableStatistics {
    pub num_rows: Precision<usize>,
    pub total_byte_size: Precision<usize>,
    pub column_statistics: Vec<ColumnStatistics>,
}

impl TableStatistics {
    /// Every field `Absent`. The honest default when nothing is known —
    /// every estimation path must produce a correct (if possibly slow) plan
    /// against this, not just against real statistics.
    pub fn unknown(num_columns: usize) -> Self {
        TableStatistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: (0..num_columns).map(|_| ColumnStatistics::unknown()).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColumnStatistics {
    pub null_count: Precision<usize>,
    pub distinct_count: Precision<usize>,
    pub min_value: Precision<ScalarValue>,
    pub max_value: Precision<ScalarValue>,
    pub histogram: Option<Histogram>,
    pub mcv: Option<MostCommonValues>,
}

impl ColumnStatistics {
    pub fn unknown() -> Self {
        ColumnStatistics {
            null_count: Precision::Absent,
            distinct_count: Precision::Absent,
            min_value: Precision::Absent,
            max_value: Precision::Absent,
            histogram: None,
            mcv: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_table_has_absent_everything() {
        let stats = TableStatistics::unknown(2);
        assert!(stats.num_rows.is_absent());
        assert!(stats.total_byte_size.is_absent());
        assert_eq!(stats.column_statistics.len(), 2);
        for col in &stats.column_statistics {
            assert!(col.null_count.is_absent());
            assert!(col.distinct_count.is_absent());
            assert!(col.min_value.is_absent());
            assert!(col.max_value.is_absent());
            assert!(col.histogram.is_none());
            assert!(col.mcv.is_none());
        }
    }
}
