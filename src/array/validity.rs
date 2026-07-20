//! `Validity` — per-slot null tracking. See LLD §2.3.

/// Per-slot null tracking for a column.
/// Invariant: `is_valid.len()` == the owning column's logical length.
#[derive(Debug, Clone, PartialEq)]
pub struct Validity {
    is_valid: Vec<bool>,
    null_count: usize,
}

impl Validity {
    pub fn new_all_valid(len: usize) -> Self {
        Validity { is_valid: vec![true; len], null_count: 0 }
    }

    pub fn from_flags(is_valid: Vec<bool>) -> Self {
        let null_count = is_valid.iter().filter(|v| !**v).count();
        Validity { is_valid, null_count }
    }

    pub fn is_valid(&self, index: usize) -> bool {
        self.is_valid[index]
    }

    pub fn is_null(&self, index: usize) -> bool {
        !self.is_valid(index)
    }

    pub fn null_count(&self) -> usize {
        self.null_count
    }

    pub fn len(&self) -> usize {
        self.is_valid.len()
    }

    pub fn is_empty(&self) -> bool {
        self.is_valid.is_empty()
    }

    pub fn set_null(&mut self, index: usize) {
        if self.is_valid[index] {
            self.is_valid[index] = false;
            self.null_count += 1;
        }
    }

    /// Produce a new Validity containing only the given row positions, in order.
    pub fn take(&self, indices: &[usize]) -> Validity {
        let flags: Vec<bool> = indices.iter().map(|&i| self.is_valid[i]).collect();
        Validity::from_flags(flags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_valid_has_no_nulls() {
        let v = Validity::new_all_valid(3);
        assert_eq!(v.null_count(), 0);
        assert!((0..3).all(|i| v.is_valid(i)));
    }

    #[test]
    fn from_flags_counts_nulls() {
        let v = Validity::from_flags(vec![true, false, true, false]);
        assert_eq!(v.null_count(), 2);
        assert!(v.is_null(1));
        assert!(v.is_valid(0));
    }

    #[test]
    fn set_null_updates_count_once() {
        let mut v = Validity::new_all_valid(2);
        v.set_null(0);
        v.set_null(0); // idempotent
        assert_eq!(v.null_count(), 1);
        assert!(v.is_null(0));
        assert!(v.is_valid(1));
    }

    #[test]
    fn take_reorders_and_subsets() {
        let v = Validity::from_flags(vec![true, false, true]);
        let taken = v.take(&[2, 1, 0]);
        assert_eq!(taken.len(), 3);
        assert!(taken.is_valid(0));
        assert!(taken.is_null(1));
        assert!(taken.is_valid(2));
        assert_eq!(taken.null_count(), 1);
    }
}
