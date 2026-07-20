//! `DataType` — the static type lattice. See LLD §2.1.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Int64,
    Float64,
    Utf8,
    Boolean,
}

impl DataType {
    pub fn is_numeric(&self) -> bool {
        matches!(self, DataType::Int64 | DataType::Float64)
    }

    pub fn name(&self) -> &'static str {
        match self {
            DataType::Int64 => "Int64",
            DataType::Float64 => "Float64",
            DataType::Utf8 => "Utf8",
            DataType::Boolean => "Boolean",
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_numeric() {
        assert!(DataType::Int64.is_numeric());
        assert!(DataType::Float64.is_numeric());
        assert!(!DataType::Utf8.is_numeric());
        assert!(!DataType::Boolean.is_numeric());
    }

    #[test]
    fn name_and_display_match() {
        for dt in [DataType::Int64, DataType::Float64, DataType::Utf8, DataType::Boolean] {
            assert_eq!(dt.name(), dt.to_string());
        }
    }

    #[test]
    fn is_copy() {
        let a = DataType::Int64;
        let b = a;
        assert_eq!(a, b);
    }
}
