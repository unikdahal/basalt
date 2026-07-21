//! `Field` and `Schema`. See LLD §2.6.

use std::sync::Arc;

use super::data_type::DataType;
use crate::error::{BasaltError, Result};

/// Every batch in a Phase 2 stream shares one schema — cloning it per batch
/// would be a real cost at `DEFAULT_BATCH_SIZE` granularity. See
/// design-docs/basalt-phase2-lld.md §3.5.
pub type SchemaRef = Arc<Schema>;

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}

impl Field {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Field {
            name: name.into(),
            data_type,
            nullable,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    fields: Vec<Field>,
}

impl Schema {
    /// Errors on duplicate names.
    pub fn new(fields: Vec<Field>) -> Result<Self> {
        use std::collections::HashSet;
        let mut seen = HashSet::with_capacity(fields.len());
        for field in &fields {
            if !seen.insert(&field.name) {
                return Err(BasaltError::Schema {
                    message: format!("duplicate field name '{}'", field.name),
                });
            }
        }
        Ok(Schema { fields })
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn field(&self, index: usize) -> Option<&Field> {
        self.fields.get(index)
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    /// Build a new schema from a subset of columns, in the given order. For projection.
    pub fn project(&self, indices: &[usize]) -> Result<Schema> {
        let fields = indices
            .iter()
            .map(|&i| {
                self.field(i).cloned().ok_or_else(|| BasaltError::Schema {
                    message: format!("field index {i} out of bounds"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Schema::new(fields)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, true),
        ])
        .unwrap()
    }

    #[test]
    fn rejects_duplicate_names() {
        let err = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("a", DataType::Utf8, false),
        ]);
        assert!(err.is_err());
    }

    #[test]
    fn index_of_resolves_names() {
        let s = schema();
        assert_eq!(s.index_of("a"), Some(0));
        assert_eq!(s.index_of("b"), Some(1));
        assert_eq!(s.index_of("nope"), None);
    }

    #[test]
    fn project_reorders_and_subsets() {
        let s = schema();
        let projected = s.project(&[1, 0]).unwrap();
        assert_eq!(projected.field(0).unwrap().name, "b");
        assert_eq!(projected.field(1).unwrap().name, "a");
    }

    #[test]
    fn project_out_of_bounds_errors() {
        let s = schema();
        assert!(s.project(&[5]).is_err());
    }
}
