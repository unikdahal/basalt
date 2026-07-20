//! CSV reader ingestion and type inference.
//!
//! Loads CSV strings or files into memory. Automatically determines schema
//! column types and nullability using a two-pass lattice resolution:
//! `Boolean` -> `Int64` -> `Float64` -> `Utf8`.

use std::path::Path;
use crate::array::builder::ColumnBuilder;
use crate::batch::RecordBatch;
use crate::error::{BasaltError, Result};
use crate::types::data_type::DataType;
use crate::types::schema::{Field, Schema};
use crate::types::value::Value;

/// Custom options for configuring CSV ingestion.
pub struct CsvReadOptions {
    /// True if the first row is a header containing column names.
    pub has_header: bool,
    /// Delimiter character separating fields (e.g. `b','`).
    pub delimiter: u8,
    /// Literal matching a null value (e.g. `""` or `"NULL"`).
    pub null_literal: String,
    /// Sample count limit to scan for Pass 1 type inference.
    pub infer_rows: Option<usize>,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        Self {
            has_header: true,
            delimiter: b',',
            null_literal: String::new(),
            infer_rows: None,
        }
    }
}

pub struct CsvReader;

impl CsvReader {
    /// Reads a file from disk and parses it into a `RecordBatch`.
    pub fn read_file<P: AsRef<Path>>(path: P, options: &CsvReadOptions) -> Result<RecordBatch> {
        let content = std::fs::read_to_string(path)?;
        Self::read_str(&content, options)
    }

    /// Reads a raw data string and parses it into a `RecordBatch`.
    pub fn read_str(data: &str, options: &CsvReadOptions) -> Result<RecordBatch> {
        let records = parse_csv_records(data, options.delimiter)?;

        if records.is_empty() {
            // An empty CSV file returns an empty batch with an empty schema.
            return Ok(RecordBatch::empty(Schema::new(Vec::new())?));
        }

        let has_header = options.has_header;
        let headers = if has_header {
            records[0].clone()
        } else {
            Vec::new()
        };

        let data_records = if has_header {
            &records[1..]
        } else {
            &records[..]
        };

        let num_cols = if has_header {
            headers.len()
        } else {
            records.first().map_or(0, |r| r.len())
        };

        if num_cols == 0 {
            return Ok(RecordBatch::empty(Schema::new(Vec::new())?));
        }

        // Pass 1: Type Inference over all or a sample of rows
        let mut can_bool = vec![true; num_cols];
        let mut can_int = vec![true; num_cols];
        let mut can_float = vec![true; num_cols];
        let mut has_null = vec![false; num_cols];
        let mut non_null_count = vec![0; num_cols];

        let infer_limit = options.infer_rows.unwrap_or(data_records.len()).min(data_records.len());

        for (r, row) in data_records.iter().take(infer_limit).enumerate() {
            let line_num = r + 1 + if has_header { 1 } else { 0 };
            if row.len() != num_cols {
                return Err(BasaltError::Csv {
                    line: line_num,
                    message: format!(
                        "ragged row in inference: expected {num_cols} fields, found {}",
                        row.len()
                    ),
                });
            }
            for c in 0..num_cols {
                let val = &row[c];
                if val == &options.null_literal || val.is_empty() {
                    has_null[c] = true;
                } else {
                    non_null_count[c] += 1;
                    if can_int[c] && val.parse::<i64>().is_err() {
                        can_int[c] = false;
                    }
                    if can_float[c] && val.parse::<f64>().is_err() {
                        can_float[c] = false;
                    }
                    if can_bool[c] {
                        let val_lower = val.to_ascii_lowercase();
                        if val_lower != "true" && val_lower != "false" {
                            can_bool[c] = false;
                        }
                    }
                }
            }
        }

        // Resolve types via LUpper-bound lattice promotion
        let mut col_types = Vec::with_capacity(num_cols);
        for c in 0..num_cols {
            let t = if non_null_count[c] == 0 {
                DataType::Utf8
            } else if can_bool[c] {
                DataType::Boolean
            } else if can_int[c] {
                DataType::Int64
            } else if can_float[c] {
                DataType::Float64
            } else {
                DataType::Utf8
            };
            col_types.push(t);
        }

        // Build Schema structure
        let mut fields = Vec::with_capacity(num_cols);
        for c in 0..num_cols {
            let name = if has_header {
                headers[c].clone()
            } else {
                format!("col_{c}")
            };
            let nullable = has_null[c] || non_null_count[c] == 0;
            fields.push(Field::new(name, col_types[c], nullable));
        }
        let schema = Schema::new(fields)?;

        // Pass 2: Ingest all values through typed builders
        let mut builders = col_types.iter().map(|&t| ColumnBuilder::new(t)).collect::<Vec<_>>();

        for (r, row) in data_records.iter().enumerate() {
            let line_num = r + 1 + if has_header { 1 } else { 0 };
            if row.len() != num_cols {
                return Err(BasaltError::Csv {
                    line: line_num,
                    message: format!(
                        "ragged row in ingestion: expected {num_cols} fields, found {}",
                        row.len()
                    ),
                });
            }
            for c in 0..num_cols {
                let val = &row[c];
                if val == &options.null_literal || val.is_empty() {
                    builders[c].append_null();
                } else {
                    let parsed_val = match col_types[c] {
                        DataType::Int64 => {
                            let v = val.parse::<i64>().map_err(|_| BasaltError::Csv {
                                line: line_num,
                                message: format!("cannot parse '{val}' as Int64"),
                            })?;
                            Value::Int64(v)
                        }
                        DataType::Float64 => {
                            let v = val.parse::<f64>().map_err(|_| BasaltError::Csv {
                                line: line_num,
                                message: format!("cannot parse '{val}' as Float64"),
                            })?;
                            Value::Float64(v)
                        }
                        DataType::Boolean => {
                            let v = match val.to_ascii_lowercase().as_str() {
                                "true" => true,
                                "false" => false,
                                _ => {
                                    return Err(BasaltError::Csv {
                                        line: line_num,
                                        message: format!("cannot parse '{val}' as Boolean"),
                                    });
                                }
                            };
                            Value::Boolean(v)
                        }
                        DataType::Utf8 => Value::Utf8(val.clone()),
                    };
                    builders[c].append_value(parsed_val)?;
                }
            }
        }

        let columns = builders.into_iter().map(|b| b.finish()).collect::<Vec<_>>();
        RecordBatch::try_new(schema, columns)
    }
}

/// Helper function to parse CSV records manually, handling quoted fields.
fn parse_csv_records(data: &str, delimiter: u8) -> Result<Vec<Vec<String>>> {
    let mut records = Vec::new();
    let mut current_record = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;
    let mut chars = data.char_indices().peekable();

    while let Some((_, c)) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek().map(|&(_, next_c)| next_c) == Some('"') {
                    // Escaped double quote inside double quotes ("" -> ")
                    chars.next();
                    current_field.push('"');
                } else {
                    // End quotes block
                    in_quotes = false;
                }
            } else {
                current_field.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == delimiter as char {
            current_record.push(current_field);
            current_field = String::new();
        } else if c == '\n' {
            // Strip carriage returns if any (\r\n -> \n)
            if current_field.ends_with('\r') {
                current_field.pop();
            }
            current_record.push(current_field);
            records.push(current_record);
            current_record = Vec::new();
            current_field = String::new();
        } else {
            current_field.push(c);
        }
    }

    // Handle last record if file does not end with a newline
    if !current_record.is_empty() || !current_field.is_empty() {
        if current_field.ends_with('\r') {
            current_field.pop();
        }
        current_record.push(current_field);
        records.push(current_record);
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_basic() {
        let csv = "id,name,active,score\n1,Alice,true,92.5\n2,Bob,false,88.0\n3,Charlie,,75.3";
        let batch = CsvReader::read_str(csv, &CsvReadOptions::default()).unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 4);

        let schema = batch.schema();
        assert_eq!(schema.field(0).unwrap().data_type, DataType::Int64);
        assert_eq!(schema.field(1).unwrap().data_type, DataType::Utf8);
        assert_eq!(schema.field(2).unwrap().data_type, DataType::Boolean);
        assert_eq!(schema.field(3).unwrap().data_type, DataType::Float64);

        assert!(schema.field(2).unwrap().nullable); // active has an empty/null cell
        assert!(!schema.field(0).unwrap().nullable);
    }

    #[test]
    fn test_csv_escaped_quotes() {
        let csv = "id,desc\n1,\"Alice \"\"Builder\"\"\"\n2,\"Bob\"";
        let batch = CsvReader::read_str(csv, &CsvReadOptions::default()).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(
            batch.column(1).unwrap().get(0),
            Some(Value::Utf8("Alice \"Builder\"".to_string()))
        );
    }

    #[test]
    fn test_csv_ragged_rows() {
        let csv = "id,val\n1,2\n3,4,5";
        let err = CsvReader::read_str(csv, &CsvReadOptions::default());
        assert!(err.is_err());
        assert!(matches!(err.unwrap_err(), BasaltError::Csv { line: 3, .. }));
    }

    #[test]
    fn test_csv_sampling_inference() {
        let csv = "val\n1\n2\n3.5";
        let mut options = CsvReadOptions::default();
        options.infer_rows = Some(2);

        let err = CsvReader::read_str(csv, &options);
        assert!(err.is_err());
        assert!(matches!(err.unwrap_err(), BasaltError::Csv { line: 4, .. }));
    }

    #[test]
    fn test_csv_empty_header_only() {
        let csv = "id,name,active";
        let batch = CsvReader::read_str(csv, &CsvReadOptions::default()).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 3);
    }
}
