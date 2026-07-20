# Basalt

A distributed, vectorized SQL query engine written from scratch in Rust. It reads
Apache Iceberg tables from object storage, plans queries with a cost-based optimizer,
executes them in parallel across worker nodes over Arrow Flight, and serves results to
any JDBC/ADBC client via Arrow Flight SQL.

**Status:** project scaffolding — no phase implemented yet.

## Layout

```
src/
├── types/   # DataType, Value, Schema, coercion
├── array/   # Validity, Column, ColumnBuilder
├── batch.rs # RecordBatch
├── io/      # CSV ingestion
├── expr/    # bound expression tree, typing, eval
├── sql/     # lexer, AST, parser
├── plan/    # binder
└── exec/    # DataFrame, operators
```

Dependency direction is strictly downward: `types → array → batch → expr → plan → exec`.
`sql` depends only on `types`.

## Build

```
cargo build
cargo test
```

## License

Apache-2.0
