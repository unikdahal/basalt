# Basalt

A distributed, vectorized SQL query engine written from scratch in Rust. It reads
Apache Iceberg tables from object storage, plans queries with a cost-based optimizer,
executes them in parallel across worker nodes over Arrow Flight, and serves results to
any JDBC/ADBC client via Arrow Flight SQL.

**Status:** Phase 1 (row-at-a-time engine) and Phase 2 (Arrow-native columnar
engine) implemented. See `BENCHMARKS.md` for measured Phase 1 vs Phase 2
numbers.

## Layout

```
src/
├── types/         # DataType, Value, Schema, coercion
├── array/         # Buffer, Bitmap, Array trait, PrimitiveArray/BooleanArray/StringArray
├── buffer/        # Buffer, MutableBuffer, Bitmap, native layout
├── compute/       # arith, comparison, boolean, filter, take, cast, concat, sort kernels
├── batch.rs       # RecordBatch (Phase 1)
├── scalar.rs       # ScalarValue (typed nulls, Phase 2)
├── io/            # CSV ingestion, Parquet read/write
├── expr/          # Phase 1 bound expression tree, typing, row-at-a-time eval
├── physical_expr/ # Phase 2 columnar PhysicalExpr (column, literal, binary, cast, unary, is_null)
├── logical_plan/  # Phase 2 logical plan (closed enum) + builder
├── physical_plan/ # Phase 2 ExecutionPlan: scan, filter, projection, limit, sort/topk,
│                  # aggregate (hash, two-phase), join (hash + nested loop), spill, planner
├── sql/           # lexer, AST, parser
├── plan/          # Phase 1 binder
└── exec/          # Phase 1 DataFrame, operators
```

Dependency direction is strictly downward: `types → array → batch → expr → plan → exec`
(Phase 1) and `types → buffer/array → compute → physical_expr → logical_plan/physical_plan`
(Phase 2). `sql` depends only on `types`.

## Build

```
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all
```

## Benchmarks

```
cargo bench --bench row_vs_batch
```

Real, measured numbers live in `BENCHMARKS.md` — not restated here.

## License

Apache-2.0
