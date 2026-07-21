//! The headline Phase 1 vs Phase 2 comparison the LLD calls "the
//! deliverable" of this phase (design-docs/basalt-phase2-lld.md §11):
//! the same arithmetic (`col + 1`), evaluated once per row (Phase 1's
//! `expr::eval::eval`) versus once per batch (Phase 2's `compute::arith::add`),
//! over the same data. Real numbers land in `BENCHMARKS.md`, not restated
//! here — this file only defines what's measured.

use std::sync::Arc;

use basalt::array::builder::ColumnBuilder;
use basalt::array::column::Column;
use basalt::batch::RecordBatch;
use basalt::compute::arith::add as batch_add;
use basalt::compute::ColumnarValue;
use basalt::expr::eval::eval;
use basalt::expr::expr::{BinaryOp, Expr};
use basalt::scalar::ScalarValue;
use basalt::types::data_type::DataType;
use basalt::types::schema::{Field, Schema};
use basalt::types::value::Value;

use basalt::array::primitive::PrimitiveBuilder;
use basalt::array::types::Int64Type;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

fn phase1_batch(n: usize) -> RecordBatch {
    let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap();
    let mut builder = ColumnBuilder::new(DataType::Int64);
    for i in 0..n {
        builder.append_value(Value::Int64(i as i64)).unwrap();
    }
    let column: Column = builder.finish();
    RecordBatch::try_new(schema, vec![column]).unwrap()
}

fn phase2_column(n: usize) -> ColumnarValue {
    let mut builder = PrimitiveBuilder::<Int64Type>::with_capacity(n);
    for i in 0..n {
        builder.append_value(i as i64);
    }
    ColumnarValue::Array(Arc::new(builder.finish()))
}

fn bench_row_vs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("col_plus_one");

    for &n in &[1_000usize, 100_000, 1_000_000] {
        let batch = phase1_batch(n);
        let expr = Expr::Binary {
            left: Box::new(Expr::Column {
                index: 0,
                data_type: DataType::Int64,
                nullable: false,
            }),
            op: BinaryOp::Add,
            right: Box::new(Expr::Literal(Value::Int64(1))),
        };

        group.bench_with_input(BenchmarkId::new("phase1_row_at_a_time", n), &n, |b, _| {
            b.iter(|| {
                for row in 0..batch.num_rows() {
                    black_box(eval(&expr, &batch, row).unwrap());
                }
            });
        });

        let lhs = phase2_column(n);
        let rhs = ColumnarValue::Scalar(ScalarValue::Int64(Some(1)));
        group.bench_with_input(BenchmarkId::new("phase2_batch_at_a_time", n), &n, |b, _| {
            b.iter(|| {
                black_box(batch_add(&lhs, &rhs).unwrap());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_row_vs_batch);
criterion_main!(benches);
