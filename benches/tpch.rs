//! End-to-end query benchmark, per design-docs/basalt-phase2-lld.md §11:
//! "the same TPC-H query, Phase 1 vs Phase 2, on the same data." Phase 1's
//! `exec/` has no aggregation at all (no GROUP BY, no SUM/COUNT/AVG), so the
//! fair, honest comparison here is a filter + projection query in the shape
//! of TPC-H Q6 (a multi-predicate `WHERE` over a date range, a discount
//! range, and a quantity bound, projecting `l_extendedprice * l_discount`)
//! with the `SUM` aggregate dropped — everything Phase 1 can actually run.
//!
//! The same `Expr` tree (`crate::expr::expr::Expr` — shared by both engines;
//! `LogicalPlan` is built directly on it, no separate Phase 2 expression
//! type) is built once and fed to both pipelines, so this measures the
//! execution model, not two independently-hand-written predicates that
//! happen to look similar.

use std::sync::Arc;

use basalt::array::builder::ColumnBuilder;
use basalt::array::column::Column;
use basalt::array::primitive::PrimitiveBuilder;
use basalt::array::types::{Float64Type, Int64Type};
use basalt::batch::{ColumnarBatch, RecordBatch};
use basalt::exec::dataframe::DataFrame;
use basalt::expr::expr::{BinaryOp, Expr};
use basalt::logical_plan::builder::LogicalPlanBuilder;
use basalt::physical_plan::planner::PhysicalPlanner;
use basalt::physical_plan::scan::MemoryTableSource;
use basalt::plan::binder::BoundProjection;
use basalt::types::data_type::DataType;
use basalt::types::schema::{Field, Schema};
use basalt::types::value::Value;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

// Column order shared by both engines: l_quantity, l_extendedprice,
// l_discount, l_shipdate.
const COL_QUANTITY: usize = 0;
const COL_EXTENDEDPRICE: usize = 1;
const COL_DISCOUNT: usize = 2;
const COL_SHIPDATE: usize = 3;

// TPC-H Q6-shaped thresholds, tuned against the synthetic generator below
// for a non-trivial (~3-4%) result set at every N.
const SHIPDATE_LO: i64 = 8_400;
const SHIPDATE_HI: i64 = 8_800;
const DISCOUNT_LO: f64 = 0.05;
const DISCOUNT_HI: f64 = 0.07;
const QUANTITY_HI: i64 = 24;

fn phase1_batch(n: usize) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("l_quantity", DataType::Int64, false),
        Field::new("l_extendedprice", DataType::Float64, false),
        Field::new("l_discount", DataType::Float64, false),
        Field::new("l_shipdate", DataType::Int64, false),
    ])
    .unwrap();

    let mut quantity = ColumnBuilder::new(DataType::Int64);
    let mut extendedprice = ColumnBuilder::new(DataType::Float64);
    let mut discount = ColumnBuilder::new(DataType::Float64);
    let mut shipdate = ColumnBuilder::new(DataType::Int64);

    for i in 0..n {
        let i = i as i64;
        quantity.append_value(Value::Int64(i % 50)).unwrap();
        extendedprice
            .append_value(Value::Float64(1000.0 + (i % 10_000) as f64 * 0.37))
            .unwrap();
        discount
            .append_value(Value::Float64((i % 11) as f64 / 100.0))
            .unwrap();
        shipdate
            .append_value(Value::Int64(8_000 + i % 1_500))
            .unwrap();
    }

    let columns: Vec<Column> = vec![
        quantity.finish(),
        extendedprice.finish(),
        discount.finish(),
        shipdate.finish(),
    ];
    RecordBatch::try_new(schema, columns).unwrap()
}

fn phase2_batch(n: usize) -> ColumnarBatch {
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("l_quantity", DataType::Int64, false),
            Field::new("l_extendedprice", DataType::Float64, false),
            Field::new("l_discount", DataType::Float64, false),
            Field::new("l_shipdate", DataType::Int64, false),
        ])
        .unwrap(),
    );

    let mut quantity = PrimitiveBuilder::<Int64Type>::with_capacity(n);
    let mut extendedprice = PrimitiveBuilder::<Float64Type>::with_capacity(n);
    let mut discount = PrimitiveBuilder::<Float64Type>::with_capacity(n);
    let mut shipdate = PrimitiveBuilder::<Int64Type>::with_capacity(n);

    for i in 0..n {
        let i = i as i64;
        quantity.append_value(i % 50);
        extendedprice.append_value(1000.0 + (i % 10_000) as f64 * 0.37);
        discount.append_value((i % 11) as f64 / 100.0);
        shipdate.append_value(8_000 + i % 1_500);
    }

    ColumnarBatch::try_new(
        schema,
        vec![
            Arc::new(quantity.finish()),
            Arc::new(extendedprice.finish()),
            Arc::new(discount.finish()),
            Arc::new(shipdate.finish()),
        ],
    )
    .unwrap()
}

/// `l_shipdate >= SHIPDATE_LO AND l_shipdate < SHIPDATE_HI AND
///  l_discount >= DISCOUNT_LO AND l_discount <= DISCOUNT_HI AND
///  l_quantity < QUANTITY_HI`
fn q6_predicate() -> Expr {
    let col = |index: usize, data_type: DataType| Expr::Column {
        index,
        data_type,
        nullable: false,
    };
    let and = |l: Expr, r: Expr| Expr::Binary {
        left: Box::new(l),
        op: BinaryOp::And,
        right: Box::new(r),
    };
    let cmp = |l: Expr, op: BinaryOp, r: Expr| Expr::Binary {
        left: Box::new(l),
        op,
        right: Box::new(r),
    };

    let shipdate_ge = cmp(
        col(COL_SHIPDATE, DataType::Int64),
        BinaryOp::GtEq,
        Expr::Literal(Value::Int64(SHIPDATE_LO)),
    );
    let shipdate_lt = cmp(
        col(COL_SHIPDATE, DataType::Int64),
        BinaryOp::Lt,
        Expr::Literal(Value::Int64(SHIPDATE_HI)),
    );
    let discount_ge = cmp(
        col(COL_DISCOUNT, DataType::Float64),
        BinaryOp::GtEq,
        Expr::Literal(Value::Float64(DISCOUNT_LO)),
    );
    let discount_le = cmp(
        col(COL_DISCOUNT, DataType::Float64),
        BinaryOp::LtEq,
        Expr::Literal(Value::Float64(DISCOUNT_HI)),
    );
    let quantity_lt = cmp(
        col(COL_QUANTITY, DataType::Int64),
        BinaryOp::Lt,
        Expr::Literal(Value::Int64(QUANTITY_HI)),
    );

    and(
        and(and(shipdate_ge, shipdate_lt), and(discount_ge, discount_le)),
        quantity_lt,
    )
}

/// `l_extendedprice * l_discount AS revenue`
fn revenue_projection() -> Expr {
    Expr::Binary {
        left: Box::new(Expr::Column {
            index: COL_EXTENDEDPRICE,
            data_type: DataType::Float64,
            nullable: false,
        }),
        op: BinaryOp::Mul,
        right: Box::new(Expr::Column {
            index: COL_DISCOUNT,
            data_type: DataType::Float64,
            nullable: false,
        }),
    }
}

fn bench_tpch_q6(c: &mut Criterion) {
    let mut group = c.benchmark_group("tpch_q6_filter_project");

    for &n in &[1_000usize, 100_000, 1_000_000] {
        let batch = phase1_batch(n);
        let predicate = q6_predicate();
        let projection = revenue_projection();

        group.bench_with_input(BenchmarkId::new("phase1_row_at_a_time", n), &n, |b, _| {
            b.iter(|| {
                let df = DataFrame::new(batch.clone());
                let df = df.filter(&predicate).unwrap();
                let df = df
                    .project(&[BoundProjection {
                        expr: projection.clone(),
                        output_name: "revenue".to_string(),
                    }])
                    .unwrap();
                black_box(df.into_batch())
            });
        });

        // Plan construction (scan -> filter -> project -> physical plan) is
        // query planning, not execution — done once here, outside the timed
        // closure, matching Phase 1's loop above where `predicate` and
        // `projection` are also built once and only *evaluated* per
        // iteration. Timing plan-building here would compare Phase 2's
        // planner against Phase 1's bare interpreter loop, not the two
        // execution models against each other.
        let source_batch = phase2_batch(n);
        let schema = source_batch.schema();
        let source = Arc::new(MemoryTableSource::new(
            schema.clone(),
            vec![source_batch.clone()],
        ));
        let logical = LogicalPlanBuilder::scan("lineitem", source)
            .filter(q6_predicate())
            .project(vec![revenue_projection()], vec!["revenue".to_string()])
            .unwrap()
            .build();
        let physical = PhysicalPlanner.create_physical_plan(&logical).unwrap();

        group.bench_with_input(BenchmarkId::new("phase2_batch_at_a_time", n), &n, |b, _| {
            b.iter(|| {
                let batches: Vec<_> = physical
                    .execute(0)
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                black_box(batches)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_tpch_q6);
criterion_main!(benches);
