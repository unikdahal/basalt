use std::sync::Arc;

use basalt::array::primitive::PrimitiveBuilder;
use basalt::array::types::Int64Type;
use basalt::compute::arith::add;
use basalt::compute::ColumnarValue;
use basalt::scalar::ScalarValue;

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(N);
    for i in 0..N as i64 {
        b.append_value(i);
    }
    let lhs = ColumnarValue::Array(Arc::new(b.finish()));
    let rhs = ColumnarValue::Scalar(ScalarValue::Int64(Some(1)));

    c.bench_function("kernel_only_1m", |bch| {
        bch.iter(|| black_box(add(&lhs, &rhs).unwrap()));
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
