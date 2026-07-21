use std::sync::Arc;
use std::time::Instant;

use basalt::array::primitive::PrimitiveBuilder;
use basalt::array::types::Int64Type;
use basalt::compute::arith::add;
use basalt::compute::ColumnarValue;
use basalt::scalar::ScalarValue;

const N: usize = 1_000_000;

fn time_it<F: FnMut() -> R, R>(label: &str, iters: u32, mut f: F) {
    for _ in 0..3 {
        std::hint::black_box(f());
    }
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(f());
    }
    let elapsed = start.elapsed();
    println!(
        "{label}: {:.3} ms/iter ({:.2} ns/elem)",
        elapsed.as_secs_f64() * 1000.0 / iters as f64,
        elapsed.as_nanos() as f64 / (iters as f64 * N as f64)
    );
}

fn main() {
    let data: Vec<i64> = (0..N as i64).collect();

    time_it("1_bare_checked_add_vec", 50, || {
        let mut out: Vec<i64> = Vec::with_capacity(N);
        for &v in &data {
            out.push(v.checked_add(1).unwrap());
        }
        out
    });

    time_it("2_bare_wrapping_add_vec", 50, || {
        let mut out: Vec<i64> = Vec::with_capacity(N);
        for &v in &data {
            out.push(v.wrapping_add(1));
        }
        out
    });

    time_it("3_iter_map_collect", 50, || {
        data.iter()
            .map(|&v| v.wrapping_add(1))
            .collect::<Vec<i64>>()
    });

    time_it("4_primitive_builder_checked", 50, || {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(N);
        for &v in &data {
            b.append_value(v.checked_add(1).unwrap());
        }
        b.finish()
    });

    time_it("5_primitive_builder_append_slice", 50, || {
        let computed: Vec<i64> = data.iter().map(|&v| v.wrapping_add(1)).collect();
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(N);
        b.append_slice(&computed);
        b.finish()
    });

    let lhs_array = {
        let mut b = PrimitiveBuilder::<Int64Type>::with_capacity(N);
        for &v in &data {
            b.append_value(v);
        }
        ColumnarValue::Array(Arc::new(b.finish()))
    };
    let rhs = ColumnarValue::Scalar(ScalarValue::Int64(Some(1)));
    time_it("6_full_kernel_add", 50, || add(&lhs_array, &rhs).unwrap());

    // Sustained-load check: criterion's 1M sample ran ~400 iterations over
    // ~5.8s continuously. If short bursts benefit from turbo boost that a
    // multi-second sustained load can't hold, this should show a slower
    // ns/elem than the short 50-iteration run above despite being the exact
    // same kernel call.
    time_it("6b_full_kernel_add_sustained_400iters", 400, || {
        add(&lhs_array, &rhs).unwrap()
    });

    // 3-second warmup, THEN measure — matches criterion's warmup discipline
    // more closely, to rule out a still-ramping clock skewing the number.
    let warmup_start = Instant::now();
    while warmup_start.elapsed().as_secs_f64() < 3.0 {
        std::hint::black_box(add(&lhs_array, &rhs).unwrap());
    }
    time_it("6c_full_kernel_add_after_3s_warmup", 400, || {
        add(&lhs_array, &rhs).unwrap()
    });
}
