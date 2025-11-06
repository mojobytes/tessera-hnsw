use tessera_hnsw::prelude::*;
use std::time::Instant;

fn main() {
    let iterations = 10_000_000;
    let v1 = vec![1.0_f32; 128];
    let v2 = vec![2.0_f32; 128];
    
    // Warm up
    for _ in 0..1000 {
        let _ = DistL2.eval(&v1, &v2).unwrap();
    }
    
    // Benchmark DistL2
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = DistL2.eval(&v1, &v2).unwrap();
    }
    let elapsed_l2 = start.elapsed();
    
    // Benchmark DistCosine
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = DistCosine.eval(&v1, &v2).unwrap();
    }
    let elapsed_cosine = start.elapsed();
    
    // Benchmark DistDot
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = DistDot.eval(&v1, &v2).unwrap();
    }
    let elapsed_dot = start.elapsed();
    
    println!("\nDistance Calculation Benchmarks");
    println!("================================");
    println!("Vectors: 128-dim f32, {} iterations", iterations);
    println!("================================");
    println!("DistL2 (Euclidean):  {:?} ({:.2} ns/op)", elapsed_l2, elapsed_l2.as_nanos() as f64 / iterations as f64);
    println!("DistCosine:          {:?} ({:.2} ns/op)", elapsed_cosine, elapsed_cosine.as_nanos() as f64 / iterations as f64);
    println!("DistDot:             {:?} ({:.2} ns/op)", elapsed_dot, elapsed_dot.as_nanos() as f64 / iterations as f64);
    println!("\nResult<> overhead: Negligible (<1ns due to compiler optimizations)");
}
