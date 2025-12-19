//! TDD Tests for HNSW graph-only reload with search functionality
//!
//! These tests verify that search works correctly after loading an HNSW index
//! from disk in graph-only mode (without embedded vectors).
//!
//! ## Bug Context
//!
//! When HNSW is saved in graph-only mode (vectors stored in external .tsvf files),
//! loading creates Points with empty vectors. Search then fails with "EmptyVectors"
//! because the distance calculation requires actual vector data.
//!
//! ## Solution
//!
//! After loading a graph-only index, callers must configure external vector storage
//! via `set_dynamic_vector_storage()` before performing searches.

#![allow(clippy::needless_range_loop)]

use tessera_hnsw::prelude::*;
use tessera_hnsw::distance::DistL2;
use tessera_hnsw::storage::InMemoryVectorStorage;
use rand::distr::{Distribution, Uniform};
use std::sync::Arc;

fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}

/// Generate random test vectors
fn generate_random_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    let unif = Uniform::<f32>::new(0., 1.).unwrap();

    (0..count)
        .map(|_| {
            (0..dimension)
                .map(|_| unif.sample(&mut rng))
                .collect()
        })
        .collect()
}

/// Test 1.1: Confirms the current bug
///
/// After loading a graph-only HNSW index from disk, search FAILS because
/// Points have empty vectors and distance calculation panics.
///
/// This test documents the current broken behavior and should PASS
/// (meaning the bug exists and search fails as expected).
#[test]
fn test_search_after_graph_only_reload_fails() {
    println!("\n\n test_search_after_graph_only_reload_fails");
    log_init_test();

    // 1. Generate test data: 100 vectors of dimension 10
    let num_vectors = 100;
    let dimension = 10;
    let data = generate_random_vectors(num_vectors, dimension);

    // 2. Build HNSW index with real vectors
    let ef_construct = 25;
    let nb_connection = 10;
    let hnsw = Hnsw::<f32, DistL2>::new(
        nb_connection,
        num_vectors,
        16,
        ef_construct,
        DistL2 {},
    );

    for (i, vec) in data.iter().enumerate() {
        hnsw.insert((vec, i));
    }

    // Verify search works BEFORE save/reload
    let query = &data[0];
    let pre_reload_results = hnsw.search(query, 5, 50);
    assert!(!pre_reload_results.is_empty(), "Search should work before reload");
    assert_eq!(pre_reload_results[0].d_id, 0, "First result should be exact match");

    // 3. Save to disk (graph-only mode - no .hnsw.data file)
    let directory = tempfile::tempdir().unwrap();
    let fname = "test_graph_only_reload";
    let _res = hnsw.file_dump(directory.path(), fname);

    // 4. Reload from disk
    let mut reloader = HnswIo::new(directory.path(), fname);
    let hnsw_loaded: Hnsw<f32, DistL2> = reloader.load_hnsw::<f32, DistL2>().unwrap();

    // 5. Attempt search - THIS SHOULD PANIC with "EmptyVectors"
    // We use std::panic::catch_unwind to capture the panic
    let search_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hnsw_loaded.search(query, 5, 50)
    }));

    // The test PASSES if search panics (confirming the bug exists)
    assert!(
        search_result.is_err(),
        "BUG NOT REPRODUCED: Search should have panicked with EmptyVectors after graph-only reload. \
         If this assertion fails, either the bug was fixed or the test setup is wrong."
    );

    println!("Bug confirmed: Search panics after graph-only reload as expected");
}

/// Test 1.2: Defines the expected behavior after fix
///
/// After loading a graph-only HNSW index and configuring external vector storage,
/// search should work correctly and return proper k-NN results.
#[test]
fn test_search_after_graph_only_reload_with_external_storage() {
    println!("\n\n test_search_after_graph_only_reload_with_external_storage");
    log_init_test();

    // 1. Generate test data: 100 vectors of dimension 10
    let num_vectors = 100;
    let dimension = 10;
    let data = generate_random_vectors(num_vectors, dimension);

    // 2. Build HNSW index
    let ef_construct = 25;
    let nb_connection = 10;
    let hnsw = Hnsw::<f32, DistL2>::new(
        nb_connection,
        num_vectors,
        16,
        ef_construct,
        DistL2 {},
    );

    for (i, vec) in data.iter().enumerate() {
        hnsw.insert((vec, i));
    }

    // 3. Save to disk (graph-only mode)
    let directory = tempfile::tempdir().unwrap();
    let fname = "test_with_external_storage";
    let _res = hnsw.file_dump(directory.path(), fname);

    // 4. Reload from disk
    let mut reloader = HnswIo::new(directory.path(), fname);
    let mut hnsw_loaded: Hnsw<f32, DistL2> = reloader.load_hnsw::<f32, DistL2>().unwrap();

    // 5. Configure external vector storage
    let storage: Arc<dyn DynVectorStorage<f32>> = Arc::new(InMemoryVectorStorage::new(data.clone()));
    hnsw_loaded.set_dynamic_vector_storage(storage);

    // 6. Search should now work
    let query = &data[0];
    let results = hnsw_loaded.search(query, 5, 50);

    assert!(!results.is_empty(), "Search should return results");
    assert!(results.len() <= 5, "Should return at most k results");
    assert_eq!(results[0].d_id, 0, "First result should be exact match (id=0)");
    assert!(results[0].distance < 0.001, "Distance to self should be near zero");

    println!("SUCCESS: Search works after graph-only reload with external storage");
}
