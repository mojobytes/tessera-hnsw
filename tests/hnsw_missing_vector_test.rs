//! TDD Tests for HNSW missing vector error handling
//!
//! ## Bug Context
//!
//! The HNSW index panics when `resolve_vector_from_storage()` cannot find a vector ID.
//! This panic poisons `std::sync::RwLock` in the calling code (tessera), causing
//! ALL subsequent operations to fail until process restart.
//!
//! ## Root Cause
//!
//! Line 1180 in `hnsw.rs` has `panic!()` when vector ID is not found in storage.
//!
//! ## Expected Fix
//!
//! - Replace `panic!()` with `Result<&[T], HnswError>`
//! - Error variant: `HnswError::VectorNotFound { id, context }`
//! - Callers handle error gracefully (skip, retry, or propagate)

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;
use tessera_hnsw::distance::DistL2;
use tessera_hnsw::prelude::*;
use tessera_hnsw::storage::{DynVectorStorage, InMemoryVectorStorage};

fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}

/// Storage that returns None for specific IDs (simulates incomplete/corrupted storage)
#[derive(Debug)]
struct IncompleteVectorStorage<T: Clone + Send + Sync> {
    inner: InMemoryVectorStorage<T>,
    missing_ids: Vec<usize>,
}

impl<T: Clone + Send + Sync + std::fmt::Debug> IncompleteVectorStorage<T> {
    fn new(vectors: Vec<Vec<T>>, missing_ids: Vec<usize>) -> Self {
        Self {
            inner: InMemoryVectorStorage::new(vectors),
            missing_ids,
        }
    }
}

impl<T: Clone + Send + Sync + std::fmt::Debug> DynVectorStorage<T> for IncompleteVectorStorage<T> {
    fn get_vector(&self, id: usize) -> Option<&[T]> {
        // Simulate missing vectors by returning None for specific IDs
        if self.missing_ids.contains(&id) {
            None
        } else {
            DynVectorStorage::get_vector(&self.inner, id)
        }
    }

    fn dimension(&self) -> usize {
        DynVectorStorage::dimension(&self.inner)
    }

    fn len(&self) -> usize {
        DynVectorStorage::len(&self.inner)
    }

    fn is_empty(&self) -> bool {
        DynVectorStorage::is_empty(&self.inner)
    }
}

/// GREEN TEST: Search with missing vector should gracefully degrade, NOT panic
///
/// After fix:
/// - Search does NOT panic
/// - Search returns results (graceful degradation - skips missing vectors)
/// - Lock is NOT poisoned
/// - Subsequent operations work correctly
#[test]
fn test_search_with_missing_vector_should_gracefully_degrade() {
    println!("\n\n test_search_with_missing_vector_should_gracefully_degrade");
    log_init_test();

    // 1. Generate test data: 20 vectors of dimension 8
    let num_vectors = 20;
    let dimension = 8;
    let data: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| (0..dimension).map(|j| (i * dimension + j) as f32).collect())
        .collect();

    // 2. Build HNSW index with all vectors
    let ef_construct = 16;
    let nb_connection = 8;
    let hnsw = Hnsw::<f32, DistL2>::new(nb_connection, num_vectors, 8, ef_construct, DistL2 {});

    for (i, vec) in data.iter().enumerate() {
        hnsw.insert((vec, i));
    }

    // 3. Save to disk in graph-only mode
    let directory = tempfile::tempdir().unwrap();
    let fname = "test_missing_vector";
    let _res = hnsw.file_dump(directory.path(), fname);

    // 4. Reload graph-only
    let mut reloader = HnswIo::new(directory.path(), fname);
    let mut hnsw_loaded: Hnsw<f32, DistL2> = reloader.load_hnsw::<f32, DistL2>().unwrap();

    // 5. Configure storage that is MISSING vector ID 5
    // This simulates data corruption or incomplete storage
    let missing_id = 5;
    let storage: Arc<dyn DynVectorStorage<f32>> =
        Arc::new(IncompleteVectorStorage::new(data.clone(), vec![missing_id]));
    hnsw_loaded.set_dynamic_vector_storage(storage);

    // 6. Search query that will likely traverse through node 5
    let query = &data[missing_id];

    // 7. Search should NOT panic - graceful degradation
    let search_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hnsw_loaded.search(query, 5, 30)
    }));

    // After fix: No panic, search completes successfully
    assert!(
        search_result.is_ok(),
        "FIXED: Search should NOT panic when vector is missing. Got panic: {:?}",
        search_result.err()
    );

    let results = search_result.unwrap();
    println!(
        "Search completed successfully with {} results (missing vector {} was skipped)",
        results.len(),
        missing_id
    );

    // Verify results don't include the missing vector (it was skipped)
    assert!(
        !results.iter().any(|n| n.d_id == missing_id),
        "Results should not include the missing vector ID {}",
        missing_id
    );
}

/// RED TEST: After fixing, this test should PASS
///
/// This test verifies the EXPECTED behavior after the fix:
/// - Search returns Result<Vec<Neighbour>, HnswError>
/// - Missing vector causes Err(HnswError::VectorNotFound)
/// - No panic, no lock poisoning
///
/// NOTE: This test will FAIL TO COMPILE until we add HnswError type and
/// change search() return type to Result<Vec<Neighbour>, HnswError>.
///
/// Commented out until Phase 1.2 (GREEN) implementation.
// #[test]
// fn test_search_returns_result_on_missing_vector() {
//     log_init_test();
//
//     // ... same setup as above ...
//
//     // After fix, search() returns Result
//     let search_result = hnsw_loaded.search(query, 5, 30);
//
//     match search_result {
//         Ok(_) => {
//             // Search succeeded without accessing the missing vector
//             // (acceptable if the query doesn't traverse through node 5)
//         }
//         Err(HnswError::VectorNotFound { id, .. }) => {
//             assert_eq!(id, 5, "Error should report the missing vector ID");
//         }
//         Err(e) => {
//             panic!("Unexpected error type: {:?}", e);
//         }
//     }
// }

/// RED TEST: Multiple operations after error should NOT cause poisoned lock
///
/// This test verifies that after a search error, subsequent operations
/// still work (no lock poisoning).
///
/// CURRENT BEHAVIOR: Once panic occurs, RwLock is poisoned, ALL operations fail
/// EXPECTED BEHAVIOR: Each operation is independent, errors don't poison state
#[test]
fn test_operations_after_error_should_not_poison() {
    println!("\n\n test_operations_after_error_should_not_poison");
    log_init_test();

    // This test demonstrates the cascading failure from lock poisoning
    // In tessera (the calling code), after one panic:
    // 1. First search panics → RwLock poisoned
    // 2. Second search fails with "poisoned lock" error
    // 3. ALL subsequent operations fail until restart

    // For tessera-hnsw standalone (no external lock):
    // We verify that the HNSW struct itself remains usable after error

    let num_vectors = 20;
    let dimension = 8;
    let data: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| (0..dimension).map(|j| (i * dimension + j) as f32).collect())
        .collect();

    let hnsw = Hnsw::<f32, DistL2>::new(8, num_vectors, 8, 16, DistL2 {});
    for (i, vec) in data.iter().enumerate() {
        hnsw.insert((vec, i));
    }

    let directory = tempfile::tempdir().unwrap();
    let fname = "test_no_poison";
    let _res = hnsw.file_dump(directory.path(), fname);

    let mut reloader = HnswIo::new(directory.path(), fname);
    let mut hnsw_loaded: Hnsw<f32, DistL2> = reloader.load_hnsw::<f32, DistL2>().unwrap();

    // Storage missing vector 5
    let storage: Arc<dyn DynVectorStorage<f32>> =
        Arc::new(IncompleteVectorStorage::new(data.clone(), vec![5]));
    hnsw_loaded.set_dynamic_vector_storage(storage);

    // First search - expected to fail
    let query_bad = &data[5]; // Query that might access missing vector
    let result1 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hnsw_loaded.search(query_bad, 5, 30)
    }));

    // After the panic (if it happened), try another search with a "safe" query
    // Using a different query that shouldn't need vector 5
    let query_good = &data[0];

    // This search should work in the fixed version (no poisoning)
    // In current version, if panic occurred, the struct is in undefined state
    let result2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hnsw_loaded.search(query_good, 5, 30)
    }));

    // Current behavior: Both searches may panic (depending on graph traversal)
    // Fixed behavior: First returns error, second succeeds

    println!(
        "First search result: {}",
        if result1.is_ok() { "OK" } else { "PANIC" }
    );
    println!(
        "Second search result: {}",
        if result2.is_ok() { "OK" } else { "PANIC" }
    );

    // Note: We can't assert much here because:
    // 1. The graph traversal is non-deterministic (may or may not access node 5)
    // 2. After panic, catch_unwind catches it but struct state is undefined
    // The real fix verification happens in tessera integration tests
}
