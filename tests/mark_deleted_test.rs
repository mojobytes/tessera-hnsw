//! Phase 2 TDD Tests: mark_deleted in tessera-hnsw
//!
//! Tests for soft-delete support in the HNSW library. Deleted nodes remain
//! in the graph for traversal but are excluded from search results.

use tessera_hnsw::hnsw::{DataId, Hnsw};
use tessera_hnsw::prelude::*;
use tessera_hnsw::distance::DistL2;
use tessera_hnsw::hnswio::HnswIo;

fn build_hnsw_with_vectors(count: usize, dim: usize) -> Hnsw<'static, f32, DistL2, NoStorage> {
    let max_nb_connection = 16;
    let ef_construction = 200;
    let max_layer = 16;
    let max_elements = count;

    let mut hnsw = Hnsw::<f32, DistL2, NoStorage>::new(max_nb_connection, max_elements, max_layer, ef_construction, DistL2 {});

    for i in 0..count {
        let data: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32).collect();
        hnsw.insert((&data, i));
    }
    hnsw.set_searching_mode(true);
    hnsw
}

// ============================================================================
// Cycle 2.1: deleted field and mark_deleted() method
// ============================================================================

/// Test 2.1.1: mark_deleted adds to deleted set and returns true
#[test]
fn test_mark_deleted_adds_to_deleted_set() {
    let hnsw = build_hnsw_with_vectors(100, 128);

    assert_eq!(hnsw.get_nb_point(), 100);
    assert!(!hnsw.is_deleted(5));

    let was_new = hnsw.mark_deleted(5);

    assert!(was_new, "First mark_deleted should return true");
    assert!(hnsw.is_deleted(5));
    assert!(!hnsw.is_deleted(6));
    assert_eq!(hnsw.get_nb_point(), 99);
    assert_eq!(hnsw.deleted_count(), 1);
}

/// Test 2.1.2: mark_deleted on nonexistent ID is accepted but caller should validate
#[test]
fn test_mark_deleted_nonexistent_id() {
    let hnsw = build_hnsw_with_vectors(100, 128);

    // ID 999 was never inserted — mark_deleted still accepts it
    let was_new = hnsw.mark_deleted(999);

    assert!(was_new, "Non-existent ID should still return true (newly inserted into deleted set)");
    assert!(hnsw.is_deleted(999));
    assert_eq!(hnsw.deleted_count(), 1);
    // get_nb_point() = layer_indexed_points.count - deleted.len()
    // Since 999 isn't in the graph, this under-counts (100-1=99)
    assert_eq!(hnsw.get_nb_point(), 99);
}

/// Test 2.1.3: mark_deleted twice is idempotent, second call returns false
#[test]
fn test_mark_deleted_twice_is_idempotent() {
    let hnsw = build_hnsw_with_vectors(100, 128);

    let first = hnsw.mark_deleted(5);
    assert!(first, "First deletion should return true");
    assert_eq!(hnsw.get_nb_point(), 99);
    assert_eq!(hnsw.deleted_count(), 1);

    let second = hnsw.mark_deleted(5);
    assert!(!second, "Second deletion should return false (already deleted)");
    assert_eq!(hnsw.get_nb_point(), 99);
    assert_eq!(hnsw.deleted_count(), 1);
}

/// Test 2.1.4: get_nb_point_total includes deleted points
#[test]
fn test_get_nb_point_total_includes_deleted() {
    let hnsw = build_hnsw_with_vectors(100, 128);

    hnsw.mark_deleted(0);
    hnsw.mark_deleted(1);
    hnsw.mark_deleted(2);

    assert_eq!(hnsw.get_nb_point(), 97);
    assert_eq!(hnsw.get_nb_point_total(), 100);
    assert_eq!(hnsw.deleted_count(), 3);
}

// ============================================================================
// Cycle 2.2: Search skips deleted nodes
// ============================================================================

/// Test 2.2.1: search excludes deleted vectors
#[test]
fn test_search_excludes_deleted_vectors() {
    let hnsw = build_hnsw_with_vectors(1000, 128);

    // Delete vectors 0..10
    for i in 0..10 {
        hnsw.mark_deleted(i);
    }

    // Search for a vector close to ID 0
    let query: Vec<f32> = (0..128).map(|d| d as f32).collect();
    let results = hnsw.search(&query, 10, 30);

    // None of the deleted IDs should appear
    let deleted_ids: std::collections::HashSet<DataId> = (0..10).collect();
    for r in &results {
        assert!(
            !deleted_ids.contains(&r.d_id),
            "Deleted ID {} appeared in search results",
            r.d_id
        );
    }
    assert_eq!(results.len(), 10, "Should still find 10 results from non-deleted vectors");
}

/// Test 2.2.2: search with all top-k deleted finds next best
#[test]
fn test_search_with_top_k_deleted_finds_next_best() {
    let dim = 4;
    let count = 100;
    let mut hnsw = Hnsw::<f32, DistL2, NoStorage>::new(16, count, 16, 200, DistL2 {});

    // Insert vectors where ID i has vector [i, i, i, i]
    // so distance from query [0,0,0,0] is proportional to i
    for i in 0..count {
        let data = vec![i as f32; dim];
        hnsw.insert((&data, i));
    }
    hnsw.set_searching_mode(true);

    // Delete the 5 nearest to [0,0,0,0] (IDs 0..5)
    for i in 0..5 {
        hnsw.mark_deleted(i);
    }

    let query = vec![0.0_f32; dim];
    let results = hnsw.search(&query, 5, 30);

    assert_eq!(results.len(), 5);
    // All results should be from IDs >= 5
    for r in &results {
        assert!(
            r.d_id >= 5,
            "Found deleted ID {} in results (expected >= 5)",
            r.d_id
        );
    }
}

/// Test 2.2.3: mark_deleted uses interior mutability (takes &self)
#[test]
fn test_mark_deleted_interior_mutability() {
    let hnsw = build_hnsw_with_vectors(10, 4);

    // This test verifies &self (not &mut self) — the fact that this compiles
    // and runs proves interior mutability via RwLock
    let hnsw_ref = &hnsw;
    hnsw_ref.mark_deleted(0);
    assert!(hnsw_ref.is_deleted(0));

    // Can still search via shared reference
    let query = vec![0.0_f32; 4];
    let results = hnsw_ref.search(&query, 5, 10);
    for r in &results {
        assert_ne!(r.d_id, 0, "Deleted ID 0 should not appear in results");
    }
}

/// Test 2.2.4: search with all points deleted returns empty
#[test]
fn test_search_all_deleted_returns_empty() {
    let hnsw = build_hnsw_with_vectors(10, 4);

    for i in 0..10 {
        hnsw.mark_deleted(i);
    }

    let query = vec![0.0_f32; 4];
    let results = hnsw.search(&query, 5, 10);
    assert!(results.is_empty(), "All points deleted — search should return empty");
}

/// Test 2.2.5: search_filter combined with mark_deleted
#[test]
fn test_search_filter_combined_with_deleted() {
    let dim = 4;
    let hnsw = build_hnsw_with_vectors(50, dim);

    // Delete IDs 0..5
    for i in 0..5 {
        hnsw.mark_deleted(i);
    }

    // Filter: only allow even IDs
    let even_filter = |id: &DataId| -> bool { *id % 2 == 0 };

    let query = vec![0.0_f32; dim];
    let results = hnsw.search_filter(&query, 10, 30, Some(&even_filter));

    for r in &results {
        assert!(r.d_id % 2 == 0, "Filter should only pass even IDs, got {}", r.d_id);
        assert!(r.d_id >= 5, "Deleted IDs 0..5 should be excluded, got {}", r.d_id);
    }
}

// ============================================================================
// Cycle 2.3: Persistence of deleted set
// ============================================================================

/// Test 2.3.1: save/load preserves deleted set
#[test]
fn test_save_load_preserves_deleted_set() {
    let dim = 4;
    let hnsw = build_hnsw_with_vectors(100, dim);

    // Delete 10 IDs
    let deleted_ids: Vec<DataId> = (0..10).collect();
    for &id in &deleted_ids {
        hnsw.mark_deleted(id);
    }
    assert_eq!(hnsw.deleted_count(), 10);
    assert_eq!(hnsw.get_nb_point(), 90);

    // Dump to temp directory
    let dir = tempfile::tempdir().unwrap();
    let basename = "test_deleted";
    let dump_result = hnsw.file_dump(dir.path(), basename);
    assert!(dump_result.is_ok(), "file_dump failed: {:?}", dump_result.err());
    let actual_basename = dump_result.unwrap();

    // Reload
    let mut reloader = HnswIo::new(dir.path(), &actual_basename);
    let loaded: Hnsw<f32, DistL2> = reloader.load_hnsw().unwrap();

    // Verify deleted set preserved
    assert_eq!(loaded.deleted_count(), 10);
    assert_eq!(loaded.get_nb_point(), 90);
    for &id in &deleted_ids {
        assert!(loaded.is_deleted(id), "ID {} should be deleted after reload", id);
    }
    // Non-deleted should remain non-deleted
    assert!(!loaded.is_deleted(50));
}

/// Test 2.3.2: load old format without deleted set file defaults to empty
#[test]
fn test_load_old_format_without_deleted_defaults_to_empty() {
    let dim = 4;
    let count = 50;
    let mut hnsw = Hnsw::<f32, DistL2, NoStorage>::new(16, count, 16, 200, DistL2 {});
    for i in 0..count {
        let data = vec![i as f32; dim];
        hnsw.insert((&data, i));
    }
    hnsw.set_searching_mode(true);

    // Dump WITHOUT any deletions (no .hnsw.deleted file created)
    let dir = tempfile::tempdir().unwrap();
    let basename = "test_no_deleted";
    let actual_basename = hnsw.file_dump(dir.path(), basename).unwrap();

    // Verify no .hnsw.deleted file exists
    let deleted_path = dir.path().join(format!("{}.hnsw.deleted", actual_basename));
    assert!(!deleted_path.exists(), "Empty deleted set should not create a file");

    // Reload — should have empty deleted set
    let mut reloader = HnswIo::new(dir.path(), &actual_basename);
    let loaded: Hnsw<f32, DistL2> = reloader.load_hnsw().unwrap();

    assert_eq!(loaded.deleted_count(), 0);
    assert_eq!(loaded.get_nb_point(), count);
}

/// Test 2.3.3: save with deletions, then save again after clearing — file removed
#[test]
fn test_save_deleted_then_clear_removes_file() {
    let dim = 4;
    let count = 50;
    let mut hnsw = Hnsw::<f32, DistL2, NoStorage>::new(16, count, 16, 200, DistL2 {});
    for i in 0..count {
        let data = vec![i as f32; dim];
        hnsw.insert((&data, i));
    }
    hnsw.set_searching_mode(true);

    let dir = tempfile::tempdir().unwrap();
    let basename = "test_clear";

    // First dump with deletions
    hnsw.mark_deleted(0);
    hnsw.mark_deleted(1);
    let actual_basename = hnsw.file_dump(dir.path(), basename).unwrap();
    let deleted_path = dir.path().join(format!("{}.hnsw.deleted", actual_basename));
    assert!(deleted_path.exists(), "Deleted file should exist after dump with deletions");

    // Now build a fresh index with no deletions and dump with same basename
    let mut hnsw2 = Hnsw::<f32, DistL2, NoStorage>::new(16, count, 16, 200, DistL2 {});
    for i in 0..count {
        let data = vec![i as f32; dim];
        hnsw2.insert((&data, i));
    }
    hnsw2.set_searching_mode(true);
    let _ = hnsw2.file_dump(dir.path(), &actual_basename).unwrap();

    // The .hnsw.deleted file should be removed (empty deleted set cleans up stale file)
    let deleted_path2 = dir.path().join(format!("{}.hnsw.deleted", actual_basename));
    assert!(!deleted_path2.exists(), "Deleted file should be removed when no deletions");

    // Reload should have empty deleted set
    let mut reloader = HnswIo::new(dir.path(), &actual_basename);
    let loaded: Hnsw<f32, DistL2> = reloader.load_hnsw().unwrap();
    assert_eq!(loaded.deleted_count(), 0);
}

// ============================================================================
// Cycle 2.3+: Binary format portability (F1: endianness fix)
// ============================================================================

/// Test: save/load uses LE u64 format for DataIds (portable across architectures)
#[test]
fn test_save_load_deleted_set_uses_le_u64_format() {
    let dim = 4;
    let hnsw = build_hnsw_with_vectors(100, dim);

    let deleted_ids: Vec<DataId> = vec![42, 100, 999];
    for &id in &deleted_ids {
        hnsw.mark_deleted(id);
    }

    let dir = tempfile::tempdir().unwrap();
    let basename = "test_le_format";
    let actual_basename = hnsw.file_dump(dir.path(), basename).unwrap();

    // Read raw bytes and verify format
    let deleted_path = dir.path().join(format!("{}.hnsw.deleted", actual_basename));
    let raw = std::fs::read(&deleted_path).unwrap();

    // Header: 4 bytes magic + 8 bytes count + 3 × 8 bytes IDs = 36 bytes
    assert_eq!(raw.len(), 4 + 8 + 3 * 8, "File size should be 36 bytes for 3 IDs");

    // Magic: u32 LE
    let magic = u32::from_le_bytes(raw[0..4].try_into().unwrap());
    assert_eq!(magic, 0x00de1e7e, "Magic should be MAGIC_DELETED in LE");

    // Count: u64 LE
    let count = u64::from_le_bytes(raw[4..12].try_into().unwrap());
    assert_eq!(count, 3, "Count should be 3");

    // All IDs should be u64 LE (readable on any architecture)
    let mut loaded_ids: Vec<DataId> = Vec::new();
    for i in 0..3 {
        let offset = 12 + i * 8;
        let id = u64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap()) as DataId;
        loaded_ids.push(id);
    }
    loaded_ids.sort();
    let mut expected = deleted_ids.clone();
    expected.sort();
    assert_eq!(loaded_ids, expected, "IDs should be stored as u64 LE");
}

/// Test: load reads u64 LE IDs correctly (manually crafted file)
#[test]
fn test_load_deleted_set_reads_le_u64_ids() {
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("manual.hnsw.deleted");

    // Manually write a valid .hnsw.deleted file
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&0x00de1e7eu32.to_le_bytes()).unwrap(); // magic
    file.write_all(&2u64.to_le_bytes()).unwrap(); // count = 2
    file.write_all(&42u64.to_le_bytes()).unwrap(); // ID 42
    file.write_all(&999u64.to_le_bytes()).unwrap(); // ID 999
    file.flush().unwrap();
    drop(file);

    // Build an hnsw and dump it so we have valid graph files
    let dim = 4;
    let mut hnsw = Hnsw::<f32, DistL2, NoStorage>::new(16, 10, 16, 200, DistL2 {});
    for i in 0..10 {
        let data = vec![i as f32; dim];
        hnsw.insert((&data, i));
    }
    hnsw.set_searching_mode(true);
    hnsw.file_dump(dir.path(), "manual").unwrap();

    // Overwrite the deleted file with our manually crafted one
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&0x00de1e7eu32.to_le_bytes()).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap();
    file.write_all(&42u64.to_le_bytes()).unwrap();
    file.write_all(&999u64.to_le_bytes()).unwrap();
    file.flush().unwrap();
    drop(file);

    // Reload and verify
    let mut reloader = HnswIo::new(dir.path(), "manual");
    let loaded: Hnsw<f32, DistL2> = reloader.load_hnsw().unwrap();

    assert_eq!(loaded.deleted_count(), 2);
    assert!(loaded.is_deleted(42));
    assert!(loaded.is_deleted(999));
    assert!(!loaded.is_deleted(0));
}

// ============================================================================
// Phase 7: Edge-case tests (F10)
// ============================================================================

/// Test: delete half the vectors, then verify remaining are searchable
#[test]
fn test_mark_deleted_half_then_search_works() {
    let dim = 4;
    let count = 20;
    let mut hnsw = Hnsw::<f32, DistL2, NoStorage>::new(16, count, 16, 200, DistL2 {});

    for i in 0..count {
        let data = vec![i as f32; dim];
        hnsw.insert((&data, i));
    }
    hnsw.set_searching_mode(true);

    // Delete the first half (0..10)
    for i in 0..10 {
        hnsw.mark_deleted(i);
    }

    assert_eq!(hnsw.get_nb_point(), 10);

    let query = vec![10.0_f32; dim];
    let results = hnsw.search(&query, 5, 30);

    assert!(!results.is_empty(), "Should find results from non-deleted vectors");
    for r in &results {
        assert!(r.d_id >= 10, "Deleted ID {} should not appear in results", r.d_id);
    }
}

/// Test: deleted set persists across multiple save/load cycles
#[test]
fn test_deleted_set_persists_across_multiple_save_load_cycles() {
    let dim = 4;
    let hnsw = build_hnsw_with_vectors(100, dim);

    let dir = tempfile::tempdir().unwrap();

    // Cycle 1: delete 10, save, load
    for i in 0..10 {
        hnsw.mark_deleted(i);
    }
    let basename1 = hnsw.file_dump(dir.path(), "multi_cycle").unwrap();
    let mut reloader1 = HnswIo::new(dir.path(), &basename1);
    let loaded1: Hnsw<f32, DistL2> = reloader1.load_hnsw().unwrap();
    assert_eq!(loaded1.deleted_count(), 10);

    // Cycle 2: delete 10 more on loaded index, save, load
    for i in 10..20 {
        loaded1.mark_deleted(i);
    }
    assert_eq!(loaded1.deleted_count(), 20);

    // Must re-dump (file_dump requires mut for set_searching_mode in some paths,
    // but the loaded index already has vectors in graph mode)
    let basename2 = loaded1.file_dump(dir.path(), &basename1).unwrap();
    let mut reloader2 = HnswIo::new(dir.path(), &basename2);
    let loaded2: Hnsw<f32, DistL2> = reloader2.load_hnsw().unwrap();

    assert_eq!(loaded2.deleted_count(), 20);
    assert_eq!(loaded2.get_nb_point(), 80);

    // Verify all 20 IDs are deleted
    for i in 0..20 {
        assert!(loaded2.is_deleted(i), "ID {} should be deleted after 2 cycles", i);
    }
    for i in 20..100 {
        assert!(!loaded2.is_deleted(i), "ID {} should NOT be deleted", i);
    }
}

/// Test: search with 80% deletion ratio still finds remaining vectors
#[test]
fn test_search_with_large_deleted_ratio() {
    let dim = 4;
    let count = 100;
    let mut hnsw = Hnsw::<f32, DistL2, NoStorage>::new(16, count, 16, 200, DistL2 {});

    for i in 0..count {
        let data = vec![i as f32; dim];
        hnsw.insert((&data, i));
    }
    hnsw.set_searching_mode(true);

    // Delete 80 out of 100 (80% deletion ratio)
    for i in 0..80 {
        hnsw.mark_deleted(i);
    }

    assert_eq!(hnsw.get_nb_point(), 20);

    let query = vec![80.0_f32; dim];
    let results = hnsw.search(&query, 5, 100); // High ef_search to compensate for deleted ratio

    assert!(!results.is_empty(), "Should find at least 1 result with 20 remaining vectors");
    for r in &results {
        assert!(
            r.d_id >= 80,
            "Result ID {} should be >= 80 (non-deleted range)",
            r.d_id
        );
    }
}
