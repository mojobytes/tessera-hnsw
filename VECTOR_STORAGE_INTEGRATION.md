# VectorStorage Integration Design

## Overview

This document describes the architectural changes needed to integrate the `VectorStorage` trait with the existing HNSW implementation, enabling external vector management.

## Current State

The current implementation stores vectors **inside** the Point structure:

```rust
pub struct Point<'b, T: Clone + Send + Sync> {
    data: PointData<'b, T>,  // Contains actual vector data
    origin_id: DataId,
    p_id: PointId,
    neighbours: Arc<RwLock<Vec<Vec<Arc<PointWithOrder<'b, T>>>>>>,
}

enum PointData<'b, T> {
    V(Vec<T>),      // Owned vector
    S(&'b [T]),     // Reference to mmap slice
}
```

**Problems**:
- Vectors are duplicated when multiple index types (HNSW, IVF, PQ) are used
- Cannot share a single `.tsvf` file between different indexes
- Memory overhead increases linearly with number of index types

## Proposed Architecture

### 1. Add VectorStorage to Hnsw

```rust
pub struct Hnsw<'b, T, D, VS>
where
    T: Clone + Send + Sync + 'b + Debug,
    D: Distance<T>,
    VS: VectorStorage<'b, T>,
{
    // ... existing fields ...

    /// External vector storage (optional, for external vector management)
    pub(crate) vector_storage: Option<&'b VS>,
}
```

### 2. Modify Point to Support Storage-Only Mode

```rust
pub struct Point<'b, T: Clone + Send + Sync> {
    data: PointData<'b, T>,
    origin_id: DataId,
    p_id: PointId,
    neighbours: Arc<RwLock<Vec<Vec<Arc<PointWithOrder<'b, T>>>>>>,
}

enum PointData<'b, T> {
    V(Vec<T>),           // Owned vector (legacy mode)
    S(&'b [T]),          // Mmap reference (legacy mode)
    VectorId(usize),     // NEW: Just store the vector ID
}
```

### 3. Vector Access Pattern

When `VectorStorage` is present, all vector access goes through it:

```rust
impl<'b, T: Clone + Send + Sync> Point<'b, T> {
    pub fn get_v(&self) -> &[T] {
        match self.data {
            PointData::V(ref v) => v.as_slice(),
            PointData::S(s) => s,
            PointData::VectorId(_) => {
                panic!("Point in storage-only mode requires VectorStorage")
            }
        }
    }

    // NEW: Access with storage
    pub fn get_v_with_storage<'a, VS: VectorStorage<'a, T>>(
        &self,
        storage: &'a VS,
    ) -> Option<&'a [T]> {
        match self.data {
            PointData::V(ref v) => Some(v.as_slice()),
            PointData::S(s) => Some(s),
            PointData::VectorId(id) => storage.get_vector(id),
        }
    }
}
```

### 4. Constructor Variants

```rust
impl<'b, T, D, VS> Hnsw<'b, T, D, VS>
where
    T: Clone + Send + Sync + Debug + 'b,
    D: Distance<T> + Send + Sync,
    VS: VectorStorage<'b, T>,
{
    /// Legacy constructor (stores vectors internally)
    pub fn new(
        max_nb_connection: usize,
        max_elements: usize,
        max_layer: usize,
        ef_construction: usize,
        f: D,
    ) -> Self {
        // ... existing implementation ...
        Hnsw {
            // ... fields ...
            vector_storage: None,
        }
    }

    /// NEW: Constructor with external vector storage
    pub fn new_with_storage(
        max_nb_connection: usize,
        max_elements: usize,
        max_layer: usize,
        ef_construction: usize,
        f: D,
        storage: &'b VS,
    ) -> Self {
        Hnsw {
            // ... fields ...
            vector_storage: Some(storage),
        }
    }
}
```

### 5. Insertion Flow Changes

**Current flow**:
```rust
pub fn insert(&self, datav_with_id: (&[T], usize)) {
    // Copies vector into Point
    let point = Point::new(data.to_vec(), origin_id, p_id);
    // ...
}
```

**New flow with storage**:
```rust
pub fn insert(&self, vector_id: usize) {
    // Vector already in storage, just reference it
    let point = Point::new_from_storage(vector_id, p_id);
    // ...
}
```

### 6. Distance Calculation Changes

All distance calculations need to retrieve vectors from storage:

```rust
// BEFORE
let dist = self.dist_f.eval(point1.data.get_v(), point2.data.get_v());

// AFTER (with storage)
let v1 = point1.get_v_with_storage(self.vector_storage.unwrap()).unwrap();
let v2 = point2.get_v_with_storage(self.vector_storage.unwrap()).unwrap();
let dist = self.dist_f.eval(v1, v2);
```

## Migration Strategy

### Phase 1: Add VectorStorage Trait (COMPLETED ✅)
- Created `src/storage.rs` with `VectorStorage` trait
- Implemented `InMemoryVectorStorage` for testing
- Exported module in `lib.rs`

### Phase 2: Backward-Compatible Integration (CURRENT)
1. Add `vector_storage: Option<&'b VS>` to `Hnsw`
2. Add `PointData::VectorId(usize)` variant
3. Add `new_with_storage()` constructor
4. Keep existing `new()` constructor working
5. Modify internal methods to check for storage:
   ```rust
   fn get_vector_data(&self, point: &Point<T>) -> &[T] {
       match self.vector_storage {
           Some(storage) => {
               match point.data {
                   PointData::VectorId(id) => storage.get_vector(id).unwrap(),
                   _ => point.get_v(),
               }
           }
           None => point.get_v(),
       }
   }
   ```

### Phase 3: Update All Vector Access Points
Key methods that need updating:
- `search_layer()` - Line 913-1055
- `insert_slice()` - Line 1068-1206
- `select_neighbours()` - Line 1287-1409
- Distance calculations throughout

### Phase 4: Testing
1. Verify legacy mode still works (existing tests)
2. Add tests for storage mode
3. Benchmark performance (should be similar, maybe slightly faster due to cache locality)

### Phase 5: Integration with Tessera
1. Implement `VectorStorage` for tessera-storage's `.tsvf` files
2. Connect with `PersistentVectorDB`
3. End-to-end testing

## Backward Compatibility

**Critical**: The existing API must continue to work:

```rust
// OLD CODE (should still work)
let hnsw = Hnsw::<f32, DistL2>::new(24, 10000, 16, 400, DistL2{});
hnsw.insert((&vec![0.1, 0.2, 0.3], 1));

// NEW CODE (with external storage)
let storage = InMemoryVectorStorage::new(vectors);
let hnsw = Hnsw::new_with_storage(24, 10000, 16, 400, DistL2{}, &storage);
hnsw.insert(0); // Just pass vector ID
```

## Performance Considerations

### Memory
- **Legacy mode**: O(N * D) where N=vectors, D=dimensions
- **Storage mode**: O(N * C) where C=connectivity (just graph)
- **Savings**: ~(D/C) reduction, typically 10-100x for high-dimensional vectors

### Access Time
- **Legacy mode**: Direct access via pointer
- **Storage mode**: One indirection via VectorStorage trait
- **Impact**: Minimal (<1%), trait should inline in release builds

### Cache Locality
- **Search phase**: Better cache locality (vectors accessed together)
- **Insertion phase**: Similar performance

## Implementation Checklist

- [x] Create VectorStorage trait
- [x] Implement InMemoryVectorStorage
- [x] Export storage module
- [ ] Add generic parameter VS to Hnsw
- [ ] Add vector_storage field to Hnsw
- [ ] Add PointData::VectorId variant
- [ ] Implement new_with_storage() constructor
- [ ] Add helper method for vector access with storage
- [ ] Update search_layer() to use storage
- [ ] Update insert_slice() to use storage
- [ ] Update select_neighbours() to use storage
- [ ] Update distance calculations throughout
- [ ] Add tests for storage mode
- [ ] Benchmark comparison
- [ ] Implement VectorStorage for tessera-storage
- [ ] Integration testing with PersistentVectorDB

## Notes

### Type Signature Complexity
Adding the `VS` generic parameter will make the type signature more complex:
```rust
Hnsw<'b, T, D, VS>
```

This is unavoidable but necessary for zero-cost abstraction. Consider providing type aliases:
```rust
type HnswEmbedded<'b, T, D> = Hnsw<'b, T, D, ()>;  // Legacy mode
type HnswExternal<'b, T, D, VS> = Hnsw<'b, T, D, VS>;  // Storage mode
```

### Lifetime Management
The lifetime `'b` ties the Hnsw to the VectorStorage. This is correct - the graph cannot outlive its vectors.

### Persistence
When dumping to disk:
- **Legacy mode**: Dump vectors with graph (current behavior)
- **Storage mode**: Dump only graph topology, vectors stay in separate `.tsvf` file

---

**Status**: Phase 1 Complete, Phase 2 In Progress
**Last Updated**: 2025-10-16
**Author**: Claude Code + Santiago Fernández Muñoz
