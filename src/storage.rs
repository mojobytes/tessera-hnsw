//! Vector storage trait for external vector management
//!
//! This module provides a trait abstraction for storing vectors outside of the HNSW graph structure.
//! This enables:
//! - Single source of truth for vector storage (e.g., `.tsvf` files)
//! - Multiple index types (HNSW, IVF, PQ) sharing the same vectors
//! - Reduced storage overhead (no duplication)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐
//! │  HNSW Graph     │  ← Stores only graph topology
//! │  (neighbors)    │
//! └────────┬────────┘
//!          │
//!          │ VectorStorage trait
//!          ↓
//! ┌─────────────────┐
//! │ Vector Storage  │  ← Single source of truth for vectors
//! │ (.tsvf, mmap)   │
//! └─────────────────┘
//! ```

use std::fmt::Debug;

/// Trait for dynamic vector storage (Arc-compatible, no lifetime parameter).
///
/// This trait is designed for use with `Arc<dyn DynVectorStorage<T>>` where the lifetime
/// of the returned slice is tied to `&self` rather than a fixed lifetime parameter.
///
/// Use this trait when you need to:
/// - Store vector storage in `Arc<dyn>` (e.g., for graph-only HNSW reload)
/// - Pass vector storage across thread boundaries
/// - Decouple vector storage lifetime from the consumer's lifetime
///
/// # Comparison with `VectorStorage<'a, T>`
///
/// - `VectorStorage<'a, T>`: Lifetime parameter on trait, used for static references
/// - `DynVectorStorage<T>`: No lifetime parameter, used for dynamic dispatch with Arc
///
/// # Example
///
/// ```rust,ignore
/// use tessera_hnsw::storage::{DynVectorStorage, InMemoryVectorStorage};
/// use std::sync::Arc;
///
/// let vectors = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
/// let storage = InMemoryVectorStorage::new(vectors);
///
/// // Can be wrapped in Arc for dynamic dispatch
/// let dyn_storage: Arc<dyn DynVectorStorage<f32>> = Arc::new(storage);
///
/// // Use with graph-only HNSW reload
/// let mut hnsw = reloader.load_hnsw::<f32, DistL2>()?;
/// hnsw.set_dynamic_vector_storage(dyn_storage);
/// ```
pub trait DynVectorStorage<T>: Send + Sync + Debug
where
    T: Clone + Send + Sync,
{
    /// Retrieves a vector by its ID.
    ///
    /// The returned slice's lifetime is tied to `&self`, making this method
    /// compatible with `Arc<dyn DynVectorStorage<T>>`.
    fn get_vector(&self, id: usize) -> Option<&[T]>;

    /// Returns the dimensionality of stored vectors.
    fn dimension(&self) -> usize;

    /// Returns the total number of vectors in storage.
    fn len(&self) -> usize;

    /// Checks if the storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait for external vector storage.
///
/// Implementors of this trait provide access to vectors stored externally to the HNSW structure.
/// This allows the HNSW graph to store only connectivity information while delegating vector
/// storage to specialized backends (e.g., memory-mapped files, databases, etc.).
///
/// # Type Parameters
///
/// * `T` - The element type of vectors (typically `f32` for floating-point embeddings)
///
/// # Safety
///
/// Implementations must ensure that:
/// - Vector IDs remain stable (a given ID always refers to the same vector)
/// - The lifetime `'a` correctly represents the validity period of returned slices
/// - Concurrent access is properly synchronized if the storage is mutable
///
/// # Example
///
/// ```rust,ignore
/// use tessera_hnsw::storage::VectorStorage;
///
/// // Simple in-memory storage
/// struct InMemoryStorage {
///     vectors: Vec<Vec<f32>>,
///     dimension: usize,
/// }
///
/// impl<'a> VectorStorage<'a, f32> for InMemoryStorage {
///     fn get_vector(&'a self, id: usize) -> Option<&'a [f32]> {
///         self.vectors.get(id).map(|v| v.as_slice())
///     }
///
///     fn dimension(&self) -> usize {
///         self.dimension
///     }
///
///     fn len(&self) -> usize {
///         self.vectors.len()
///     }
///
///     fn is_empty(&self) -> bool {
///         self.vectors.is_empty()
///     }
/// }
/// ```
pub trait VectorStorage<'a, T>: Send + Sync + Debug
where
    T: Clone + Send + Sync + 'a,
{
    /// Retrieves a vector by its ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier of the vector (typically 0-indexed)
    ///
    /// # Returns
    ///
    /// * `Some(&[T])` - A slice reference to the vector if it exists
    /// * `None` - If the ID is invalid or the vector doesn't exist
    ///
    /// # Performance
    ///
    /// Implementations should strive for O(1) access time. For large datasets,
    /// consider using memory-mapped files or other efficient storage mechanisms.
    fn get_vector(&'a self, id: usize) -> Option<&'a [T]>;

    /// Returns the dimensionality of vectors stored.
    ///
    /// All vectors in the storage must have the same dimension.
    ///
    /// # Returns
    ///
    /// The number of elements per vector.
    fn dimension(&self) -> usize;

    /// Returns the total number of vectors in storage.
    ///
    /// # Returns
    ///
    /// The count of vectors. This may differ from the maximum valid ID
    /// if the storage has gaps or uses non-sequential IDs.
    fn len(&self) -> usize;

    /// Checks if the storage is empty.
    ///
    /// # Returns
    ///
    /// `true` if no vectors are stored, `false` otherwise.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validates that a vector ID is valid and accessible.
    ///
    /// Default implementation checks if `get_vector(id)` returns `Some`.
    /// Implementations may override this for more efficient validation.
    ///
    /// # Arguments
    ///
    /// * `id` - The vector ID to validate
    ///
    /// # Returns
    ///
    /// `true` if the ID is valid, `false` otherwise.
    fn contains(&'a self, id: usize) -> bool {
        self.get_vector(id).is_some()
    }

    /// Returns metadata about the storage (optional).
    ///
    /// This can be used to provide information about the storage backend,
    /// such as file paths, memory usage, etc.
    ///
    /// # Returns
    ///
    /// A human-readable description of the storage.
    fn storage_info(&self) -> String {
        format!(
            "VectorStorage {{ vectors: {}, dimension: {} }}",
            self.len(),
            self.dimension()
        )
    }
}

/// Simple in-memory vector storage implementation.
///
/// Useful for testing and small datasets. For production use with large datasets,
/// consider using memory-mapped storage or database-backed implementations.
///
/// # Example
///
/// ```rust,ignore
/// use tessera_hnsw::storage::{VectorStorage, InMemoryVectorStorage};
///
/// let vectors = vec![
///     vec![0.1, 0.2, 0.3],
///     vec![0.4, 0.5, 0.6],
///     vec![0.7, 0.8, 0.9],
/// ];
///
/// let storage = InMemoryVectorStorage::new(vectors);
/// assert_eq!(storage.dimension(), 3);
/// assert_eq!(storage.len(), 3);
///
/// let vec0 = storage.get_vector(0).unwrap();
/// assert_eq!(vec0, &[0.1, 0.2, 0.3]);
/// ```
#[derive(Debug, Clone)]
pub struct InMemoryVectorStorage<T: Clone + Send + Sync> {
    vectors: Vec<Vec<T>>,
    dimension: usize,
}

impl<T: Clone + Send + Sync> InMemoryVectorStorage<T> {
    /// Creates a new in-memory vector storage from a vector of vectors.
    ///
    /// # Arguments
    ///
    /// * `vectors` - A vector of vectors to store
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The input is empty
    /// - Vectors have different dimensions
    pub fn new(vectors: Vec<Vec<T>>) -> Self {
        assert!(!vectors.is_empty(), "Cannot create empty vector storage");
        let dimension = vectors[0].len();
        assert!(
            vectors.iter().all(|v| v.len() == dimension),
            "All vectors must have the same dimension"
        );

        InMemoryVectorStorage { vectors, dimension }
    }

    /// Creates storage from a slice of slices (copies data).
    ///
    /// # Arguments
    ///
    /// * `vectors` - A slice of slices to copy into storage
    pub fn from_slices(vectors: &[&[T]]) -> Self {
        assert!(!vectors.is_empty(), "Cannot create empty vector storage");
        let dimension = vectors[0].len();
        assert!(
            vectors.iter().all(|v| v.len() == dimension),
            "All vectors must have the same dimension"
        );

        let owned_vectors: Vec<Vec<T>> = vectors.iter().map(|&v| v.to_vec()).collect();
        InMemoryVectorStorage {
            vectors: owned_vectors,
            dimension,
        }
    }
}

impl<'a, T: Clone + Send + Sync + Debug + 'a> VectorStorage<'a, T> for InMemoryVectorStorage<T> {
    fn get_vector(&'a self, id: usize) -> Option<&'a [T]> {
        self.vectors.get(id).map(|v| v.as_slice())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    fn storage_info(&self) -> String {
        format!(
            "InMemoryVectorStorage {{ vectors: {}, dimension: {}, memory: ~{} KB }}",
            self.vectors.len(),
            self.dimension,
            (self.vectors.len() * self.dimension * std::mem::size_of::<T>()) / 1024
        )
    }
}

/// Implementation of `DynVectorStorage` for `InMemoryVectorStorage`.
///
/// This allows `InMemoryVectorStorage` to be used with `Arc<dyn DynVectorStorage<T>>`
/// for dynamic dispatch scenarios like graph-only HNSW reload.
impl<T: Clone + Send + Sync + Debug> DynVectorStorage<T> for InMemoryVectorStorage<T> {
    fn get_vector(&self, id: usize) -> Option<&[T]> {
        self.vectors.get(id).map(|v| v.as_slice())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_storage_creation() {
        let vectors = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let storage = InMemoryVectorStorage::new(vectors);
        assert_eq!(DynVectorStorage::dimension(&storage), 3);
        assert_eq!(DynVectorStorage::len(&storage), 2);
        assert!(!DynVectorStorage::is_empty(&storage));
    }

    #[test]
    fn test_in_memory_storage_get_vector() {
        let vectors = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let storage = InMemoryVectorStorage::new(vectors);

        let vec0 = DynVectorStorage::get_vector(&storage, 0).unwrap();
        assert_eq!(vec0, &[1.0, 2.0, 3.0]);

        let vec1 = DynVectorStorage::get_vector(&storage, 1).unwrap();
        assert_eq!(vec1, &[4.0, 5.0, 6.0]);

        assert!(DynVectorStorage::get_vector(&storage, 2).is_none());
    }

    #[test]
    fn test_in_memory_storage_contains() {
        let vectors = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];

        let storage = InMemoryVectorStorage::new(vectors);

        // contains() is only defined on VectorStorage, use DynVectorStorage::get_vector
        assert!(DynVectorStorage::get_vector(&storage, 0).is_some());
        assert!(DynVectorStorage::get_vector(&storage, 1).is_some());
        assert!(DynVectorStorage::get_vector(&storage, 2).is_none());
    }

    #[test]
    fn test_in_memory_storage_from_slices() {
        let v0 = vec![1.0f32, 2.0, 3.0];
        let v1 = vec![4.0, 5.0, 6.0];
        let slices: Vec<&[f32]> = vec![v0.as_slice(), v1.as_slice()];

        let storage = InMemoryVectorStorage::from_slices(&slices);

        assert_eq!(DynVectorStorage::dimension(&storage), 3);
        assert_eq!(DynVectorStorage::len(&storage), 2);
        assert_eq!(DynVectorStorage::get_vector(&storage, 0).unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "Cannot create empty vector storage")]
    fn test_in_memory_storage_empty_panics() {
        let vectors: Vec<Vec<f32>> = vec![];
        let _ = InMemoryVectorStorage::new(vectors);
    }

    #[test]
    #[should_panic(expected = "All vectors must have the same dimension")]
    fn test_in_memory_storage_mismatched_dimensions_panics() {
        let vectors = vec![vec![1.0f32, 2.0], vec![3.0, 4.0, 5.0]];
        let _ = InMemoryVectorStorage::new(vectors);
    }

    #[test]
    fn test_storage_info() {
        let vectors = vec![vec![1.0f32; 384]; 1000];
        let storage = InMemoryVectorStorage::new(vectors);

        let info = storage.storage_info();
        assert!(info.contains("1000"));
        assert!(info.contains("384"));
    }
}
