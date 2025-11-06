//! Distance functions for vector similarity calculations
//!
//! High-performance distance implementations using simsimd SIMD backend.
//! Supports common distance metrics: L2 (Euclidean), L1 (Manhattan),
//! Cosine, Dot Product, and Hamming distance.
//!
//! Additional distances (Jaccard, Jeffreys, etc.) available with
//! `anndists-fallback` feature flag.

use simsimd::{BinarySimilarity, SpatialSimilarity};

/// Error type for distance calculations
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceError {
    /// Vector lengths don't match
    LengthMismatch {
        expected: usize,
        got: usize,
    },
    /// Empty vectors provided
    EmptyVectors,
    /// Invalid floating point value encountered (NaN or Infinity)
    InvalidValue(String),
    /// Feature not available without anndists-fallback
    FeatureNotAvailable(String),
}

impl std::fmt::Display for DistanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceError::LengthMismatch { expected, got } => {
                write!(f, "Vector length mismatch: expected {}, got {}", expected, got)
            }
            DistanceError::EmptyVectors => write!(f, "Empty vectors not allowed"),
            DistanceError::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
            DistanceError::FeatureNotAvailable(msg) => write!(f, "Feature not available: {}", msg),
        }
    }
}

impl std::error::Error for DistanceError {}

/// Trait for distance calculation between vectors
pub trait Distance<T: Send + Sync> {
    /// Calculate distance between two vectors
    ///
    /// # Errors
    ///
    /// Returns `DistanceError` if:
    /// - Vectors have different lengths
    /// - Vectors are empty
    /// - Invalid values (NaN, Infinity) are encountered
    fn eval(&self, va: &[T], vb: &[T]) -> Result<f32, DistanceError>;
}

/// Validate input vectors (length and emptiness)
#[inline]
fn validate_inputs<T>(va: &[T], vb: &[T]) -> Result<(), DistanceError> {
    if va.is_empty() || vb.is_empty() {
        return Err(DistanceError::EmptyVectors);
    }

    if va.len() != vb.len() {
        return Err(DistanceError::LengthMismatch {
            expected: va.len(),
            got: vb.len(),
        });
    }

    Ok(())
}

// Module will be populated with distance implementations in Phase 2

/// Euclidean distance (L2)
#[derive(Debug, Clone, Copy, Default)]
pub struct DistL2;

impl Distance<f32> for DistL2 {
    fn eval(&self, va: &[f32], vb: &[f32]) -> Result<f32, DistanceError> {
        // Validate inputs
        if va.is_empty() || vb.is_empty() {
            return Err(DistanceError::EmptyVectors);
        }

        if va.len() != vb.len() {
            return Err(DistanceError::LengthMismatch {
                expected: va.len(),
                got: vb.len(),
            });
        }

        // Debug assertions for NaN/Inf (only in debug builds for performance)
        debug_assert!(
            !va.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Input vector contains NaN or Infinity"
        );
        debug_assert!(
            !vb.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Input vector contains NaN or Infinity"
        );

        // Calculate distance using simsimd
        let sqd = f32::sqeuclidean(va, vb)
            .ok_or_else(|| DistanceError::InvalidValue("simsimd sqeuclidean returned None".to_string()))?;

        Ok((sqd as f32).sqrt())
    }
}

impl Distance<f64> for DistL2 {
    fn eval(&self, va: &[f64], vb: &[f64]) -> Result<f32, DistanceError> {
        validate_inputs(va, vb)?;

        debug_assert!(
            !va.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Input vector contains NaN or Infinity"
        );
        debug_assert!(
            !vb.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Input vector contains NaN or Infinity"
        );

        let sqd = f64::sqeuclidean(va, vb)
            .ok_or_else(|| DistanceError::InvalidValue("simsimd sqeuclidean returned None".to_string()))?;

        Ok((sqd as f32).sqrt())
    }
}

// Macro for integer implementations of DistL2
macro_rules! impl_dist_l2_integer {
    ($($t:ty),+) => {
        $(
            impl Distance<$t> for DistL2 {
                fn eval(&self, va: &[$t], vb: &[$t]) -> Result<f32, DistanceError> {
                    validate_inputs(va, vb)?;

                    let sum: f32 = va.iter()
                        .zip(vb.iter())
                        .map(|(a, b)| {
                            let diff = (*a as f32) - (*b as f32);
                            diff * diff
                        })
                        .sum();

                    Ok(sum.sqrt())
                }
            }
        )+
    };
}

// For integer types, convert to f32 first (matches anndists behavior)
impl_dist_l2_integer!(i32, u32, u16, u8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dist_l2_f32_basic() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        let dist = DistL2.eval(&a, &b).unwrap();

        // Expected: sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
        assert!((dist - 5.196).abs() < 0.01, "Got {}", dist);
    }

    #[test]
    fn test_dist_l2_f32_zero() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let dist = DistL2.eval(&a, &a).unwrap();
        assert!((dist - 0.0).abs() < 1e-6, "Got {}", dist);
    }

    #[test]
    fn test_dist_l2_i32() {
        let a = vec![1_i32, 2, 3];
        let b = vec![4_i32, 5, 6];
        let dist = DistL2.eval(&a, &b).unwrap();
        assert!((dist - 5.196).abs() < 0.01, "Got {}", dist);
    }

    // ========== Edge Case Tests ==========

    #[test]
    fn test_empty_vectors_error() {
        let empty: Vec<f32> = vec![];
        let non_empty = vec![1.0_f32, 2.0];

        // Both empty
        let result = DistL2.eval(&empty, &empty);
        assert!(matches!(result, Err(DistanceError::EmptyVectors)));

        // First empty
        let result = DistL2.eval(&empty, &non_empty);
        assert!(matches!(result, Err(DistanceError::EmptyVectors)));

        // Second empty
        let result = DistL2.eval(&non_empty, &empty);
        assert!(matches!(result, Err(DistanceError::EmptyVectors)));
    }

    #[test]
    fn test_length_mismatch_error() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.0];

        let result = DistL2.eval(&a, &b);
        assert!(matches!(
            result,
            Err(DistanceError::LengthMismatch { expected: 3, got: 2 })
        ));

        // Reverse direction
        let result = DistL2.eval(&b, &a);
        assert!(matches!(
            result,
            Err(DistanceError::LengthMismatch { expected: 2, got: 3 })
        ));
    }

    #[test]
    fn test_single_element_vectors() {
        let a = vec![5.0_f32];
        let b = vec![8.0_f32];

        // L2: |8-5| = 3
        let dist = DistL2.eval(&a, &b).unwrap();
        assert!((dist - 3.0).abs() < 1e-6, "Got {}", dist);

        // Same element
        let dist = DistL2.eval(&a, &a).unwrap();
        assert!((dist - 0.0).abs() < 1e-6, "Got {}", dist);
    }

    #[test]
    fn test_large_dimension_vectors() {
        let dim = 1024;
        let a: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect();

        // Distance should be sqrt(1024) = 32
        let dist = DistL2.eval(&a, &b).unwrap();
        assert!((dist - 32.0).abs() < 0.01, "Got {}", dist);
    }

    #[test]
    fn test_negative_values() {
        let a = vec![-1.0_f32, -2.0, -3.0];
        let b = vec![1.0_f32, 2.0, 3.0];

        // Expected: sqrt((1-(-1))^2 + (2-(-2))^2 + (3-(-3))^2) = sqrt(4 + 16 + 36) = sqrt(56) ≈ 7.48
        let dist = DistL2.eval(&a, &b).unwrap();
        assert!((dist - 7.48).abs() < 0.01, "Got {}", dist);
    }

    #[test]
    fn test_zero_vectors() {
        let zero = vec![0.0_f32, 0.0, 0.0];
        let non_zero = vec![3.0_f32, 4.0, 0.0];

        // Distance from zero to (3, 4, 0) is 5
        let dist = DistL2.eval(&zero, &non_zero).unwrap();
        assert!((dist - 5.0).abs() < 1e-6, "Got {}", dist);

        // Two zero vectors
        let dist = DistL2.eval(&zero, &zero).unwrap();
        assert!((dist - 0.0).abs() < 1e-6, "Got {}", dist);
    }

    #[test]
    fn test_dist_l1_basic() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];

        // L1: |4-1| + |5-2| + |6-3| = 3 + 3 + 3 = 9
        let dist = DistL1.eval(&a, &b).unwrap();
        assert!((dist - 9.0).abs() < 1e-6, "Got {}", dist);
    }

    #[test]
    fn test_dist_l1_negative() {
        let a = vec![-1.0_f32, -2.0, -3.0];
        let b = vec![1.0_f32, 2.0, 3.0];

        // L1: |1-(-1)| + |2-(-2)| + |3-(-3)| = 2 + 4 + 6 = 12
        let dist = DistL1.eval(&a, &b).unwrap();
        assert!((dist - 12.0).abs() < 1e-6, "Got {}", dist);
    }

    #[test]
    fn test_dist_cosine_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];

        // Orthogonal vectors: cosine distance ≈ 1 (similarity = 0)
        // simsimd's cosine() returns distance directly
        let dist = DistCosine.eval(&a, &b).unwrap();
        assert!((dist - 1.0).abs() < 0.1, "Got {} (expected ~1.0)", dist);
    }

    #[test]
    fn test_dist_cosine_parallel() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![2.0_f32, 4.0, 6.0]; // Same direction

        // Parallel vectors: cosine distance ≈ 0 (similarity = 1)
        // simsimd's cosine() returns distance directly
        let dist = DistCosine.eval(&a, &b).unwrap();
        assert!((dist - 0.0).abs() < 0.1, "Got {}", dist);
    }

    #[test]
    fn test_dist_cosine_opposite() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![-1.0_f32, -2.0, -3.0]; // Opposite direction

        // Opposite vectors: cosine distance ≈ 2 (similarity = -1)
        // simsimd's cosine() returns distance directly
        let dist = DistCosine.eval(&a, &b).unwrap();
        assert!((dist - 2.0).abs() < 0.1, "Got {}", dist);
    }

    #[test]
    fn test_dist_dot_basic() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];

        // Dot product: 1*4 + 2*5 + 3*6 = 32, negated = -32
        let dist = DistDot.eval(&a, &b).unwrap();
        assert!((dist + 32.0).abs() < 1e-6, "Got {}", dist);
    }

    #[test]
    fn test_dist_dot_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];

        // Orthogonal: dot product = 0, negated = 0
        let dist = DistDot.eval(&a, &b).unwrap();
        assert!((dist - 0.0).abs() < 1e-6, "Got {}", dist);
    }

    #[test]
    fn test_dist_hamming_identical() {
        let a = vec![1_u32, 2, 3, 4];
        let b = vec![1_u32, 2, 3, 4];

        // Identical vectors: Hamming distance = 0
        let dist = DistHamming.eval(&a, &b).unwrap();
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_dist_hamming_different() {
        let a = vec![1_u32, 2, 3, 4];
        let b = vec![1_u32, 5, 3, 7];

        // 2 different elements: Hamming distance = 2
        let dist = DistHamming.eval(&a, &b).unwrap();
        assert_eq!(dist, 2.0);
    }

    #[test]
    fn test_mixed_integer_types() {
        // u8
        let a_u8 = vec![1_u8, 2, 3];
        let b_u8 = vec![4_u8, 5, 6];
        let dist = DistL2.eval(&a_u8, &b_u8).unwrap();
        assert!((dist - 5.196).abs() < 0.01, "Got {}", dist);

        // u16
        let a_u16 = vec![1_u16, 2, 3];
        let b_u16 = vec![4_u16, 5, 6];
        let dist = DistL2.eval(&a_u16, &b_u16).unwrap();
        assert!((dist - 5.196).abs() < 0.01, "Got {}", dist);

        // u32
        let a_u32 = vec![1_u32, 2, 3];
        let b_u32 = vec![4_u32, 5, 6];
        let dist = DistL2.eval(&a_u32, &b_u32).unwrap();
        assert!((dist - 5.196).abs() < 0.01, "Got {}", dist);
    }
}

/// Manhattan distance (L1)
#[derive(Debug, Clone, Copy, Default)]
pub struct DistL1;

impl Distance<f32> for DistL1 {
    fn eval(&self, va: &[f32], vb: &[f32]) -> Result<f32, DistanceError> {
        validate_inputs(va, vb)?;

        debug_assert!(
            !va.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Input vector contains NaN or Infinity"
        );
        debug_assert!(
            !vb.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Input vector contains NaN or Infinity"
        );

        Ok(va.iter().zip(vb.iter()).map(|(a, b)| (a - b).abs()).sum())
    }
}

// Macro for integer implementations of DistL1
macro_rules! impl_dist_l1_integer {
    ($($t:ty),+) => {
        $(
            impl Distance<$t> for DistL1 {
                fn eval(&self, va: &[$t], vb: &[$t]) -> Result<f32, DistanceError> {
                    validate_inputs(va, vb)?;
                    Ok(va.iter().zip(vb.iter())
                        .map(|(a, b)| (*a as f32 - *b as f32).abs())
                        .sum())
                }
            }
        )+
    };
}

impl_dist_l1_integer!(i32, u32, u16, u8);

/// Cosine distance
/// Note: simsimd's cosine() returns similarity, so we compute 1 - similarity for distance
#[derive(Debug, Clone, Copy, Default)]
pub struct DistCosine;

impl Distance<f32> for DistCosine {
    fn eval(&self, va: &[f32], vb: &[f32]) -> Result<f32, DistanceError> {
        validate_inputs(va, vb)?;

        debug_assert!(
            !va.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Input vector contains NaN or Infinity"
        );
        debug_assert!(
            !vb.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Input vector contains NaN or Infinity"
        );

        // simsimd's cosine() returns DISTANCE (1 - similarity), not raw similarity
        let distance = f32::cosine(va, vb)
            .ok_or_else(|| DistanceError::InvalidValue("simsimd cosine returned None".to_string()))?;

        Ok(distance as f32)
    }
}

/// Dot product (negative for distance semantics)
#[derive(Debug, Clone, Copy, Default)]
pub struct DistDot;

impl Distance<f32> for DistDot {
    fn eval(&self, va: &[f32], vb: &[f32]) -> Result<f32, DistanceError> {
        validate_inputs(va, vb)?;

        debug_assert!(
            !va.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Input vector contains NaN or Infinity"
        );
        debug_assert!(
            !vb.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Input vector contains NaN or Infinity"
        );

        let dot = f32::dot(va, vb)
            .ok_or_else(|| DistanceError::InvalidValue("simsimd dot returned None".to_string()))?;

        Ok(-(dot as f32))
    }
}

/// Hamming distance (count of differing elements)
#[derive(Debug, Clone, Copy, Default)]
pub struct DistHamming;

impl Distance<u8> for DistHamming {
    fn eval(&self, va: &[u8], vb: &[u8]) -> Result<f32, DistanceError> {
        validate_inputs(va, vb)?;

        let hamming = u8::hamming(va, vb)
            .ok_or_else(|| DistanceError::InvalidValue("simsimd hamming returned None".to_string()))?;

        Ok(hamming as f32)
    }
}

// Macro for manual Hamming implementations (integer types without simsimd support)
macro_rules! impl_dist_hamming_manual {
    ($($t:ty),+) => {
        $(
            impl Distance<$t> for DistHamming {
                fn eval(&self, va: &[$t], vb: &[$t]) -> Result<f32, DistanceError> {
                    validate_inputs(va, vb)?;
                    Ok(va.iter().zip(vb.iter()).filter(|(a, b)| a != b).count() as f32)
                }
            }
        )+
    };
}

impl_dist_hamming_manual!(u16, u32, i32);

/// Re-export unsupported distances from anndists (optional fallback)
#[cfg(feature = "anndists-fallback")]
pub use anndists::dist::distances::{
    DistJaccard, DistJeffreys, DistJensenShannon, DistLevenshtein, NoDist,
};

// If anndists not available, provide stub implementations
#[cfg(not(feature = "anndists-fallback"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct DistJaccard;
#[cfg(not(feature = "anndists-fallback"))]
impl<T: Send + Sync> Distance<T> for DistJaccard {
    fn eval(&self, _va: &[T], _vb: &[T]) -> Result<f32, DistanceError> {
        Err(DistanceError::FeatureNotAvailable(
            "DistJaccard requires anndists-fallback feature".to_string()
        ))
    }
}

#[cfg(not(feature = "anndists-fallback"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct DistJeffreys;
#[cfg(not(feature = "anndists-fallback"))]
impl<T: Send + Sync> Distance<T> for DistJeffreys {
    fn eval(&self, _va: &[T], _vb: &[T]) -> Result<f32, DistanceError> {
        Err(DistanceError::FeatureNotAvailable(
            "DistJeffreys requires anndists-fallback feature".to_string()
        ))
    }
}

#[cfg(not(feature = "anndists-fallback"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct DistJensenShannon;
#[cfg(not(feature = "anndists-fallback"))]
impl<T: Send + Sync> Distance<T> for DistJensenShannon {
    fn eval(&self, _va: &[T], _vb: &[T]) -> Result<f32, DistanceError> {
        Err(DistanceError::FeatureNotAvailable(
            "DistJensenShannon requires anndists-fallback feature".to_string()
        ))
    }
}

#[cfg(not(feature = "anndists-fallback"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct DistLevenshtein;
#[cfg(not(feature = "anndists-fallback"))]
impl<T: Send + Sync> Distance<T> for DistLevenshtein {
    fn eval(&self, _va: &[T], _vb: &[T]) -> Result<f32, DistanceError> {
        Err(DistanceError::FeatureNotAvailable(
            "DistLevenshtein requires anndists-fallback feature".to_string()
        ))
    }
}

#[cfg(not(feature = "anndists-fallback"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct NoDist;
#[cfg(not(feature = "anndists-fallback"))]
impl<T: Send + Sync> Distance<T> for NoDist {
    fn eval(&self, _va: &[T], _vb: &[T]) -> Result<f32, DistanceError> {
        // NoDist always returns 0.0 - validates inputs but always succeeds
        validate_inputs(_va, _vb)?;
        Ok(0.0)
    }
}

#[cfg(not(feature = "anndists-fallback"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct DistHellinger;
#[cfg(not(feature = "anndists-fallback"))]
impl<T: Send + Sync> Distance<T> for DistHellinger {
    fn eval(&self, _va: &[T], _vb: &[T]) -> Result<f32, DistanceError> {
        Err(DistanceError::FeatureNotAvailable(
            "DistHellinger requires anndists-fallback feature".to_string()
        ))
    }
}

#[cfg(not(feature = "anndists-fallback"))]
#[derive(Debug, Clone)]
pub struct DistCFFI<T> {
    _phantom: std::marker::PhantomData<T>,
}
#[cfg(not(feature = "anndists-fallback"))]
impl<T: Send + Sync> DistCFFI<T> {
    pub fn new(_func: extern "C" fn(*const T, *const T, u64) -> f32) -> Self {
        unimplemented!("DistCFFI requires anndists-fallback feature")
    }
}
#[cfg(not(feature = "anndists-fallback"))]
impl<T: Send + Sync> Distance<T> for DistCFFI<T> {
    fn eval(&self, _va: &[T], _vb: &[T]) -> Result<f32, DistanceError> {
        Err(DistanceError::FeatureNotAvailable(
            "DistCFFI requires anndists-fallback feature".to_string()
        ))
    }
}

/// Distance from function pointer
#[derive(Clone)]
pub struct DistPtr<T, D> {
    func: fn(&[T], &[D]) -> f32,
}

impl<T: Send + Sync, D: Send + Sync> DistPtr<T, D> {
    pub fn new(func: fn(&[T], &[D]) -> f32) -> Self {
        DistPtr { func }
    }
}

impl<T: Send + Sync, D: Send + Sync> Distance<T> for DistPtr<T, D>
where
    T: Clone,
    D: From<T>,
{
    fn eval(&self, va: &[T], vb: &[T]) -> Result<f32, DistanceError> {
        validate_inputs(va, vb)?;

        let vb_converted: Vec<D> = vb.iter().map(|x| D::from(x.clone())).collect();
        Ok((self.func)(va, &vb_converted))
    }
}

/// L2 normalize a vector in place
pub fn l2_normalize<T>(v: &mut [T])
where
    T: num_traits::Float + std::iter::Sum,
{
    let norm: T = v.iter().map(|x| *x * *x).sum::<T>().sqrt();
    if norm > T::zero() {
        for x in v.iter_mut() {
            *x = *x / norm;
        }
    }
}

