# tessera-hnsw

> **Fork of [hnswlib-rs](https://github.com/jean-pierreBoth/hnswlib-rs)** with architectural modifications for the Tessera vector database.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## About This Fork

This is a fork of the excellent [hnsw_rs](https://github.com/jean-pierreBoth/hnswlib-rs) crate by Jean Pierre Both, modified to support external vector storage for the [Tessera vector database](https://github.com/mojobytes/tessera) project.

### Key Differences from Upstream

1. **VectorStorage Trait**: Added abstraction layer to enable external vector management
2. **Separated Persistence**: Graph topology can be persisted independently of vectors
3. **Tessera Integration**: Designed to work with `tessera-storage` for unified vector management

### Why Fork?

The original `hnsw_rs` embeds vectors within the HNSW structure. For a multi-index vector database (HNSW, IVF, PQ, etc.), this would require duplicating vectors for each index algorithm.

This fork enables:
- ✅ Single source of truth for vector storage (`.tsvf` files)
- ✅ Multiple index types sharing the same vectors
- ✅ Reduced storage overhead (no duplication)
- ✅ Consistent vector access patterns

## Original Features

This fork maintains all the excellent features of the original `hnsw_rs`:

- Multithreaded insertion and search
- Standard distance metrics (L1, L2, Cosine, Jaccard, Hamming, etc.)
- Custom distance functions via Trait
- Filtering during search
- Dump and reload functionality
- Excellent performance (see original benchmarks)

## Usage

**Note**: This crate is currently under active development as part of Tessera. API may change.

```rust
use tessera_hnsw::{Hnsw, dist::DistL2};

// Basic usage (compatible with upstream)
let max_nb_connection = 24;
let nb_elements = 10000;
let nb_layers = 16;
let ef_construction = 400;

let mut hnsw = Hnsw::<f32, DistL2>::new(
    max_nb_connection,
    nb_elements,
    nb_layers,
    ef_construction,
    DistL2{}
);

// Insert vectors
let data: Vec<(&Vec<f32>, usize)> = vec![
    (&vec![0.1, 0.2, 0.3], 1),
    (&vec![0.4, 0.5, 0.6], 2),
];
hnsw.parallel_insert(&data);

// Search
let query = vec![0.15, 0.25, 0.35];
let k = 10;
let ef_search = 50;
let results = hnsw.parallel_search(&[query], k, ef_search);
```

### With VectorStorage (New)

```rust
// Coming soon - external vector storage integration
// Will allow graph-only persistence with vectors in Tessera storage
```

## Building

```bash
# Standard build
cargo build --release

# With SIMD (x86_64)
cargo build --release --features simdeez_f

# With portable SIMD (requires nightly)
cargo build --release --features stdsimd
```

## Attribution

See [ATTRIBUTION.md](ATTRIBUTION.md) for full credits and licensing information.

**Original Author**: Jean Pierre Both
**Original Project**: [hnswlib-rs](https://github.com/jean-pierreBoth/hnswlib-rs)
**Original License**: Apache-2.0 OR MIT
**Fork License**: MIT (maintaining compatibility with upstream)

## Contributing

Contributions are welcome! Please note:
- Maintain compatibility with upstream where possible
- Preserve the VectorStorage abstraction layer
- Add tests for new functionality
- Follow Rust conventions

## Status

🚧 **Under Active Development**

This fork is being actively developed as part of the Tessera vector database project. The VectorStorage trait and external storage integration are in progress.

## License

This fork is licensed under the **MIT License** to maintain compatibility with the original dual-license project.

See [LICENSE-MIT](LICENSE-MIT) for details.

---

**Part of the [Tessera Vector Database](https://github.com/mojobytes/tessera) Project**
Forked: October 2025
Upstream: [hnswlib-rs](https://github.com/jean-pierreBoth/hnswlib-rs) v0.3.2
