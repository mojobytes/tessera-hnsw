# Attribution

This project is a fork of [hnswlib-rs](https://github.com/jean-pierreBoth/hnswlib-rs)
by Jean Pierre Both, originally licensed under dual Apache-2.0/MIT license.

## Original Project

- **Author**: Jean Pierre Both
- **Repository**: https://github.com/jean-pierreBoth/hnswlib-rs
- **License**: Apache-2.0 OR MIT
- **Description**: Rust implementation of HNSW (Hierarchical Navigable Small World)
  algorithm for approximate nearest neighbor search

## Fork Purpose and Modifications

This fork (`tessera-hnsw`) was created to integrate HNSW with the Tessera vector database
project, with the following key modifications:

### Architectural Changes

1. **External Vector Storage**: Added `VectorStorage` trait to support external vector
   management instead of embedding vectors in the HNSW structure

2. **Separated Persistence**: Modified serialization to persist only graph topology,
   allowing vectors to be stored separately in Tessera's custom storage format

3. **Tessera Integration**: Designed to work seamlessly with `tessera-storage` crate
   for unified vector management across multiple index types

### Rationale

The original `hnsw_rs` crate stores vectors within the HNSW structure, which would
require duplicating vectors for each index algorithm (HNSW, IVF, PQ, etc.) in a
multi-index vector database. This fork enables:

- Single source of truth for vector storage (`.tsvf` files)
- Multiple index types sharing the same vectors
- Reduced storage overhead
- Consistent vector access patterns across index types

## Upstream Synchronization

This fork maintains compatibility with the original API where possible. We track
upstream changes and selectively merge relevant improvements while maintaining
our custom storage architecture.

## License

This fork is licensed under **MIT License** (same as one of the original licenses),
to maintain maximum compatibility and honor the original author's licensing choice.

See [LICENSE-MIT](LICENSE-MIT) for the full license text.

## Credits

We are deeply grateful to Jean Pierre Both for creating and maintaining `hnsw_rs`,
which provided an excellent foundation for this work. The core HNSW algorithm
implementation, graph construction, and search logic are largely unchanged from
the original.

## Contributions

Contributions to `tessera-hnsw` are welcome. Please note that modifications should
maintain the separation between graph topology and vector storage, which is the
core architectural difference from upstream.

---

**Tessera HNSW Fork**
Part of the [Tessera Vector Database](https://github.com/mojobytes/tessera) project
Forked: October 2025
Original project: https://github.com/jean-pierreBoth/hnswlib-rs
