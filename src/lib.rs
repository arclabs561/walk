//! # walk
//!
//! Graph random-walk family primitives: PageRank / personalized PageRank, unbiased random walks,
//! and biased 2nd-order walks (Node2Vec / Node2Vec+).
//!
//! ## Design contract
//!
//! - **Algorithmic clarity > cleverness**: implementations are meant to be readable and testable.
//! - **Determinism is a first-class option**: when a caller supplies an RNG/seed, results should
//!   be reproducible. (Benchmarks and statistical smoke tests should avoid flakiness.)
//! - **No backend lock-in**: this crate is about walks and ranks, not tensors.
//!
//! ## References (what motivated the implementations/tests)
//!
//! - Page et al. (1999): PageRank.
//! - Grover & Leskovec (2016): Node2Vec (biased second-order random walks).
//! - Liu, Hirn, Krishnan (2023): Node2Vec+ for weighted networks (*Accurately modeling biased
//!   random walks on weighted networks using node2vec+*, Bioinformatics 39(1): btad047).
//!   The `WeightedNode2VecPlusConfig` path is intended to keep the interface explicit about the
//!   additional terms.
//! - Alias sampling method: Walker (1974) / Vose (1991) style alias tables for O(1) categorical draws.

pub mod graph;
#[cfg(feature = "petgraph")]
pub mod betweenness;
pub mod node2vec;
pub mod pagerank;
pub mod ppr;
pub mod random_walk;
pub mod reachability;
pub mod topk;

pub use graph::{AdjacencyMatrix, Graph, GraphRef, WeightedGraph, WeightedGraphRef};
#[cfg(feature = "petgraph")]
pub use betweenness::betweenness_centrality;
pub use node2vec::{
    generate_biased_walks_precomp_ref,
    generate_biased_walks_precomp_ref_from_nodes,
    generate_biased_walks_weighted_ref,
    generate_biased_walks_weighted_plus_ref,
    PrecomputedBiasedWalks,
    WeightedNode2VecPlusConfig,
};

#[cfg(feature = "parallel")]
pub use node2vec::generate_biased_walks_precomp_ref_parallel_from_nodes;
pub use pagerank::{pagerank, pagerank_weighted, PageRankConfig};
pub use ppr::personalized_pagerank;
pub use reachability::reachability_counts_edges;
pub use random_walk::{
    generate_biased_walks,
    generate_biased_walks_from_nodes,
    generate_biased_walks_ref,
    generate_biased_walks_ref_from_nodes,
    generate_biased_walks_ref_streaming_from_nodes,
    generate_walks,
    generate_walks_from_nodes,
    generate_walks_ref,
    generate_walks_ref_from_nodes,
    generate_walks_ref_streaming_from_nodes,
    WalkConfig,
};

#[cfg(feature = "parallel")]
pub use random_walk::{
    generate_biased_walks_ref_parallel,
    generate_biased_walks_ref_parallel_from_nodes,
    generate_walks_ref_parallel,
    generate_walks_ref_parallel_from_nodes,
};
pub use topk::{top_k, normalize};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, Error>;
