//! # walk
//!
//! Graph random-walk family primitives for the representational stack.

pub mod graph;
pub mod pagerank;
pub mod ppr;
pub mod random_walk;
pub mod topk;

pub use graph::{Graph, WeightedGraph, AdjacencyMatrix};
pub use pagerank::{pagerank, PageRankConfig};
pub use ppr::personalized_pagerank;
pub use random_walk::{generate_walks, generate_biased_walks, WalkConfig};
pub use topk::{top_k, normalize};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, Error>;
