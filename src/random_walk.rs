//! Random walk generation.

use crate::graph::{Graph, GraphRef};
use kuji::reservoir::ReservoirSampler;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WalkConfig {
    pub length: usize,
    pub walks_per_node: usize,
    pub p: f32,
    pub q: f32,
    pub seed: u64,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self { length: 80, walks_per_node: 10, p: 1.0, q: 1.0, seed: 42 }
    }
}

/// Deterministically sample up to `k` start nodes from `0..node_count`.
///
/// This is a practical “bridge” between `walk` and the lower-level sampling crate `kuji`:
/// when `node_count` is huge, materializing all nodes just to choose a subset can be wasteful.
///
/// We use reservoir sampling (uniform sample without replacement) to keep memory bounded.
pub fn sample_start_nodes_reservoir(node_count: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut sampler = ReservoirSampler::new(k);
    for i in 0..node_count {
        sampler.add_with_rng(i, &mut rng);
    }
    sampler.samples().to_vec()
}

pub fn generate_walks<G: Graph>(graph: &G, config: WalkConfig) -> Vec<Vec<usize>> {
    let start_nodes: Vec<usize> = (0..graph.node_count()).collect();
    generate_walks_from_nodes(graph, &start_nodes, config)
}

/// Random walk generation (unbiased), but restricted to an explicit set of start nodes.
///
/// Determinism contract:
/// - For a fixed `config` and identical `start_nodes` content, output is deterministic.
/// - We still shuffle start nodes per epoch to avoid systematic ordering bias.
pub fn generate_walks_from_nodes<G: Graph>(
    graph: &G,
    start_nodes: &[usize],
    config: WalkConfig,
) -> Vec<Vec<usize>> {
    let mut walks = Vec::with_capacity(start_nodes.len() * config.walks_per_node);
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut epoch_nodes: Vec<usize> = start_nodes.to_vec();

    for _ in 0..config.walks_per_node {
        // Shuffle start nodes to avoid systematic ordering bias.
        epoch_nodes.shuffle(&mut rng);
        for &node in &epoch_nodes {
            walks.push(unbiased_walk(graph, node, config.length, &mut rng));
        }
    }
    walks
}

fn unbiased_walk<G: Graph, R: Rng>(graph: &G, start: usize, length: usize, rng: &mut R) -> Vec<usize> {
    let mut walk = Vec::with_capacity(length);
    walk.push(start);
    let mut curr = start;
    for _ in 1..length {
        let neighbors = graph.neighbors(curr);
        if neighbors.is_empty() { break; }
        curr = *neighbors.choose(rng).unwrap();
        walk.push(curr);
    }
    walk
}

/// Random walk generation for graphs that can return borrowed neighbor slices.
///
/// This avoids the per-step `Vec` allocation implicit in [`Graph::neighbors`].
pub fn generate_walks_ref<G: GraphRef>(graph: &G, config: WalkConfig) -> Vec<Vec<usize>> {
    let start_nodes: Vec<usize> = (0..graph.node_count()).collect();
    generate_walks_ref_from_nodes(graph, &start_nodes, config)
}

/// Random walk generation (unbiased), but restricted to an explicit set of start nodes.
///
/// This is useful for “delta walk” updates in dynamic embeddings (dynnode2vec-style),
/// and for sharding work across machines (partition `start_nodes`).
///
/// Determinism contract:
/// - For a fixed `config` and identical `start_nodes` content, output is deterministic.
/// - We still shuffle start nodes per epoch to avoid systematic ordering bias.
pub fn generate_walks_ref_from_nodes<G: GraphRef>(
    graph: &G,
    start_nodes: &[usize],
    config: WalkConfig,
) -> Vec<Vec<usize>> {
    let mut walks = Vec::with_capacity(start_nodes.len() * config.walks_per_node);
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut epoch_nodes: Vec<usize> = start_nodes.to_vec();

    for _ in 0..config.walks_per_node {
        epoch_nodes.shuffle(&mut rng);
        for &node in &epoch_nodes {
            walks.push(unbiased_walk_ref(graph, node, config.length, &mut rng));
        }
    }
    walks
}

fn unbiased_walk_ref<G: GraphRef, R: Rng>(graph: &G, start: usize, length: usize, rng: &mut R) -> Vec<usize> {
    let mut walk = Vec::with_capacity(length);
    walk.push(start);
    let mut curr = start;
    for _ in 1..length {
        let neighbors = graph.neighbors_ref(curr);
        if neighbors.is_empty() {
            break;
        }
        curr = *neighbors.choose(rng).unwrap();
        walk.push(curr);
    }
    walk
}

/// Streaming unbiased random walk generation (borrowed neighbor slices).
///
/// Motivation (practical): for large graphs, materializing **all walks** in memory can dominate
/// runtime and memory. This API emits each walk to a caller-provided callback.
///
/// Determinism contract:
/// - For fixed `config` and identical `start_nodes` content, the emitted walk sequence is stable.
pub fn generate_walks_ref_streaming_from_nodes<G, F>(
    graph: &G,
    start_nodes: &[usize],
    config: WalkConfig,
    mut on_walk: F,
)
where
    G: GraphRef,
    F: FnMut(&[usize]),
{
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut epoch_nodes: Vec<usize> = start_nodes.to_vec();
    let mut buf: Vec<usize> = Vec::with_capacity(config.length);

    for _ in 0..config.walks_per_node {
        epoch_nodes.shuffle(&mut rng);
        for &node in &epoch_nodes {
            buf.clear();
            unbiased_walk_ref_into(graph, node, config.length, &mut rng, &mut buf);
            on_walk(&buf);
        }
    }
}

fn unbiased_walk_ref_into<G: GraphRef, R: Rng>(
    graph: &G,
    start: usize,
    length: usize,
    rng: &mut R,
    out: &mut Vec<usize>,
) {
    out.reserve(length.saturating_sub(out.capacity()));
    out.push(start);
    let mut curr = start;
    for _ in 1..length {
        let neighbors = graph.neighbors_ref(curr);
        if neighbors.is_empty() {
            break;
        }
        curr = *neighbors.choose(rng).unwrap();
        out.push(curr);
    }
}

pub fn generate_biased_walks<G: Graph>(graph: &G, config: WalkConfig) -> Vec<Vec<usize>> {
    let start_nodes: Vec<usize> = (0..graph.node_count()).collect();
    generate_biased_walks_from_nodes(graph, &start_nodes, config)
}

/// Node2Vec-style biased walk generation, restricted to an explicit set of start nodes.
pub fn generate_biased_walks_from_nodes<G: Graph>(
    graph: &G,
    start_nodes: &[usize],
    config: WalkConfig,
) -> Vec<Vec<usize>> {
    let mut walks = Vec::with_capacity(start_nodes.len() * config.walks_per_node);
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut epoch_nodes: Vec<usize> = start_nodes.to_vec();

    for _ in 0..config.walks_per_node {
        epoch_nodes.shuffle(&mut rng);
        for &node in &epoch_nodes {
            walks.push(biased_walk(graph, node, config, &mut rng));
        }
    }
    walks
}

fn biased_walk<G: Graph, R: Rng>(graph: &G, start: usize, config: WalkConfig, rng: &mut R) -> Vec<usize> {
    let mut walk = Vec::with_capacity(config.length);
    walk.push(start);
    let mut curr = start;
    let mut prev: Option<usize> = None;
    let mut prev_neighbors: Vec<usize> = Vec::new();

    for _ in 1..config.length {
        let neighbors = graph.neighbors(curr);
        if neighbors.is_empty() { break; }
        let next = if let Some(p_node) = prev {
            sample_biased_rejection(rng, p_node, &prev_neighbors, &neighbors, config.p, config.q)
        } else {
            *neighbors.choose(rng).unwrap()
        };
        walk.push(next);
        prev = Some(curr);
        prev_neighbors = neighbors;
        curr = next;
    }
    walk
}

/// Node2Vec-style biased walk generation for graphs that can return borrowed neighbor slices.
pub fn generate_biased_walks_ref<G: GraphRef>(graph: &G, config: WalkConfig) -> Vec<Vec<usize>> {
    let start_nodes: Vec<usize> = (0..graph.node_count()).collect();
    generate_biased_walks_ref_from_nodes(graph, &start_nodes, config)
}

/// Node2Vec-style biased walk generation, restricted to an explicit set of start nodes.
pub fn generate_biased_walks_ref_from_nodes<G: GraphRef>(
    graph: &G,
    start_nodes: &[usize],
    config: WalkConfig,
) -> Vec<Vec<usize>> {
    let mut walks = Vec::with_capacity(start_nodes.len() * config.walks_per_node);
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut epoch_nodes: Vec<usize> = start_nodes.to_vec();

    for _ in 0..config.walks_per_node {
        epoch_nodes.shuffle(&mut rng);
        for &node in &epoch_nodes {
            walks.push(biased_walk_ref(graph, node, config, &mut rng));
        }
    }
    walks
}

fn biased_walk_ref<G: GraphRef, R: Rng>(graph: &G, start: usize, config: WalkConfig, rng: &mut R) -> Vec<usize> {
    let mut walk = Vec::with_capacity(config.length);
    walk.push(start);

    let mut curr = start;
    let mut prev: Option<usize> = None;
    let mut prev_neighbors: &[usize] = &[];

    for _ in 1..config.length {
        let neighbors = graph.neighbors_ref(curr);
        if neighbors.is_empty() {
            break;
        }

        let next = if let Some(p_node) = prev {
            sample_biased_rejection(rng, p_node, prev_neighbors, neighbors, config.p, config.q)
        } else {
            *neighbors.choose(rng).unwrap()
        };

        walk.push(next);

        // Cache neighbors(curr) for the next step as "prev_neighbors".
        prev = Some(curr);
        prev_neighbors = neighbors;

        curr = next;
    }
    walk
}

/// Streaming node2vec-style biased walk generation (borrowed neighbor slices).
///
/// Same motivation as `generate_walks_ref_streaming_from_nodes`: avoid materializing all walks.
pub fn generate_biased_walks_ref_streaming_from_nodes<G, F>(
    graph: &G,
    start_nodes: &[usize],
    config: WalkConfig,
    mut on_walk: F,
)
where
    G: GraphRef,
    F: FnMut(&[usize]),
{
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut epoch_nodes: Vec<usize> = start_nodes.to_vec();
    let mut buf: Vec<usize> = Vec::with_capacity(config.length);

    for _ in 0..config.walks_per_node {
        epoch_nodes.shuffle(&mut rng);
        for &node in &epoch_nodes {
            buf.clear();
            biased_walk_ref_into(graph, node, config, &mut rng, &mut buf);
            on_walk(&buf);
        }
    }
}

fn biased_walk_ref_into<G: GraphRef, R: Rng>(
    graph: &G,
    start: usize,
    config: WalkConfig,
    rng: &mut R,
    out: &mut Vec<usize>,
) {
    out.reserve(config.length.saturating_sub(out.capacity()));
    out.push(start);

    let mut curr = start;
    let mut prev: Option<usize> = None;
    let mut prev_neighbors: &[usize] = &[];

    for _ in 1..config.length {
        let neighbors = graph.neighbors_ref(curr);
        if neighbors.is_empty() {
            break;
        }

        let next = if let Some(p_node) = prev {
            sample_biased_rejection(rng, p_node, prev_neighbors, neighbors, config.p, config.q)
        } else {
            *neighbors.choose(rng).unwrap()
        };

        out.push(next);

        prev = Some(curr);
        prev_neighbors = neighbors;
        curr = next;
    }
}

fn sample_biased_rejection<R: Rng>(rng: &mut R, prev_node: usize, prev_neighbors: &[usize], neighbors: &[usize], p: f32, q: f32) -> usize {
    let max_prob = (1.0 / p).max(1.0).max(1.0 / q);
    loop {
        let candidate = *neighbors.choose(rng).unwrap();
        let r: f32 = rng.random();
        let is_in_edge = prev_neighbors.iter().any(|&x| x == candidate);
        let unnorm_prob = if candidate == prev_node { 1.0 / p } else if is_in_edge { 1.0 } else { 1.0 / q };
        if r < unnorm_prob / max_prob { return candidate; }
    }
}

#[cfg(feature = "parallel")]
fn mix64(mut x: u64) -> u64 {
    // SplitMix64 finalizer (stable, good diffusion).
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

/// Deterministic parallel unbiased walk generation.
///
/// Invariant: output is stable for a fixed `seed`, independent of Rayon thread count.
#[cfg(feature = "parallel")]
pub fn generate_walks_ref_parallel<G: GraphRef + Sync>(graph: &G, config: WalkConfig) -> Vec<Vec<usize>> {
    let n = graph.node_count();
    let start_nodes: Vec<usize> = (0..n).collect();
    generate_walks_ref_parallel_from_nodes(graph, &start_nodes, config)
}

/// Deterministic parallel unbiased walk generation, restricted to an explicit set of start nodes.
///
/// Invariant: output is stable for a fixed `seed`, independent of Rayon thread count.
#[cfg(feature = "parallel")]
pub fn generate_walks_ref_parallel_from_nodes<G: GraphRef + Sync>(
    graph: &G,
    start_nodes: &[usize],
    config: WalkConfig,
) -> Vec<Vec<usize>> {
    use rayon::prelude::*;

    let mut epoch_nodes: Vec<usize> = start_nodes.to_vec();
    let mut jobs: Vec<(u32, usize)> = Vec::with_capacity(start_nodes.len() * config.walks_per_node);

    for epoch in 0..(config.walks_per_node as u32) {
        let mut rng = ChaCha8Rng::seed_from_u64(mix64(config.seed ^ (epoch as u64)));
        epoch_nodes.shuffle(&mut rng);
        for &node in &epoch_nodes {
            jobs.push((epoch, node));
        }
    }

    jobs.par_iter()
        .enumerate()
        .map(|(i, (epoch, node))| {
            let seed = mix64(config.seed ^ ((*epoch as u64) << 32) ^ (*node as u64) ^ (i as u64));
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            unbiased_walk_ref(graph, *node, config.length, &mut rng)
        })
        .collect()
}

/// Deterministic parallel node2vec-style biased walk generation.
///
/// Invariant: output is stable for a fixed `seed`, independent of Rayon thread count.
#[cfg(feature = "parallel")]
pub fn generate_biased_walks_ref_parallel<G: GraphRef + Sync>(graph: &G, config: WalkConfig) -> Vec<Vec<usize>> {
    let n = graph.node_count();
    let start_nodes: Vec<usize> = (0..n).collect();
    generate_biased_walks_ref_parallel_from_nodes(graph, &start_nodes, config)
}

/// Deterministic parallel biased walk generation, restricted to an explicit set of start nodes.
///
/// Invariant: output is stable for a fixed `seed`, independent of Rayon thread count.
#[cfg(feature = "parallel")]
pub fn generate_biased_walks_ref_parallel_from_nodes<G: GraphRef + Sync>(
    graph: &G,
    start_nodes: &[usize],
    config: WalkConfig,
) -> Vec<Vec<usize>> {
    use rayon::prelude::*;

    let mut epoch_nodes: Vec<usize> = start_nodes.to_vec();
    let mut jobs: Vec<(u32, usize)> = Vec::with_capacity(start_nodes.len() * config.walks_per_node);

    for epoch in 0..(config.walks_per_node as u32) {
        let mut rng = ChaCha8Rng::seed_from_u64(mix64(config.seed ^ (epoch as u64)));
        epoch_nodes.shuffle(&mut rng);
        for &node in &epoch_nodes {
            jobs.push((epoch, node));
        }
    }

    jobs.par_iter()
        .enumerate()
        .map(|(i, (epoch, node))| {
            let seed =
                mix64(config.seed ^ ((*epoch as u64) << 32) ^ (*node as u64) ^ (i as u64));
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            biased_walk_ref(graph, *node, config, &mut rng)
        })
        .collect()
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct RefAdj {
        adj: Vec<Vec<usize>>,
    }

    impl GraphRef for RefAdj {
        fn node_count(&self) -> usize {
            self.adj.len()
        }

        fn neighbors_ref(&self, node: usize) -> &[usize] {
            self.adj.get(node).map(Vec::as_slice).unwrap_or(&[])
        }
    }

    #[test]
    fn streaming_matches_collect_unbiased() {
        // A small graph with an isolate and a degree-1 node.
        let g = RefAdj {
            adj: vec![
                vec![1],      // 0
                vec![0, 2],   // 1
                vec![1],      // 2
                vec![],       // 3 isolate
            ],
        };

        let config = WalkConfig { length: 8, walks_per_node: 3, seed: 123, p: 1.0, q: 1.0 };
        let start_nodes = [0usize, 1, 2, 3];

        let collected = generate_walks_ref_from_nodes(&g, &start_nodes, config);
        let mut streamed: Vec<Vec<usize>> = Vec::new();
        generate_walks_ref_streaming_from_nodes(&g, &start_nodes, config, |w| {
            streamed.push(w.to_vec());
        });

        assert_eq!(streamed, collected);
    }

    #[test]
    fn streaming_matches_collect_biased() {
        let g = RefAdj {
            adj: vec![
                vec![1],      // 0
                vec![0, 2],   // 1
                vec![1],      // 2
                vec![],       // 3 isolate
            ],
        };

        let config = WalkConfig { length: 8, walks_per_node: 3, seed: 999, p: 0.5, q: 2.0 };
        let start_nodes = [0usize, 1, 2, 3];

        let collected = generate_biased_walks_ref_from_nodes(&g, &start_nodes, config);
        let mut streamed: Vec<Vec<usize>> = Vec::new();
        generate_biased_walks_ref_streaming_from_nodes(&g, &start_nodes, config, |w| {
            streamed.push(w.to_vec());
        });

        assert_eq!(streamed, collected);
    }
}
