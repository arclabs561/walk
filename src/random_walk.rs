//! Random walk generation.

use crate::graph::Graph;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

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

pub fn generate_walks<G: Graph>(graph: &G, config: WalkConfig) -> Vec<Vec<usize>> {
    let mut walks = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    for node in 0..graph.node_count() {
        for _ in 0..config.walks_per_node {
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

pub fn generate_biased_walks<G: Graph>(graph: &G, config: WalkConfig) -> Vec<Vec<usize>> {
    let mut walks = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    for node in 0..graph.node_count() {
        for _ in 0..config.walks_per_node {
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
    let mut prev_neighbors: HashSet<usize> = HashSet::new();

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
        prev_neighbors.clear();
        prev_neighbors.extend(graph.neighbors(curr));
        curr = next;
    }
    walk
}

fn sample_biased_rejection<R: Rng>(rng: &mut R, prev_node: usize, prev_neighbors: &HashSet<usize>, neighbors: &[usize], p: f32, q: f32) -> usize {
    let max_prob = (1.0 / p).max(1.0).max(1.0 / q);
    loop {
        let candidate = *neighbors.choose(rng).unwrap();
        let r: f32 = rng.random();
        let unnorm_prob = if candidate == prev_node { 1.0 / p } else if prev_neighbors.contains(&candidate) { 1.0 } else { 1.0 / q };
        if r < unnorm_prob / max_prob { return candidate; }
    }
}
