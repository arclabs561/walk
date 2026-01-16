//! PageRank centrality.

use crate::graph::Graph;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PageRankConfig {
    pub damping: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self { damping: 0.85, max_iterations: 100, tolerance: 1e-6 }
    }
}

pub fn pagerank<G: Graph>(graph: &G, config: PageRankConfig) -> Vec<f64> {
    let n = graph.node_count();
    if n == 0 { return Vec::new(); }
    let n_f64 = n as f64;
    let mut scores = vec![1.0 / n_f64; n];
    let mut new_scores = vec![0.0; n];
    let out_degrees: Vec<usize> = (0..n).map(|i| graph.out_degree(i)).collect();

    for _ in 0..config.max_iterations {
        let dangling_sum: f64 = out_degrees.iter().enumerate().filter(|(_, &deg)| deg == 0).map(|(i, _)| scores[i]).sum();
        let dangling_contrib = config.damping * dangling_sum / n_f64;
        let teleport = (1.0 - config.damping) / n_f64;
        new_scores.fill(teleport + dangling_contrib);

        for u in 0..n {
            let deg = out_degrees[u];
            if deg > 0 {
                let share = config.damping * scores[u] / deg as f64;
                for v in graph.neighbors(u) {
                    new_scores[v] += share;
                }
            }
        }

        let diff: f64 = scores.iter().zip(new_scores.iter()).map(|(old, new)| (old - new).abs()).sum();
        std::mem::swap(&mut scores, &mut new_scores);
        if diff < config.tolerance { break; }
    }
    scores
}
