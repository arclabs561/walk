//! PageRank centrality.

use crate::graph::{Graph, WeightedGraph};

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

/// Weighted PageRank centrality.
///
/// Edges are treated as having non-negative weights, and a node's outgoing mass is split
/// proportionally to outgoing edge weights.
///
/// This matches the Markov chain transition:
/// \[
///   P(u \to v) = \frac{w(u,v)}{\sum_x w(u,x)}
/// \]
pub fn pagerank_weighted<G: WeightedGraph>(graph: &G, config: PageRankConfig) -> Vec<f64> {
    let n = graph.node_count();
    if n == 0 { return Vec::new(); }

    let n_f64 = n as f64;
    let mut scores = vec![1.0 / n_f64; n];
    let mut new_scores = vec![0.0; n];

    // Precompute outgoing neighbors once (Graph) and outgoing weight sums (WeightedGraph).
    let neighbors: Vec<Vec<usize>> = (0..n).map(|u| graph.neighbors(u)).collect();
    let out_wsum: Vec<f64> = (0..n)
        .map(|u| neighbors[u].iter().map(|&v| graph.edge_weight(u, v).max(0.0)).sum())
        .collect();

    for _ in 0..config.max_iterations {
        let dangling_sum: f64 = out_wsum
            .iter()
            .enumerate()
            .filter(|(_, &ws)| ws == 0.0)
            .map(|(i, _)| scores[i])
            .sum();

        let dangling_contrib = config.damping * dangling_sum / n_f64;
        let teleport = (1.0 - config.damping) / n_f64;
        new_scores.fill(teleport + dangling_contrib);

        for u in 0..n {
            let ws = out_wsum[u];
            if ws > 0.0 {
                // distribute along outgoing edges proportional to weight
                for &v in &neighbors[u] {
                    let w = graph.edge_weight(u, v).max(0.0);
                    if w > 0.0 {
                        new_scores[v] += config.damping * scores[u] * (w / ws);
                    }
                }
            }
        }

        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();
        std::mem::swap(&mut scores, &mut new_scores);
        if diff < config.tolerance { break; }
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AdjacencyMatrix;

    #[test]
    fn test_pagerank_weighted_sums_to_one() {
        // 3 nodes, weighted edges:
        // 0 -> 1 (2.0), 0 -> 2 (1.0)
        // 1 -> 2 (1.0)
        // 2 -> (dangling)
        let adj = vec![
            vec![0.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ];
        let g = AdjacencyMatrix(&adj);
        let scores = pagerank_weighted(&g, PageRankConfig::default());
        let total: f64 = scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "sum={total}");
    }

    #[test]
    fn test_pagerank_weight_biases_toward_heavier_edge() {
        // 0 links to 1 twice as strongly as to 2, so 1 should rank >= 2.
        let adj = vec![
            vec![0.0, 2.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        let g = AdjacencyMatrix(&adj);
        let scores = pagerank_weighted(&g, PageRankConfig::default());
        assert!(scores[1] >= scores[2], "scores[1]={} scores[2]={}", scores[1], scores[2]);
    }
}
