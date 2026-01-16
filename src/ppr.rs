//! Personalized PageRank.

use crate::graph::Graph;
use crate::pagerank::PageRankConfig;

pub fn personalized_pagerank<G: Graph>(graph: &G, config: PageRankConfig, personalization: &[f64]) -> Vec<f64> {
    let n = graph.node_count();
    if n == 0 { return Vec::new(); }
    let p_sum: f64 = personalization.iter().sum();
    let p_vec: Vec<f64> = if p_sum > 0.0 { personalization.iter().map(|&x| x / p_sum).collect() } else { vec![1.0 / n as f64; n] };
    let mut scores = p_vec.clone();
    let mut new_scores = vec![0.0; n];
    let out_degrees: Vec<usize> = (0..n).map(|i| graph.out_degree(i)).collect();

    for _ in 0..config.max_iterations {
        let dangling_sum: f64 = out_degrees.iter().enumerate().filter(|(_, &deg)| deg == 0).map(|(i, _)| scores[i]).sum();
        for i in 0..n {
            new_scores[i] = (1.0 - config.damping) * p_vec[i] + config.damping * dangling_sum * p_vec[i];
        }
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
