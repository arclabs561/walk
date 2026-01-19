//! Betweenness centrality.
//!
//! This module is feature-gated behind `petgraph` because the most practical caller in this
//! workspace is a `petgraph::Graph`/`DiGraph` (e.g. dependency graphs, module graphs).
//!
//! Public invariant:
//! - The output vector is indexed by `NodeIndex::index()` (stable ordering).
//! - Disconnected graphs are allowed; unreachable pairs contribute 0.
//!
//! Notes:
//! - This is Brandes' algorithm for **directed, unweighted** graphs.
//! - Normalization uses the directed convention \(1/((n-1)(n-2))\) for \(n \ge 3\).

use petgraph::prelude::*;

/// Betweenness centrality (Brandes) for directed, unweighted graphs.
///
/// Returns one score per `NodeIndex`, ordered by index.
pub fn betweenness_centrality<N, E, Ix>(graph: &petgraph::Graph<N, E, Directed, Ix>) -> Vec<f64>
where
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();
    if n <= 2 {
        return vec![0.0; n];
    }

    let mut betweenness = vec![0.0; n];

    for s in graph.node_indices() {
        let mut stack: Vec<NodeIndex<Ix>> = Vec::new();
        let mut pred: Vec<Vec<NodeIndex<Ix>>> = vec![vec![]; n];
        let mut sigma = vec![0.0f64; n];
        let mut dist: Vec<i32> = vec![-1; n];

        sigma[s.index()] = 1.0;
        dist[s.index()] = 0;

        let mut queue: std::collections::VecDeque<NodeIndex<Ix>> = std::collections::VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for w in graph.neighbors_directed(v, Direction::Outgoing) {
                if dist[w.index()] < 0 {
                    dist[w.index()] = dist[v.index()] + 1;
                    queue.push_back(w);
                }
                if dist[w.index()] == dist[v.index()] + 1 {
                    sigma[w.index()] += sigma[v.index()];
                    pred[w.index()].push(v);
                }
            }
        }

        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w.index()] {
                // sigma[w] can be 0 for disconnected nodes; guard division.
                let sigma_w = sigma[w.index()];
                if sigma_w > 0.0 {
                    delta[v.index()] += (sigma[v.index()] / sigma_w) * (1.0 + delta[w.index()]);
                }
            }
            if w != s {
                betweenness[w.index()] += delta[w.index()];
            }
        }
    }

    // Directed normalization to [0,1] for connected-ish graphs.
    let norm = 1.0 / ((n - 1) * (n - 2)) as f64;
    for b in &mut betweenness {
        *b *= norm;
    }
    betweenness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_graph_middle_is_highest() {
        // 0 -> 1 -> 2 -> 3
        let mut g: DiGraph<(), ()> = DiGraph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        let d = g.add_node(());
        g.add_edge(a, b, ());
        g.add_edge(b, c, ());
        g.add_edge(c, d, ());

        let bc = betweenness_centrality(&g);
        // endpoints should be 0; middle nodes should be > 0.
        assert_eq!(bc[a.index()], 0.0);
        assert_eq!(bc[d.index()], 0.0);
        assert!(bc[b.index()] > 0.0, "b={}", bc[b.index()]);
        assert!(bc[c.index()] > 0.0, "c={}", bc[c.index()]);
    }
}

