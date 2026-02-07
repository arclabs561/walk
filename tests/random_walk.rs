use proptest::prelude::*;
use walk::{
    generate_biased_walks, generate_biased_walks_precomp_ref,
    generate_biased_walks_precomp_ref_from_nodes, generate_biased_walks_ref,
    generate_biased_walks_weighted_plus_ref, generate_biased_walks_weighted_ref, generate_walks,
    generate_walks_ref, normalize, pagerank, personalized_pagerank, top_k, PageRankConfig,
    PrecomputedBiasedWalks, WalkConfig, WeightedGraphRef, WeightedNode2VecPlusConfig,
};
use walk::{Graph, GraphRef};

#[derive(Debug, Clone)]
struct AdjListGraph {
    adj: Vec<Vec<usize>>,
}

impl AdjListGraph {
    fn new(mut adj: Vec<Vec<usize>>) -> Self {
        for nbrs in &mut adj {
            nbrs.sort_unstable();
        }
        Self { adj }
    }
}

impl Graph for AdjListGraph {
    fn node_count(&self) -> usize {
        self.adj.len()
    }

    fn neighbors(&self, node: usize) -> Vec<usize> {
        self.adj.get(node).cloned().unwrap_or_default()
    }
}

impl GraphRef for AdjListGraph {
    fn node_count(&self) -> usize {
        self.adj.len()
    }

    fn neighbors_ref(&self, node: usize) -> &[usize] {
        self.adj.get(node).map(Vec::as_slice).unwrap_or(&[])
    }
}

#[derive(Debug, Clone)]
struct WeightedAdjListGraph {
    adj: Vec<Vec<usize>>,
    wts: Vec<Vec<f32>>,
}

impl WeightedAdjListGraph {
    fn new(mut adj: Vec<Vec<usize>>, mut wts: Vec<Vec<f32>>) -> Self {
        assert_eq!(adj.len(), wts.len());
        for i in 0..adj.len() {
            assert_eq!(adj[i].len(), wts[i].len());

            // Keep neighbor/weight pairs aligned while sorting by neighbor id.
            let mut pairs: Vec<(usize, f32)> =
                adj[i].iter().copied().zip(wts[i].iter().copied()).collect();
            pairs.sort_by_key(|(n, _)| *n);
            adj[i] = pairs.iter().map(|(n, _)| *n).collect();
            wts[i] = pairs.iter().map(|(_, w)| *w).collect();
        }
        Self { adj, wts }
    }
}

impl WeightedGraphRef for WeightedAdjListGraph {
    fn node_count(&self) -> usize {
        self.adj.len()
    }

    fn neighbors_and_weights_ref(&self, node: usize) -> (&[usize], &[f32]) {
        let nbrs = self.adj.get(node).map(Vec::as_slice).unwrap_or(&[]);
        let wts = self.wts.get(node).map(Vec::as_slice).unwrap_or(&[]);
        (nbrs, wts)
    }
}

fn assert_walks_sane(walks: &[Vec<usize>], n: usize, max_len: usize) {
    for w in walks {
        assert!(!w.is_empty(), "walk should never be empty");
        assert!(w.len() <= max_len, "walk length exceeded config");
        for &v in w {
            assert!(v < n, "walk node index out of range: {v} >= {n}");
        }
    }
}

fn assert_walks_follow_edges_ref(g: &AdjListGraph, walks: &[Vec<usize>]) {
    for w in walks {
        for win in w.windows(2) {
            let u = win[0];
            let v = win[1];
            let nbrs = g.neighbors_ref(u);
            assert!(
                nbrs.binary_search(&v).is_ok(),
                "walk step {u} -> {v} is not an edge"
            );
        }
    }
}

#[test]
fn unbiased_ref_matches_vec_api() {
    // A small undirected graph:
    // 0--1--2
    //    \  |
    //     \ |
    //       3
    let g = AdjListGraph::new(vec![
        vec![1],       // 0
        vec![0, 2, 3], // 1
        vec![1, 3],    // 2
        vec![1, 2],    // 3
    ]);

    let cfg = WalkConfig {
        length: 8,
        walks_per_node: 3,
        p: 1.0,
        q: 1.0,
        seed: 42,
    };

    let a = generate_walks(&g, cfg);
    let b = generate_walks_ref(&g, cfg);

    assert_eq!(a, b, "Graph vs GraphRef paths should match");
    assert_walks_sane(&a, Graph::node_count(&g), cfg.length);
    assert_walks_follow_edges_ref(&g, &a);
}

#[test]
fn biased_ref_matches_vec_api() {
    let g = AdjListGraph::new(vec![
        vec![1],       // 0
        vec![0, 2, 3], // 1
        vec![1, 3],    // 2
        vec![1, 2],    // 3
    ]);

    let cfg = WalkConfig {
        length: 8,
        walks_per_node: 3,
        p: 0.5,
        q: 2.0,
        seed: 42,
    };

    let a = generate_biased_walks(&g, cfg);
    let b = generate_biased_walks_ref(&g, cfg);

    assert_eq!(a, b, "Graph vs GraphRef paths should match");
    assert_walks_sane(&a, Graph::node_count(&g), cfg.length);
    assert_walks_follow_edges_ref(&g, &a);
}

#[test]
fn reproducible_given_seed() {
    let g = AdjListGraph::new(vec![
        vec![1],       // 0
        vec![0, 2, 3], // 1
        vec![1, 3],    // 2
        vec![1, 2],    // 3
    ]);

    let cfg = WalkConfig {
        length: 6,
        walks_per_node: 2,
        p: 0.5,
        q: 2.0,
        seed: 123,
    };

    let w1 = generate_biased_walks_ref(&g, cfg);
    let w2 = generate_biased_walks_ref(&g, cfg);
    assert_eq!(w1, w2, "same seed should yield identical walks");
    assert_walks_follow_edges_ref(&g, &w1);
}

#[test]
fn isolated_node_walks_have_length_1() {
    let g = AdjListGraph::new(vec![vec![]]);
    let cfg = WalkConfig {
        length: 10,
        walks_per_node: 3,
        p: 0.5,
        q: 2.0,
        seed: 7,
    };

    let u = generate_walks(&g, cfg);
    let ur = generate_walks_ref(&g, cfg);
    assert_eq!(u, ur);
    assert_eq!(u.len(), 3);
    assert!(u.iter().all(|w| w.as_slice() == [0]));

    let b = generate_biased_walks(&g, cfg);
    let br = generate_biased_walks_ref(&g, cfg);
    assert_eq!(b, br);
    assert_eq!(b.len(), 3);
    assert!(b.iter().all(|w| w.as_slice() == [0]));
}

#[test]
fn topk_and_normalize_basic() {
    let scores = [0.0, 2.0, f64::NAN, 1.0, f64::INFINITY, -1.0];
    let got = top_k(&scores, 2);
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].0, 1);
    assert_eq!(got[0].1, 2.0);
    assert_eq!(got[1].0, 3);
    assert_eq!(got[1].1, 1.0);

    let mut v = vec![1.0, 1.0, 2.0];
    normalize(&mut v);
    let s: f64 = v.iter().sum();
    assert!((s - 1.0).abs() < 1e-12);
    assert!((v[0] - 0.25).abs() < 1e-12);
    assert!((v[1] - 0.25).abs() < 1e-12);
    assert!((v[2] - 0.5).abs() < 1e-12);
}

#[test]
fn pagerank_cycle_is_uniform() {
    // 0 -> 1 -> 2 -> 0
    let g = AdjListGraph::new(vec![vec![1], vec![2], vec![0]]);
    let cfg = PageRankConfig {
        damping: 0.85,
        max_iterations: 200,
        tolerance: 1e-12,
    };
    let pr = pagerank(&g, cfg);
    assert_eq!(pr.len(), 3);
    let s: f64 = pr.iter().sum();
    assert!((s - 1.0).abs() < 1e-9);
    for &x in &pr {
        assert!((x - (1.0 / 3.0)).abs() < 1e-6);
    }
}

#[test]
fn pagerank_empty_graph_is_empty() {
    let g = AdjListGraph::new(vec![]);
    let cfg = PageRankConfig::default();
    let pr = pagerank(&g, cfg);
    assert!(pr.is_empty());
}

#[test]
fn pagerank_symmetric_with_dangling_node() {
    // 0 <-> 1, and 2 is dangling
    let g = AdjListGraph::new(vec![vec![1], vec![0], vec![]]);
    let cfg = PageRankConfig {
        damping: 0.85,
        max_iterations: 200,
        tolerance: 1e-12,
    };
    let pr = pagerank(&g, cfg);
    assert_eq!(pr.len(), 3);
    let s: f64 = pr.iter().sum();
    assert!((s - 1.0).abs() < 1e-9);
    // Symmetry: 0 and 1 should match.
    assert!((pr[0] - pr[1]).abs() < 1e-9);
    // All are non-negative.
    assert!(pr.iter().all(|&x| x >= 0.0));
}

#[test]
fn personalized_pagerank_respects_personalization() {
    // Line: 0 - 1 - 2 (undirected edges)
    let g = AdjListGraph::new(vec![vec![1], vec![0, 2], vec![1]]);
    let cfg = PageRankConfig {
        damping: 0.85,
        max_iterations: 200,
        tolerance: 1e-12,
    };

    // Compare to uniform personalization (sum=0 => fallback to uniform).
    let pr_uniform = personalized_pagerank(&g, cfg, &[0.0, 0.0, 0.0]);
    let s0: f64 = pr_uniform.iter().sum();
    assert!((s0 - 1.0).abs() < 1e-9);

    // Personalize to node 0: expect mass shifts toward node 0.
    let pr0 = personalized_pagerank(&g, cfg, &[1.0, 0.0, 0.0]);
    let s1: f64 = pr0.iter().sum();
    assert!((s1 - 1.0).abs() < 1e-9);
    assert!(pr0.iter().all(|&x| x >= 0.0));
    assert!(pr0[0] > pr_uniform[0]);
    assert!(pr0[2] < pr_uniform[2]);

    // Symmetry check: personalizing to node 2 should mirror the vector.
    let pr2 = personalized_pagerank(&g, cfg, &[0.0, 0.0, 1.0]);
    let s2: f64 = pr2.iter().sum();
    assert!((s2 - 1.0).abs() < 1e-9);
    assert!((pr0[0] - pr2[2]).abs() < 1e-9);
    assert!((pr0[1] - pr2[1]).abs() < 1e-9);
    assert!((pr0[2] - pr2[0]).abs() < 1e-9);
}

#[test]
fn personalized_pagerank_empty_graph_is_empty() {
    let g = AdjListGraph::new(vec![]);
    let cfg = PageRankConfig::default();
    let pr = personalized_pagerank(&g, cfg, &[]);
    assert!(pr.is_empty());
}

proptest! {
    // Property: all emitted steps are in-range and follow edges.
    //
    // This catches bugs where we accidentally pick an invalid neighbor or corrupt indices.
    #[test]
    fn prop_walks_follow_edges_and_are_in_range(
        n in 1usize..8,
        adj in prop::collection::vec(prop::collection::vec(0usize..8, 0..8), 1..8),
        seed in any::<u64>(),
    ) {
        // Normalize shapes to exactly n nodes and clamp neighbor ids into range.
        let mut adj2: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, nbrs) in adj.into_iter().take(n).enumerate() {
            adj2[i] = nbrs.into_iter().map(|x| x % n).collect();
        }
        let g = AdjListGraph::new(adj2);

        let cfg = WalkConfig { length: 10, walks_per_node: 2, p: 0.5, q: 2.0, seed };
        let u = generate_walks_ref(&g, cfg);
        assert_walks_sane(&u, Graph::node_count(&g), cfg.length);
        assert_walks_follow_edges_ref(&g, &u);

        let b = generate_biased_walks_ref(&g, cfg);
        assert_walks_sane(&b, Graph::node_count(&g), cfg.length);
        assert_walks_follow_edges_ref(&g, &b);
    }
}

#[test]
fn ref_from_nodes_is_reproducible_and_subset_sized() {
    let g = AdjListGraph::new(vec![vec![1], vec![0, 2, 3], vec![1, 3], vec![1, 2]]);
    let cfg = WalkConfig {
        length: 6,
        walks_per_node: 4,
        p: 0.5,
        q: 2.0,
        seed: 123,
    };
    let starts = [0usize, 2usize];

    let w1 = walk::generate_walks_ref_from_nodes(&g, &starts, cfg);
    let w2 = walk::generate_walks_ref_from_nodes(&g, &starts, cfg);
    assert_eq!(w1, w2);
    assert_eq!(w1.len(), starts.len() * cfg.walks_per_node);
    assert_walks_sane(&w1, Graph::node_count(&g), cfg.length);

    let b1 = walk::generate_biased_walks_ref_from_nodes(&g, &starts, cfg);
    let b2 = walk::generate_biased_walks_ref_from_nodes(&g, &starts, cfg);
    assert_eq!(b1, b2);
    assert_eq!(b1.len(), starts.len() * cfg.walks_per_node);
    assert_walks_sane(&b1, Graph::node_count(&g), cfg.length);
}

#[test]
fn graph_from_nodes_is_reproducible_and_subset_sized() {
    let g = AdjListGraph::new(vec![vec![1], vec![0, 2, 3], vec![1, 3], vec![1, 2]]);
    let cfg = WalkConfig {
        length: 6,
        walks_per_node: 4,
        p: 0.5,
        q: 2.0,
        seed: 123,
    };
    let starts = [0usize, 2usize];

    let w1 = walk::generate_walks_from_nodes(&g, &starts, cfg);
    let w2 = walk::generate_walks_from_nodes(&g, &starts, cfg);
    assert_eq!(w1, w2);
    assert_eq!(w1.len(), starts.len() * cfg.walks_per_node);
    assert_walks_sane(&w1, Graph::node_count(&g), cfg.length);

    let b1 = walk::generate_biased_walks_from_nodes(&g, &starts, cfg);
    let b2 = walk::generate_biased_walks_from_nodes(&g, &starts, cfg);
    assert_eq!(b1, b2);
    assert_eq!(b1.len(), starts.len() * cfg.walks_per_node);
    assert_walks_sane(&b1, Graph::node_count(&g), cfg.length);
}

#[test]
fn precomp_is_reproducible_and_sane() {
    let g = AdjListGraph::new(vec![
        vec![1],       // 0
        vec![0, 2, 3], // 1
        vec![1, 3],    // 2
        vec![1, 2],    // 3
    ]);

    let cfg = WalkConfig {
        length: 8,
        walks_per_node: 3,
        p: 0.5,
        q: 2.0,
        seed: 7,
    };

    let pre = PrecomputedBiasedWalks::new(&g, cfg.p, cfg.q);
    let w1 = generate_biased_walks_precomp_ref(&pre, cfg);
    let w2 = generate_biased_walks_precomp_ref(&pre, cfg);

    assert_eq!(
        w1, w2,
        "precomputed walks must be deterministic for same seed"
    );
    assert_walks_sane(&w1, Graph::node_count(&g), cfg.length);
}

#[test]
fn precomp_from_nodes_is_reproducible_and_subset_sized() {
    let g = AdjListGraph::new(vec![
        vec![1],       // 0
        vec![0, 2, 3], // 1
        vec![1, 3],    // 2
        vec![1, 2],    // 3
    ]);

    let cfg = WalkConfig {
        length: 8,
        walks_per_node: 3,
        p: 0.5,
        q: 2.0,
        seed: 7,
    };
    let starts = [1usize, 3usize];

    let pre = PrecomputedBiasedWalks::new(&g, cfg.p, cfg.q);
    let w1 = generate_biased_walks_precomp_ref_from_nodes(&pre, &starts, cfg);
    let w2 = generate_biased_walks_precomp_ref_from_nodes(&pre, &starts, cfg);
    assert_eq!(w1, w2);
    assert_eq!(w1.len(), starts.len() * cfg.walks_per_node);
    assert_walks_sane(&w1, Graph::node_count(&g), cfg.length);
}

#[test]
fn node2vec_plus_matches_node2vec_on_unit_weights() {
    // For unit weights, node2vec+ reduces to node2vec (PecanPy semantics):
    // - thresholds are constant per node
    // - “loose common edges” never trigger
    // - suppress condition never triggers
    let g = WeightedAdjListGraph::new(
        vec![
            vec![1],       // 0
            vec![0, 2, 3], // 1
            vec![1, 3],    // 2
            vec![1, 2],    // 3
        ],
        vec![
            vec![1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ],
    );

    let cfg = WeightedNode2VecPlusConfig {
        length: 8,
        walks_per_node: 3,
        p: 0.5,
        q: 2.0,
        gamma: 2.0,
        seed: 123,
    };

    let w = generate_biased_walks_weighted_ref(&g, cfg);
    let w_plus = generate_biased_walks_weighted_plus_ref(&g, cfg);
    assert_eq!(
        w, w_plus,
        "node2vec+ must match node2vec when weights are all 1"
    );
}

#[test]
fn precomp_does_not_panic_on_non_reciprocal_edges() {
    // Directed-like adjacency (non-reciprocal): 0 -> 1, 1 -> 2, 2 -> []
    // This is a known footgun for PreComp-style node2vec implementations:
    // the second-order step may assume `prev ∈ neighbors(cur)` (reciprocity).
    let g = AdjListGraph::new(vec![
        vec![1], // 0
        vec![2], // 1
        vec![],  // 2
    ]);

    let cfg = WalkConfig {
        length: 6,
        walks_per_node: 2,
        p: 0.5,
        q: 2.0,
        seed: 1,
    };

    let pre = PrecomputedBiasedWalks::new(&g, cfg.p, cfg.q);
    let walks = generate_biased_walks_precomp_ref(&pre, cfg);
    assert_walks_sane(&walks, Graph::node_count(&g), cfg.length);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_is_thread_count_invariant() {
    use walk::{
        generate_biased_walks_precomp_ref_parallel_from_nodes, generate_biased_walks_ref_parallel,
        generate_walks_ref_parallel,
    };

    let g = AdjListGraph::new(vec![
        vec![1],       // 0
        vec![0, 2, 3], // 1
        vec![1, 3],    // 2
        vec![1, 2],    // 3
    ]);

    let cfg = WalkConfig {
        length: 8,
        walks_per_node: 5,
        p: 0.5,
        q: 2.0,
        seed: 999,
    };

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let u1 = pool1.install(|| generate_walks_ref_parallel(&g, cfg));
    let u4 = pool4.install(|| generate_walks_ref_parallel(&g, cfg));
    assert_eq!(
        u1, u4,
        "unbiased parallel output must be thread-count invariant"
    );

    let b1 = pool1.install(|| generate_biased_walks_ref_parallel(&g, cfg));
    let b4 = pool4.install(|| generate_biased_walks_ref_parallel(&g, cfg));
    assert_eq!(
        b1, b4,
        "biased parallel output must be thread-count invariant"
    );

    // PreComp variant (delta nodes): should also be thread-count invariant.
    let pre = PrecomputedBiasedWalks::new(&g, cfg.p, cfg.q);
    let starts = [0usize, 2usize];
    let p1 =
        pool1.install(|| generate_biased_walks_precomp_ref_parallel_from_nodes(&pre, &starts, cfg));
    let p4 =
        pool4.install(|| generate_biased_walks_precomp_ref_parallel_from_nodes(&pre, &starts, cfg));
    assert_eq!(
        p1, p4,
        "precomp biased parallel output must be thread-count invariant"
    );
}
