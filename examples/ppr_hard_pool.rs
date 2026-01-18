//! End-to-end sketch: PPR → weighted candidate pool → stochastic top-k.
//!
//! This example is intentionally “graph-only” (no tensors) but tries to exercise
//! the exact seams we care about:
//! - `walk::personalized_pagerank` as a cheap locality signal
//! - `kuji::gumbel_topk_sample_with_rng` as a stochastic top-k without replacement
//!
//! A common pattern is “hard negatives from graph locality”:
//! - define a weight vector \(w\) over nodes (e.g. PPR from an anchor)
//! - sample \(k\) candidates without replacement, biased toward larger \(w\)
//!
//! For a single draw (\(k=1\)), Gumbel-max with logits \(\log w_i\) samples
//! \(i\) with probability \(w_i / \sum_j w_j\).

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use walk::{personalized_pagerank, PageRankConfig};

#[derive(Debug, Clone)]
struct Adj {
    adj: Vec<Vec<usize>>,
}

impl Adj {
    fn sbm_two_block(n: usize, p_in: f64, p_out: f64, seed: u64) -> Self {
        assert!(n >= 4);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut adj = vec![Vec::new(); n];
        let half = n / 2;
        for i in 0..n {
            for j in (i + 1)..n {
                let same = (i < half) == (j < half);
                let p = if same { p_in } else { p_out };
                if rng.random::<f64>() < p {
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
        }
        for nbrs in &mut adj {
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        Self { adj }
    }
}

impl walk::Graph for Adj {
    fn node_count(&self) -> usize {
        self.adj.len()
    }
    fn neighbors(&self, node: usize) -> Vec<usize> {
        self.adj[node].clone()
    }
    fn out_degree(&self, node: usize) -> usize {
        self.adj[node].len()
    }
}

fn main() {
    // A “not tiny” graph, but still cheap to run locally.
    let n = 500usize;
    let g = Adj::sbm_two_block(n, 0.02, 0.002, 123);

    let head = 7usize;
    let mut p = vec![0.0f64; n];
    p[head] = 1.0;

    let cfg = PageRankConfig {
        damping: 0.85,
        max_iterations: 80,
        tolerance: 1e-10,
    };

    let scores = personalized_pagerank(&g, cfg, &p);

    // Convert to logits for Gumbel-top-k:
    // logits_i = log(score_i + eps).
    let eps = 1e-12f64;
    let logits: Vec<f32> = scores.iter().map(|&s| (s + eps).ln() as f32).collect();

    // “Hard pool” size. In practice, you’d pass these candidates into downstream
    // negative sampling (e.g., corrupt tail from this pool).
    let k = 30usize;

    // Deterministic stochasticity: seeded RNG.
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let picked = kuji::gumbel_topk_sample_with_rng(&logits, k, &mut rng);

    // Show the top-k-by-score (deterministic) and one stochastic top-k draw.
    let mut by_score: Vec<usize> = (0..n).collect();
    by_score.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));

    println!("graph: n={n}, anchor={head}");
    println!("top-10 by PPR score:");
    for &i in by_score.iter().take(10) {
        println!("  node {i:4}  score={:.6e}", scores[i]);
    }

    println!();
    println!("one stochastic top-{k} draw (Gumbel-top-k on log(PPR)):");
    for &i in &picked {
        println!("  node {i:4}  score={:.6e}", scores[i]);
    }
}

