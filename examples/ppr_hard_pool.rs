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

use walk::{personalized_pagerank, Graph, PageRankConfig};
use std::path::Path;

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

    /// Load an undirected edge list (two whitespace-separated node ids per line).
    ///
    /// Lines starting with `#` are ignored.
    fn from_undirected_edgelist(path: &Path) -> Result<Self, String> {
        let txt = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut max_node = 0usize;

        for (line_no, line) in txt.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let a = it
                .next()
                .ok_or_else(|| format!("line {}: missing src", line_no + 1))?;
            let b = it
                .next()
                .ok_or_else(|| format!("line {}: missing dst", line_no + 1))?;
            let u: usize = a
                .parse()
                .map_err(|e| format!("line {}: bad src '{a}': {e}", line_no + 1))?;
            let v: usize = b
                .parse()
                .map_err(|e| format!("line {}: bad dst '{b}': {e}", line_no + 1))?;
            max_node = max_node.max(u).max(v);
            edges.push((u, v));
        }

        let n = max_node + 1;
        if n == 0 {
            return Err("edgelist produced empty graph".to_string());
        }

        let mut adj = vec![Vec::new(); n];
        for (u, v) in edges {
            if u == v {
                continue;
            }
            adj[u].push(v);
            adj[v].push(u);
        }
        for nbrs in &mut adj {
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        Ok(Self { adj })
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
    // If you have a real graph, point to it:
    //
    // WALK_EDGELIST=/path/to/edges.txt cargo run --example ppr_hard_pool
    //
    // Format: two whitespace-separated integer node ids per line, undirected.
    let g = if let Ok(path) = std::env::var("WALK_EDGELIST") {
        Adj::from_undirected_edgelist(Path::new(&path)).expect("failed to load WALK_EDGELIST")
    } else {
        // Prefer a small real graph if present in-repo.
        let karate = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/karate_club.edgelist");
        if karate.exists() {
            Adj::from_undirected_edgelist(&karate).expect("failed to load testdata/karate_club.edgelist")
        } else {
            // Otherwise, use a seeded SBM graph (realistic topology, deterministic).
            Adj::sbm_two_block(500, 0.02, 0.002, 123)
        }
    };
    let n = g.node_count();

    let head = 7usize.min(n.saturating_sub(1));
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

