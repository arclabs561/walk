//! Benchmarks for walk generation and sampling strategies.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand::SeedableRng;
use std::hint::black_box;
use walk::{
    generate_biased_walks_precomp_ref, generate_biased_walks_ref, generate_walks_ref,
    PrecomputedBiasedWalks, WalkConfig,
};
use walk::{Graph, GraphRef};

#[derive(Debug, Clone)]
struct AdjListGraph {
    adj: Vec<Vec<usize>>,
}

impl AdjListGraph {
    fn ring(n: usize) -> Self {
        let mut adj = vec![Vec::new(); n];
        for i in 0..n {
            adj[i].push((i + 1) % n);
            adj[i].push((i + n - 1) % n);
            adj[i].sort_unstable();
        }
        Self { adj }
    }

    /// Preferential attachment graph (Barabási–Albert) with `m` edges per new node.
    ///
    /// This yields a heavy-tailed degree distribution that’s closer to many real graphs
    /// than a ring/grid.
    fn barabasi_albert(n: usize, m: usize, seed: u64) -> Self {
        assert!(n >= m.max(2));
        assert!(m >= 1);

        let mut rng = StdRng::seed_from_u64(seed);
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Start with a clique of size m+1.
        let init = m + 1;
        let mut targets: Vec<usize> = Vec::new(); // node ids repeated by degree
        for i in 0..init {
            for j in (i + 1)..init {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
        for i in 0..init {
            for _ in 0..adj[i].len() {
                targets.push(i);
            }
        }

        // Add nodes, attaching to existing nodes proportional to degree.
        for v in init..n {
            let mut chosen: Vec<usize> = Vec::with_capacity(m);
            while chosen.len() < m {
                let u = targets[rng.random_range(0..targets.len())];
                if u != v && !chosen.contains(&u) {
                    chosen.push(u);
                }
            }
            for &u in &chosen {
                adj[v].push(u);
                adj[u].push(v);
            }
            // Update targets: each new edge increases degree of both endpoints by 1.
            for &u in &chosen {
                targets.push(u);
                targets.push(v);
            }
        }

        for nbrs in &mut adj {
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        Self { adj }
    }

    /// Simple stochastic block model: `blocks` equal-sized communities.
    fn sbm(n: usize, blocks: usize, p_in: f64, p_out: f64, seed: u64) -> Self {
        assert!(blocks >= 2);
        assert!(n >= blocks);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let bsz = (n + blocks - 1) / blocks;

        for i in 0..n {
            let bi = (i / bsz).min(blocks - 1);
            for j in (i + 1)..n {
                let bj = (j / bsz).min(blocks - 1);
                let p = if bi == bj { p_in } else { p_out };
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

fn bench_walk_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("walk_generation");

    for n in [1_000usize, 10_000] {
        // Use a few graph families to avoid overfitting perf intuition to a toy topology.
        let graphs = [
            ("ring", AdjListGraph::ring(n)),
            ("ba_m4", AdjListGraph::barabasi_albert(n, 4, 123)),
            ("sbm4", AdjListGraph::sbm(n, 4, 0.02, 0.002, 123)),
        ];

        // Keep total work bounded.
        let cfg = WalkConfig {
            length: 40,
            walks_per_node: 2,
            p: 0.5,
            q: 2.0,
            seed: 123,
        };

        for (name, g) in graphs {
            group.bench_with_input(
                BenchmarkId::new(format!("{name}/unbiased_ref"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        let walks = generate_walks_ref(black_box(&g), black_box(cfg));
                        black_box(walks);
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("{name}/biased_ref_otf"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        let walks = generate_biased_walks_ref(black_box(&g), black_box(cfg));
                        black_box(walks);
                    })
                },
            );

            // PreComp setup cost is separate from walk generation.
            let pre = PrecomputedBiasedWalks::new(&g, cfg.p, cfg.q);
            group.bench_with_input(
                BenchmarkId::new(format!("{name}/biased_ref_precomp"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        let walks =
                            generate_biased_walks_precomp_ref(black_box(&pre), black_box(cfg));
                        black_box(walks);
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_walk_generation);
criterion_main!(benches);
