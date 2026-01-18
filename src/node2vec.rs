//! Node2Vec / Node2Vec+ walk generation (OTF + PreComp).
//!
//! Grounded against PecanPy:
//! - `src/pecanpy/rw/sparse_rw.py` (`get_normalized_probs`, `get_extended_normalized_probs`)
//! - `src/pecanpy/rw/dense_rw.py` (same semantics for dense graphs)
//! - `src/pecanpy/pecanpy.py` (overall walk skeleton)

use crate::graph::{GraphRef, WeightedGraphRef};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Parameters for weighted node2vec / node2vec+ walk generation.
#[derive(Debug, Clone, Copy)]
pub struct WeightedNode2VecPlusConfig {
    /// Maximum walk length (in nodes).
    pub length: usize,
    /// Number of walks per node.
    pub walks_per_node: usize,
    /// Return parameter \(p\).
    pub p: f32,
    /// In-out parameter \(q\).
    pub q: f32,
    /// Node2vec+ parameter \(\gamma\) controlling the “noisy edge” threshold.
    pub gamma: f32,
    /// Seed for deterministic RNG.
    pub seed: u64,
}

impl Default for WeightedNode2VecPlusConfig {
    fn default() -> Self {
        Self {
            length: 80,
            walks_per_node: 10,
            p: 1.0,
            q: 1.0,
            gamma: 0.0,
            seed: 42,
        }
    }
}

pub fn generate_biased_walks_weighted_ref<G: WeightedGraphRef>(
    graph: &G,
    config: WeightedNode2VecPlusConfig,
) -> Vec<Vec<usize>> {
    generate_biased_walks_weighted_impl(graph, config, false)
}

pub fn generate_biased_walks_weighted_plus_ref<G: WeightedGraphRef>(
    graph: &G,
    config: WeightedNode2VecPlusConfig,
) -> Vec<Vec<usize>> {
    generate_biased_walks_weighted_impl(graph, config, true)
}

fn generate_biased_walks_weighted_impl<G: WeightedGraphRef>(
    graph: &G,
    config: WeightedNode2VecPlusConfig,
    extend: bool,
) -> Vec<Vec<usize>> {
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut start_nodes: Vec<usize> = (0..graph.node_count()).collect();

    let noise_thresholds = if extend {
        compute_noise_thresholds(graph, config.gamma)
    } else {
        Vec::new()
    };

    let mut walks = Vec::with_capacity(graph.node_count() * config.walks_per_node);
    for _ in 0..config.walks_per_node {
        start_nodes.shuffle(&mut rng);
        for &node in &start_nodes {
            walks.push(weighted_walk(
                graph,
                node,
                config,
                extend,
                &noise_thresholds,
                &mut rng,
            ));
        }
    }
    walks
}

fn weighted_walk<G: WeightedGraphRef, R: Rng>(
    graph: &G,
    start: usize,
    config: WeightedNode2VecPlusConfig,
    extend: bool,
    noise_thresholds: &[f32],
    rng: &mut R,
) -> Vec<usize> {
    let mut walk = Vec::with_capacity(config.length);
    walk.push(start);

    let mut curr = start;
    let mut prev: Option<usize> = None;
    let mut buf: Vec<f32> = Vec::new();

    for _ in 1..config.length {
        let (nbrs, wts) = graph.neighbors_and_weights_ref(curr);
        if nbrs.is_empty() {
            break;
        }
        debug_assert_eq!(nbrs.len(), wts.len());

        let next = if let Some(prev_idx) = prev {
            if extend {
                sample_next_node2vec_plus(
                    graph,
                    curr,
                    prev_idx,
                    nbrs,
                    wts,
                    config,
                    noise_thresholds,
                    &mut buf,
                    rng,
                )
            } else {
                sample_next_node2vec_weighted(
                    graph,
                    prev_idx,
                    nbrs,
                    wts,
                    config,
                    &mut buf,
                    rng,
                )
            }
        } else {
            sample_cdf(rng, nbrs, wts)
        };

        walk.push(next);
        prev = Some(curr);
        curr = next;
    }

    walk
}

fn sample_next_node2vec_weighted<G: WeightedGraphRef, R: Rng>(
    graph: &G,
    prev: usize,
    nbrs: &[usize],
    wts: &[f32],
    config: WeightedNode2VecPlusConfig,
    buf: &mut Vec<f32>,
    rng: &mut R,
) -> usize {
    fill_next_node2vec_weighted_buf(graph, prev, nbrs, wts, config, buf);
    sample_cdf(rng, nbrs, buf)
}

fn fill_next_node2vec_weighted_buf<G: WeightedGraphRef>(
    graph: &G,
    prev: usize,
    nbrs: &[usize],
    wts: &[f32],
    config: WeightedNode2VecPlusConfig,
    buf: &mut Vec<f32>,
) {
    // Classic node2vec: out edges are neighbors(cur) that are not neighbors(prev).
    let (prev_nbrs, _prev_wts) = graph.neighbors_and_weights_ref(prev);

    buf.clear();
    buf.extend_from_slice(wts);

    // return bias
    if let Some(i) = nbrs.iter().position(|&x| x == prev) {
        buf[i] /= config.p;
    }

    for i in 0..nbrs.len() {
        let x = nbrs[i];
        if x == prev {
            continue;
        }
        let is_common = prev_nbrs.iter().any(|&y| y == x);
        if !is_common {
            buf[i] /= config.q;
        }
    }
}

fn sample_next_node2vec_plus<G: WeightedGraphRef, R: Rng>(
    graph: &G,
    cur: usize,
    prev: usize,
    nbrs: &[usize],
    wts: &[f32],
    config: WeightedNode2VecPlusConfig,
    noise_thresholds: &[f32],
    buf: &mut Vec<f32>,
    rng: &mut R,
) -> usize {
    fill_next_node2vec_plus_buf(
        graph,
        cur,
        prev,
        nbrs,
        wts,
        config,
        noise_thresholds,
        buf,
    );
    sample_cdf(rng, nbrs, buf)
}

fn fill_next_node2vec_plus_buf<G: WeightedGraphRef>(
    graph: &G,
    cur: usize,
    prev: usize,
    nbrs: &[usize],
    wts: &[f32],
    config: WeightedNode2VecPlusConfig,
    noise_thresholds: &[f32],
    buf: &mut Vec<f32>,
) {
    // PecanPy semantics (SparseRWGraph.get_extended_normalized_probs):
    // - Determine out edges via `isnotin_extended`.
    // - alpha(out) = 1/q + (1 - 1/q) * t(out), where:
    //   - t=0 for non-common neighbors
    //   - t=w(prev,x)/threshold[x] for “loose common” edges (when w(prev,x) < threshold[x])
    // - suppress: if w(cur,x) < threshold[cur], alpha = min(1, 1/q).

    let (prev_nbrs, prev_wts) = graph.neighbors_and_weights_ref(prev);

    buf.clear();
    buf.extend_from_slice(wts);

    // return bias
    if let Some(i) = nbrs.iter().position(|&x| x == prev) {
        buf[i] /= config.p;
    }

    let inv_q = 1.0 / config.q;
    let thr_cur = noise_thresholds[cur];

    for i in 0..nbrs.len() {
        let x = nbrs[i];
        if x == prev {
            continue;
        }

        let mut is_out = true;
        let mut t: f32 = 0.0;

        if let Some(j) = prev_nbrs.iter().position(|&y| y == x) {
            let thr_x = noise_thresholds[x];
            let w_prev_x = prev_wts[j];
            if thr_x > 0.0 && w_prev_x >= thr_x {
                // strong common edge => in-edge
                is_out = false;
            } else if thr_x > 0.0 {
                // loose common edge => out-edge with t in (0, 1)
                t = (w_prev_x / thr_x).max(0.0);
            }
        }

        if is_out {
            let mut alpha = inv_q + (1.0 - inv_q) * t;
            if buf[i] < thr_cur {
                alpha = inv_q.min(1.0);
            }
            buf[i] *= alpha;
        }
    }
}

fn compute_noise_thresholds<G: WeightedGraphRef>(graph: &G, gamma: f32) -> Vec<f32> {
    let n = graph.node_count();
    let mut thr = vec![0.0f32; n];

    for v in 0..n {
        let (_nbrs, wts) = graph.neighbors_and_weights_ref(v);
        if wts.is_empty() {
            thr[v] = 0.0;
            continue;
        }

        let mean = wts.iter().copied().sum::<f32>() / (wts.len() as f32);
        let var = wts
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d
            })
            .sum::<f32>()
            / (wts.len() as f32);
        let std = var.sqrt();

        thr[v] = (mean + gamma * std).max(0.0);
    }

    thr
}

fn sample_cdf<R: Rng>(rng: &mut R, nbrs: &[usize], weights: &[f32]) -> usize {
    debug_assert_eq!(nbrs.len(), weights.len());
    if nbrs.len() == 1 {
        return nbrs[0];
    }

    let sum = weights.iter().copied().sum::<f32>();
    if !(sum > 0.0) {
        return *nbrs.choose(rng).unwrap();
    }

    let mut r = rng.random::<f32>() * sum;
    for (i, &w) in weights.iter().enumerate() {
        if r <= w {
            return nbrs[i];
        }
        r -= w;
    }
    *nbrs.last().unwrap()
}

/// Precomputed alias tables for classic node2vec biased walks (unweighted).
#[derive(Debug, Clone)]
pub struct PrecomputedBiasedWalks {
    neighbors: Vec<Vec<usize>>,
    alias_dim: Vec<u32>,
    alias_indptr: Vec<u64>,
    alias_j: Vec<u32>,
    alias_q: Vec<f32>,
    p: f32,
    q: f32,
}

impl PrecomputedBiasedWalks {
    pub fn new<G: GraphRef>(graph: &G, p: f32, q: f32) -> Self {
        let n = graph.node_count();
        let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut alias_dim: Vec<u32> = Vec::with_capacity(n);

        for v in 0..n {
            let mut nbrs = graph.neighbors_ref(v).to_vec();
            nbrs.sort_unstable();
            alias_dim.push(nbrs.len() as u32);
            neighbors.push(nbrs);
        }

        let mut alias_indptr: Vec<u64> = vec![0; n + 1];
        for i in 0..n {
            let deg = alias_dim[i] as u64;
            alias_indptr[i + 1] = alias_indptr[i] + deg * deg;
        }
        let total = alias_indptr[n] as usize;

        let mut alias_j = vec![0u32; total];
        let mut alias_q = vec![0.0f32; total];

        let mut out_ind: Vec<bool> = Vec::new();
        let mut probs: Vec<f32> = Vec::new();

        for cur in 0..n {
            let deg = alias_dim[cur] as usize;
            if deg == 0 {
                continue;
            }
            let offset = alias_indptr[cur] as usize;
            let cur_nbrs = &neighbors[cur];

            out_ind.clear();
            out_ind.resize(deg, true);
            probs.clear();
            probs.resize(deg, 1.0);

            for prev_j in 0..deg {
                let prev = cur_nbrs[prev_j];
                let prev_nbrs = &neighbors[prev];

                mark_non_common(cur_nbrs, prev_nbrs, &mut out_ind);
                out_ind[prev_j] = false; // exclude prev from out biases

                probs.fill(1.0);
                for i in 0..deg {
                    if out_ind[i] {
                        probs[i] /= q;
                    }
                }
                probs[prev_j] /= p;

                normalize_in_place(&mut probs);
                let (j, qtab) = alias_setup(&probs);

                let start = offset + deg * prev_j;
                let end = start + deg;
                alias_j[start..end].copy_from_slice(&j);
                alias_q[start..end].copy_from_slice(&qtab);
            }
        }

        Self { neighbors, alias_dim, alias_indptr, alias_j, alias_q, p, q }
    }
}

pub fn generate_biased_walks_precomp_ref(
    pre: &PrecomputedBiasedWalks,
    config: crate::random_walk::WalkConfig,
) -> Vec<Vec<usize>> {
    let start_nodes: Vec<usize> = (0..pre.neighbors.len()).collect();
    generate_biased_walks_precomp_ref_from_nodes(pre, &start_nodes, config)
}

/// Precomputed node2vec biased walks, restricted to an explicit set of start nodes.
///
/// This is the “delta walk” primitive for PreComp mode: generate new walks only for the
/// subset of nodes whose neighborhood changed (dynamic graphs), or for sharding.
pub fn generate_biased_walks_precomp_ref_from_nodes(
    pre: &PrecomputedBiasedWalks,
    start_nodes: &[usize],
    config: crate::random_walk::WalkConfig,
) -> Vec<Vec<usize>> {
    if (pre.p - config.p).abs() > 1e-6 || (pre.q - config.q).abs() > 1e-6 {
        panic!("PrecomputedBiasedWalks p/q do not match WalkConfig");
    }

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut epoch_nodes: Vec<usize> = start_nodes.to_vec();
    let mut walks = Vec::with_capacity(start_nodes.len() * config.walks_per_node);

    for _ in 0..config.walks_per_node {
        epoch_nodes.shuffle(&mut rng);
        for &node in &epoch_nodes {
            walks.push(biased_walk_precomp(pre, node, config.length, &mut rng));
        }
    }

    walks
}

/// Deterministic parallel precomputed node2vec biased walks (delta/sharded start nodes).
///
/// Invariant: output is stable for a fixed `seed`, independent of Rayon thread count.
#[cfg(feature = "parallel")]
pub fn generate_biased_walks_precomp_ref_parallel_from_nodes(
    pre: &PrecomputedBiasedWalks,
    start_nodes: &[usize],
    config: crate::random_walk::WalkConfig,
) -> Vec<Vec<usize>> {
    use rayon::prelude::*;

    if (pre.p - config.p).abs() > 1e-6 || (pre.q - config.q).abs() > 1e-6 {
        panic!("PrecomputedBiasedWalks p/q do not match WalkConfig");
    }

    // Copy start nodes once; shuffle per epoch using a seed that depends only on (seed, epoch).
    let mut epoch_nodes: Vec<usize> = start_nodes.to_vec();
    let mut jobs: Vec<(u32, usize)> = Vec::with_capacity(start_nodes.len() * config.walks_per_node);

    for epoch in 0..(config.walks_per_node as u32) {
        // Keep a local mix64 here to avoid exposing random_walk::mix64 publicly.
        fn mix64(mut x: u64) -> u64 {
            x ^= x >> 30;
            x = x.wrapping_mul(0xbf58476d1ce4e5b9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            x
        }

        let mut rng = ChaCha8Rng::seed_from_u64(mix64(config.seed ^ (epoch as u64)));
        epoch_nodes.shuffle(&mut rng);
        for &node in &epoch_nodes {
            jobs.push((epoch, node));
        }
    }

    jobs.par_iter()
        .enumerate()
        .map(|(i, (epoch, node))| {
            fn mix64(mut x: u64) -> u64 {
                x ^= x >> 30;
                x = x.wrapping_mul(0xbf58476d1ce4e5b9);
                x ^= x >> 27;
                x = x.wrapping_mul(0x94d049bb133111eb);
                x ^= x >> 31;
                x
            }

            let seed = mix64(config.seed ^ ((*epoch as u64) << 32) ^ (*node as u64) ^ (i as u64));
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            biased_walk_precomp(pre, *node, config.length, &mut rng)
        })
        .collect()
}

fn biased_walk_precomp<R: Rng>(
    pre: &PrecomputedBiasedWalks,
    start: usize,
    length: usize,
    rng: &mut R,
) -> Vec<usize> {
    let mut walk = Vec::with_capacity(length);
    walk.push(start);
    let mut curr = start;
    let mut prev: Option<usize> = None;

    for _ in 1..length {
        let nbrs = &pre.neighbors[curr];
        if nbrs.is_empty() {
            break;
        }

        let next = if let Some(p) = prev {
            sample_precomp(pre, curr, p, rng)
        } else {
            *nbrs.choose(rng).unwrap()
        };

        walk.push(next);
        prev = Some(curr);
        curr = next;
    }

    walk
}

fn sample_precomp<R: Rng>(pre: &PrecomputedBiasedWalks, cur: usize, prev: usize, rng: &mut R) -> usize {
    let nbrs = &pre.neighbors[cur];
    let deg = pre.alias_dim[cur] as usize;
    let prev_j = match nbrs.binary_search(&prev) {
        Ok(i) => i,
        Err(_) => {
            // This can happen on directed / non-reciprocal graphs: we might have walked
            // from `prev -> cur`, but `cur` may not have `prev` in its neighbor list.
            // PecanPy prints "FATAL ERROR! Neighbor not found." in this situation.
            //
            // In Rust, we choose a safe fallback that preserves determinism and avoids
            // returning a nonsense index: fall back to a 1st-order uniform step.
            return *nbrs.choose(rng).unwrap();
        }
    };

    let offset = pre.alias_indptr[cur] + (deg as u64) * (prev_j as u64);
    let start = offset as usize;
    let end = start + deg;

    let choice = alias_draw(&pre.alias_j[start..end], &pre.alias_q[start..end], rng);
    nbrs[choice]
}

fn normalize_in_place(x: &mut [f32]) {
    let s = x.iter().copied().sum::<f32>();
    if s > 0.0 {
        for v in x {
            *v /= s;
        }
    }
}

fn mark_non_common(cur: &[usize], prev: &[usize], out: &mut [bool]) {
    debug_assert_eq!(cur.len(), out.len());
    let mut j = 0usize;
    for (i, &x) in cur.iter().enumerate() {
        while j < prev.len() && prev[j] < x {
            j += 1;
        }
        out[i] = !(j < prev.len() && prev[j] == x);
    }
}

fn alias_setup(probs: &[f32]) -> (Vec<u32>, Vec<f32>) {
    // Alias table construction (O(k)) for O(1) categorical draws.
    //
    // This implementation matches the common “Walker/Vose alias method” presentation and is
    // intentionally structured to mirror PecanPy’s implementation to reduce drift when comparing
    // outputs and edge cases.
    //
    // References:
    // - Walker (1974): An efficient method for generating discrete random variables with general distributions.
    // - Vose (1991): A linear algorithm for generating random numbers with a given distribution.
    // - PecanPy (software reference implementation): https://github.com/krishnanlab/PecanPy
    let k = probs.len();
    let mut q = vec![0.0f32; k];
    let mut j = vec![0u32; k];

    let mut smaller: Vec<usize> = Vec::with_capacity(k);
    let mut larger: Vec<usize> = Vec::with_capacity(k);

    for kk in 0..k {
        q[kk] = (k as f32) * probs[kk];
        if q[kk] < 1.0 {
            smaller.push(kk);
        } else {
            larger.push(kk);
        }
    }

    while let (Some(small), Some(large)) = (smaller.pop(), larger.pop()) {
        j[small] = large as u32;
        q[large] = q[large] + q[small] - 1.0;
        if q[large] < 1.0 {
            smaller.push(large);
        } else {
            larger.push(large);
        }
    }

    (j, q)
}

fn alias_draw<R: Rng>(j: &[u32], q: &[f32], rng: &mut R) -> usize {
    debug_assert_eq!(j.len(), q.len());
    let k = j.len();
    let kk = rng.random_range(0..k);
    if rng.random::<f32>() < q[kk] {
        kk
    } else {
        j[kk] as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct RefAdj {
        adj: Vec<Vec<usize>>,
    }

    impl RefAdj {
        fn new(mut adj: Vec<Vec<usize>>) -> Self {
            for nbrs in &mut adj {
                nbrs.sort_unstable();
            }
            Self { adj }
        }
    }

    impl GraphRef for RefAdj {
        fn node_count(&self) -> usize {
            self.adj.len()
        }

        fn neighbors_ref(&self, node: usize) -> &[usize] {
            self.adj.get(node).map(Vec::as_slice).unwrap_or(&[])
        }
    }

    #[derive(Debug, Clone)]
    struct RefWeightedAdj {
        adj: Vec<Vec<usize>>,
        wts: Vec<Vec<f32>>,
    }

    impl RefWeightedAdj {
        fn new(mut adj: Vec<Vec<usize>>, mut wts: Vec<Vec<f32>>) -> Self {
            assert_eq!(adj.len(), wts.len());
            for i in 0..adj.len() {
                assert_eq!(adj[i].len(), wts[i].len());
                let mut pairs: Vec<(usize, f32)> =
                    adj[i].iter().copied().zip(wts[i].iter().copied()).collect();
                pairs.sort_by_key(|(n, _)| *n);
                adj[i] = pairs.iter().map(|(n, _)| *n).collect();
                wts[i] = pairs.iter().map(|(_, w)| *w).collect();
            }
            Self { adj, wts }
        }
    }

    impl WeightedGraphRef for RefWeightedAdj {
        fn node_count(&self) -> usize {
            self.adj.len()
        }

        fn neighbors_and_weights_ref(&self, node: usize) -> (&[usize], &[f32]) {
            let nbrs = self.adj.get(node).map(Vec::as_slice).unwrap_or(&[]);
            let wts = self.wts.get(node).map(Vec::as_slice).unwrap_or(&[]);
            (nbrs, wts)
        }
    }

    fn assert_close_f32(a: f32, b: f32, eps: f32) {
        assert!(
            (a - b).abs() <= eps,
            "expected |{a} - {b}| <= {eps}, got {}",
            (a - b).abs()
        );
    }

    #[test]
    fn alias_tables_match_expected_for_line_graph() {
        // Graph: 0 -- 1 -- 2
        // Using p=0.5, q=2.0, when at cur=1 coming from prev=0:
        // weights are [1/p, 1/q] => [2.0, 0.5] => normalized [0.8, 0.2]
        let g = RefAdj::new(vec![vec![1], vec![0, 2], vec![1]]);
        let pre = PrecomputedBiasedWalks::new(&g, 0.5, 2.0);

        assert_eq!(pre.alias_dim, vec![1, 2, 1]);
        assert_eq!(pre.alias_indptr, vec![0, 1, 5, 6]);

        // cur = 1, neighbors = [0, 2], deg=2.
        // prev=0 corresponds to prev_j=0, slice is offset=alias_indptr[1]=1, start=1, end=3.
        let j01 = &pre.alias_j[1..3];
        let q01 = &pre.alias_q[1..3];
        assert_eq!(j01, &[0u32, 0u32]);
        assert_close_f32(q01[0], 1.0, 1e-6);
        assert_close_f32(q01[1], 0.4, 1e-6);

        // prev=2 corresponds to prev_j=1, start=3, end=5.
        let j21 = &pre.alias_j[3..5];
        let q21 = &pre.alias_q[3..5];
        assert_eq!(j21, &[1u32, 0u32]);
        assert_close_f32(q21[0], 0.4, 1e-6);
        assert_close_f32(q21[1], 1.0, 1e-6);
    }

    #[test]
    fn noise_thresholds_match_mean_plus_gamma_std() {
        // One node with two outgoing weights: [1, 3]
        // mean=2, std=1, tau = mean + gamma*std
        let g = RefWeightedAdj::new(vec![vec![0]], vec![vec![1.0]]);
        let thr0 = compute_noise_thresholds(&g, 2.0);
        assert_eq!(thr0.len(), 1);
        // Single weight => std=0, tau=mean=1
        assert_close_f32(thr0[0], 1.0, 1e-6);

        let g2 = RefWeightedAdj::new(vec![vec![0, 1]], vec![vec![1.0, 3.0]]);
        let thr2 = compute_noise_thresholds(&g2, 2.0);
        assert_eq!(thr2.len(), 1);
        assert_close_f32(thr2[0], 4.0, 1e-6);
    }

    #[test]
    fn node2vec_plus_suppress_caps_inv_q_when_q_lt_1() {
        // Construct a situation where:
        // - q < 1 (so inv_q > 1 would amplify out-edges in classic node2vec)
        // - node2vec+ suppresses that amplification for “noisy” edges with
        //   w(cur, x) < threshold[cur].
        //
        // Graph: 0 -- 1 -- 2 (weighted, symmetric for existence, but asymmetric weights at node 1)
        // At cur=1 coming from prev=0, candidate x=2 is an out-edge.
        let g = RefWeightedAdj::new(
            vec![vec![1], vec![0, 2], vec![1]],
            vec![vec![1.0], vec![1.0, 0.9], vec![1.0]],
        );

        let cfg = WeightedNode2VecPlusConfig {
            length: 3,
            walks_per_node: 1,
            p: 1.0,
            q: 0.5,      // inv_q = 2.0
            gamma: 0.0,  // threshold is mean
            seed: 0,
        };

        let thr = compute_noise_thresholds(&g, cfg.gamma);
        assert_eq!(thr.len(), 3);
        // For node 1: weights [1.0, 0.9], mean=0.95 => threshold=0.95
        assert_close_f32(thr[1], 0.95, 1e-6);

        let (nbrs, wts) = g.neighbors_and_weights_ref(1);
        assert_eq!(nbrs, &[0, 2]);
        assert_eq!(wts, &[1.0, 0.9]);

        let mut buf_weighted = Vec::new();
        let mut buf_plus = Vec::new();

        fill_next_node2vec_weighted_buf(&g, 0, nbrs, wts, cfg, &mut buf_weighted);
        fill_next_node2vec_plus_buf(&g, 1, 0, nbrs, wts, cfg, &thr, &mut buf_plus);

        // For x=2 (out-edge) classic weighted node2vec divides by q (q=0.5 => multiply by 2).
        assert_close_f32(buf_weighted[1], 1.8, 1e-6);

        // For node2vec+, since w(cur,2)=0.9 < threshold[cur]=0.95 and inv_q>1,
        // suppress caps alpha at 1.0, so out-edge stays at 0.9.
        assert_close_f32(buf_plus[1], 0.9, 1e-6);
    }

    #[test]
    fn alias_draw_distribution_smoke() {
        // Deterministic chi-squared smoke test: catches egregious alias bugs
        // without being overly sensitive/flaky.
        //
        // Distribution: [0.1, 0.2, 0.7]
        let probs = vec![0.1f32, 0.2f32, 0.7f32];
        let (j, q) = alias_setup(&probs);

        let trials = 20_000usize;
        let mut counts = [0usize; 3];
        for t in 0..trials {
            let mut rng = ChaCha8Rng::seed_from_u64(t as u64);
            let k = alias_draw(&j, &q, &mut rng);
            counts[k] += 1;
        }

        let expected = [
            trials as f64 * 0.1,
            trials as f64 * 0.2,
            trials as f64 * 0.7,
        ];
        let chi2: f64 = counts
            .iter()
            .zip(expected.iter())
            .map(|(&c, &e)| {
                let diff = c as f64 - e;
                (diff * diff) / e
            })
            .sum();

        // df = 2; E[chi2] ~ 2, Var ~ 4. Use a very conservative cutoff.
        assert!(
            chi2 < 50.0,
            "chi2 too large (chi2={chi2:.2}). counts={counts:?} expected={expected:?}"
        );
    }
}

