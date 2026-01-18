use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::alloc::System;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[derive(Debug, Clone)]
struct RefAdj {
    adj: Vec<Vec<usize>>,
}

impl walk::GraphRef for RefAdj {
    fn node_count(&self) -> usize {
        self.adj.len()
    }

    fn neighbors_ref(&self, node: usize) -> &[usize] {
        self.adj.get(node).map(Vec::as_slice).unwrap_or(&[])
    }
}

#[test]
fn streaming_walks_use_far_fewer_allocations_than_collecting() {
    // This is a “resource consumption” test:
    // - collecting APIs allocate per-walk (Vec<Vec<...>> + each walk Vec)
    // - streaming APIs should be close to allocation-flat w.r.t. number of walks
    //
    // We test this by counting allocations, not RSS (portable across OSes/CI).

    // Build a simple chain graph.
    let n = 1_000usize;
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        if i > 0 {
            adj[i].push(i - 1);
        }
        if i + 1 < n {
            adj[i].push(i + 1);
        }
    }
    let g = RefAdj { adj };

    let start_nodes: Vec<usize> = (0..n).collect();
    let config = walk::WalkConfig {
        length: 80,
        walks_per_node: 2,
        p: 1.0,
        q: 1.0,
        seed: 123,
    };

    // Collecting (allocates per-walk).
    let r_collect = Region::new(&GLOBAL);
    let walks = walk::generate_walks_ref_from_nodes(&g, &start_nodes, config);
    let s_collect = r_collect.change();
    assert_eq!(walks.len(), n * config.walks_per_node);

    // Streaming (should allocate much less; we don't store walks).
    let r_stream = Region::new(&GLOBAL);
    let mut count = 0usize;
    walk::generate_walks_ref_streaming_from_nodes(&g, &start_nodes, config, |_w| {
        count += 1;
    });
    let s_stream = r_stream.change();
    assert_eq!(count, n * config.walks_per_node);

    // This is intentionally coarse: exact allocation counts vary by allocator/platform.
    // We care about the qualitative guarantee: streaming should not allocate O(#walks).
    //
    // If this flakes in practice, we can tighten the streaming implementation further
    // or switch this to assert on bytes, not counts.
    let a_collect = s_collect.allocations;
    let a_stream = s_stream.allocations;

    assert!(
        a_collect > a_stream,
        "expected collecting allocations > streaming allocations (collect={a_collect}, stream={a_stream})"
    );

    // Heuristic guardrail: streaming should be at least 10x fewer allocations.
    assert!(
        a_stream * 10 < a_collect,
        "expected streaming allocations << collecting allocations (collect={a_collect}, stream={a_stream})"
    );
}

