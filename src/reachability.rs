//! Reachability helpers (transitive closure counts).
//!
//! This module is intentionally small and allocation-light:
//! - Build adjacency lists once.
//! - Use a "visited stamp" (`Vec<u32>`) to avoid re-allocating `seen` for every start node.
//!
//! Edges are interpreted as `u -> v` (directed).

/// Count transitive reachability for each node in a directed graph.
///
/// `edges` are `u -> v` edges.
///
/// Returns `(dependents, dependencies)` where:
/// - `dependencies[u]` is the number of distinct nodes reachable from `u` following `u -> v`
/// - `dependents[u]` is the number of distinct nodes that can reach `u` (reachability in the
///   reversed graph)
///
/// Notes:
/// - Counts do **not** include the start node itself.
/// - This is \(O(n (n + m))\) in the worst case; it’s intended for graphs up to a few thousand
///   nodes where “blast radius” style counts are useful.
pub fn reachability_counts_edges(n: usize, edges: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
    let mut fwd: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut rev: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        if u >= n || v >= n {
            // Out-of-range edges are ignored (callers should validate, but be robust).
            continue;
        }
        fwd[u].push(v);
        rev[v].push(u);
    }

    let mut dependencies = vec![0usize; n];
    let mut dependents = vec![0usize; n];

    // One visited buffer reused for all BFS runs.
    let mut visited: Vec<u32> = vec![0u32; n];
    let mut stamp: u32 = 0;
    let mut q: Vec<usize> = Vec::new();

    for start in 0..n {
        // Forward reachability (dependencies)
        stamp = stamp.wrapping_add(1);
        q.clear();
        visited[start] = stamp; // exclude `start` from counts even if cycles return to it
        q.push(start);
        let mut head = 0usize;
        let mut count = 0usize;
        while head < q.len() {
            let cur = q[head];
            head += 1;
            for &nx in &fwd[cur] {
                if visited[nx] != stamp {
                    visited[nx] = stamp;
                    q.push(nx);
                    count += 1;
                }
            }
        }
        dependencies[start] = count;

        // Reverse reachability (dependents)
        stamp = stamp.wrapping_add(1);
        q.clear();
        visited[start] = stamp; // exclude `start` from counts even if cycles return to it
        q.push(start);
        let mut head = 0usize;
        let mut count = 0usize;
        while head < q.len() {
            let cur = q[head];
            head += 1;
            for &nx in &rev[cur] {
                if visited[nx] != stamp {
                    visited[nx] = stamp;
                    q.push(nx);
                    count += 1;
                }
            }
        }
        dependents[start] = count;
    }

    (dependents, dependencies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reachability_counts_edges_small() {
        // 0 -> 1 -> 2, 0 -> 2, and 3 isolated
        let n = 4;
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let (dependents, dependencies) = reachability_counts_edges(n, &edges);

        // dependencies
        assert_eq!(dependencies[0], 2); // {1,2}
        assert_eq!(dependencies[1], 1); // {2}
        assert_eq!(dependencies[2], 0);
        assert_eq!(dependencies[3], 0);

        // dependents (reverse reachability)
        assert_eq!(dependents[2], 2); // {0,1}
        assert_eq!(dependents[1], 1); // {0}
        assert_eq!(dependents[0], 0);
        assert_eq!(dependents[3], 0);
    }

    #[test]
    fn test_reachability_does_not_count_start_via_cycle() {
        // 0 -> 1 -> 2 -> 0 (cycle)
        let n = 3;
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let (dependents, dependencies) = reachability_counts_edges(n, &edges);

        // From any node, the reachable set is the other two nodes; the start node must not be counted.
        assert_eq!(dependencies[0], 2);
        assert_eq!(dependencies[1], 2);
        assert_eq!(dependencies[2], 2);

        assert_eq!(dependents[0], 2);
        assert_eq!(dependents[1], 2);
        assert_eq!(dependents[2], 2);
    }
}

