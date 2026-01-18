# walk

Graph random-walk primitives:

- Unbiased random walks
- Node2Vec / Node2Vec-style biased walks (including precomputed alias tables)
- PageRank and personalized PageRank

## Determinism

Most APIs take a `WalkConfig { seed, .. }`. For fixed `seed` and identical inputs,
walk generation is intended to be reproducible.

## Graph interfaces

- `Graph`: returns owned neighbor lists (`Vec<usize>`) per query (simple but allocates per step)
- `GraphRef`: returns borrowed neighbor slices (`&[usize]`) per query (faster for walking)

## Sampling integration

This crate uses `kuji` for reservoir sampling via `sample_start_nodes_reservoir`, which is useful
when `node_count` is too large to materialize `0..node_count` just to choose a subset of starts.

## References (starting points)

- Page et al. (1999): PageRank.
- Grover & Leskovec (2016): Node2Vec.
- Walker (1974) / Vose (1991): alias method for O(1) categorical sampling.

