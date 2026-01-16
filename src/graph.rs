//! Minimal graph adapter traits.

pub trait Graph {
    fn node_count(&self) -> usize;
    fn neighbors(&self, node: usize) -> Vec<usize>;
    fn out_degree(&self, node: usize) -> usize {
        self.neighbors(node).len()
    }
}

pub trait WeightedGraph: Graph {
    fn edge_weight(&self, source: usize, target: usize) -> f64;
}

pub struct AdjacencyMatrix<'a>(pub &'a [Vec<f64>]);

impl<'a> Graph for AdjacencyMatrix<'a> {
    fn node_count(&self) -> usize {
        self.0.len()
    }
    fn neighbors(&self, node: usize) -> Vec<usize> {
        self.0[node].iter().enumerate().filter(|(_, &w)| w > 0.0).map(|(i, _)| i).collect()
    }
}

impl<'a> WeightedGraph for AdjacencyMatrix<'a> {
    fn edge_weight(&self, source: usize, target: usize) -> f64 {
        self.0[source][target]
    }
}

#[cfg(feature = "petgraph")]
impl<N, E, Ty, Ix> Graph for petgraph::Graph<N, E, Ty, Ix>
where
    Ty: petgraph::EdgeType,
    Ix: petgraph::graph::IndexType,
{
    fn node_count(&self) -> usize {
        self.node_count()
    }
    fn neighbors(&self, node: usize) -> Vec<usize> {
        self.neighbors(petgraph::graph::NodeIndex::new(node)).map(|idx| idx.index()).collect()
    }
}
