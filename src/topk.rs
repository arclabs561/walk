//! Ranking utilities.

use ordered_float::NotNan;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub fn top_k(scores: &[f64], k: usize) -> Vec<(usize, f64)> {
    if k == 0 || scores.is_empty() { return Vec::new(); }
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for (i, &score) in scores.iter().enumerate() {
        if !score.is_finite() || score <= 0.0 { continue; }
        let s = NotNan::new(score).unwrap();
        if heap.len() < k {
            heap.push(Reverse((s, i)));
        } else if let Some(&Reverse((min_score, _))) = heap.peek() {
            if s > min_score {
                heap.pop();
                heap.push(Reverse((s, i)));
            }
        }
    }
    let mut results: Vec<(usize, f64)> = heap.into_iter().map(|Reverse((s, i))| (i, s.into_inner())).collect();
    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

pub fn normalize(scores: &mut [f64]) {
    let sum: f64 = scores.iter().sum();
    if sum > 0.0 {
        for s in scores { *s /= sum; }
    }
}
