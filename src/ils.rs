use crate::shared::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::num::Wrapping;

// Config ─────────────────────────────────────────────────────

pub struct IlsConfig {}

impl IlsConfig {
    pub fn default_for(grid: &Grid) -> Self {
        IlsConfig {}
    }
}

//  Per-iteration log ─────────────────────────────────────────

// Best solution of each iteration , used for CSV export
pub struct IterationLog {
    pub iteration: usize,
    pub fitness: f64,
    pub distance: f64,
    pub revisits: usize,
    pub unvisited: usize,
    pub moves: Vec<Move>,
}

// ── Result ────────────────────────────────────────────────────

pub struct IlsResult {
    pub best_moves: Vec<Move>,
    pub best_fitness: Fitness,
    pub history: Vec<IterationLog>, // one entry per iteration
}

// ── Main ILS loop ──────────────────────────────────────────────

pub fn ils_run(grid: &Grid, cfg: &IlsConfig) {}

// ── Minimal Spanning Tree ──────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Edge {
    pub u: Position,
    pub v: Position,
    pub weight: u32,
}

/// Computes an MST of all free cells using Prim's algorithm.
pub fn compute_mst(grid: &Grid) -> Vec<Edge> {
    let mut mst = Vec::new();
    let mut visited = HashSet::new();
    let mut pq = BinaryHeap::new();

    // Find the first free cell to start from (ideally START if it's free)
    let start_node = if is_free(START.0 as isize, START.1 as isize, grid) {
        Some(START)
    } else {
        find_first_free_cell(grid)
    };

    let Some(start) = start_node else {
        return mst;
    };

    visited.insert(start);
    add_neighbors(start, grid, &visited, &mut pq);

    while let Some(Reverse((weight, u, v))) = pq.pop() {
        if visited.contains(&v) {
            continue;
        }

        visited.insert(v);
        mst.push(Edge { u, v, weight });
        add_neighbors(v, grid, &visited, &mut pq);
    }

    mst
}

fn find_first_free_cell(grid: &Grid) -> Option<Position> {
    for (r, row) in grid.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            if val == 0 {
                return Some((r, c));
            }
        }
    }
    None
}

fn add_neighbors(
    pos: Position,
    grid: &Grid,
    visited: &HashSet<Position>,
    pq: &mut BinaryHeap<Reverse<(u32, Position, Position)>>,
) {
    let (r, c) = (pos.0 as isize, pos.1 as isize);
    let dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for (dr, dc) in dirs {
        let nr = r + dr;
        let nc = c + dc;
        if is_free(nr, nc, grid) {
            let npos = (nr as usize, nc as usize);
            if !visited.contains(&npos) {
                pq.push(Reverse((1, pos, npos)))
            }
        }
    }
}
