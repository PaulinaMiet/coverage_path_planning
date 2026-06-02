use rand::Rng;

use crate::shared::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

// Config ─────────────────────────────────────────────────────

pub struct IlsConfig {
    pub n_iterations: usize,  // Total ILS cycles
    pub ls_iterations: usize, // Local search attempts per cycle
    pub perturb_size: usize,  // How many moves to change during perturbation
}

impl IlsConfig {
    pub fn default_for(grid: &Grid) -> Self {
        IlsConfig {
            n_iterations: 100,
            ls_iterations: 150,
            perturb_size: 3,
        }
    }
    pub fn to_json_map(&self) -> String {
        format!(
            "\"n_iterations\": {}, \"ls_iterations\": {}, \"perturb_size\": {}",
            self.n_iterations, self.ls_iterations, self.perturb_size
        )
    }
}

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

// ── Main ILS loop ──────────────────────────────────────────────

pub fn ils_run(grid: &Grid, cfg: &IlsConfig) -> Result {
    let mst = compute_mst(grid);
    let mut current_moves = mst_to_moves(grid, &mst);

    let mut best_moves = current_moves.clone();
    let mut best_fitness = evaluate(&decode(&best_moves, grid), grid);
    let mut history = Vec::new();

    for i in 0..cfg.n_iterations {
        local_search(&mut current_moves, grid, cfg.ls_iterations);

        let fitness = evaluate(&decode(&current_moves, grid), grid);

        if fitness.total < best_fitness.total {
            best_fitness = fitness;
            best_moves = current_moves.clone();
        }

        history.push(IterationLog {
            iteration: i,
            fitness: best_fitness.total,
            distance: best_fitness.distance,
            revisits: best_fitness.revisits,
            unvisited: best_fitness.unvisited,
            moves: best_moves.clone(),
        });
        perturb(&mut current_moves, cfg.perturb_size);
    }

    Result {
        best_moves,
        best_fitness,
        history,
    }
}

fn positions_to_moves(path: &[Position]) -> Vec<Move> {
    let mut moves = Vec::new();
    for window in path.windows(2) {
        let (r1, c1) = window[0];
        let (r2, c2) = window[1];

        let mv = match (r2 as isize - r1 as isize, c2 as isize - c1 as isize) {
            (-1, 0) => Move::Up,
            (1, 0) => Move::Down,
            (0, -1) => Move::Left,
            (0, 1) => Move::Right,
            _ => continue,
        };
        moves.push(mv);
    }

    moves
}

pub fn mst_to_moves(grid: &Grid, mst: &[Edge]) -> Vec<Move> {
    let mut adj: HashMap<Position, Vec<(usize, usize)>> = HashMap::new(); // maps a point to a list of its neighbors

    for edge in mst {
        // let u_list = adj.entry(edge.u).or_insert(Vec::new());
        // u_list.push(edge.v);
        adj.entry(edge.u).or_insert(Vec::new()).push(edge.v);
        adj.entry(edge.v).or_insert(Vec::new()).push(edge.u);
    }

    let mut path = Vec::new();
    let mut visited: HashSet<Position> = HashSet::new();

    fn dfs(
        u: Position,
        adj: &HashMap<Position, Vec<(usize, usize)>>,
        visited: &mut HashSet<Position>,
        path: &mut Vec<Position>,
    ) {
        visited.insert(u);
        path.push(u);
        if let Some(neighbors) = adj.get(&u) {
            for &v in neighbors {
                if !visited.contains(&v) {
                    dfs(v, adj, visited, path);
                    // After visiting a branch, we return to 'u' to continue to other branches
                    path.push(u);
                }
            }
        }
    }

    dfs(START, &adj, &mut visited, &mut path);

    positions_to_moves(&path)
}

fn local_search(moves: &mut Vec<Move>, grid: &Grid, imp: usize) {
    let mut rng = rand::thread_rng();
    let mut current_fitness = evaluate(&decode(moves, grid), grid).total;

    // try n random improvements
    for _ in 0..imp {
        let idx = rng.gen_range(0..moves.len());
        let old_move = moves[idx];
        let new_move = ALL_MOVES[rng.gen_range(0..ALL_MOVES.len())];

        moves[idx] = new_move;
        let new_fitness = evaluate(&decode(moves, grid), grid).total;

        if new_fitness < current_fitness {
            current_fitness = new_fitness;
        } else {
            moves[idx] = old_move; //revert if not better
        }
    }
}

fn perturb(moves: &mut Vec<Move>, perturb_size: usize) {
    let mut rng = rand::thread_rng();
    let start_idx = rng.gen_range(0..moves.len().saturating_sub(perturb_size));

    for i in 0..perturb_size {
        moves[start_idx + i] = ALL_MOVES[rng.gen_range(0..ALL_MOVES.len())]
    }
}
