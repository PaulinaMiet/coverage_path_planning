

use crate::shared::*;
use rand::Rng;
use std::collections::HashSet;

//  Config 

pub struct AcoConfig {
    pub n_ants:          usize,
    pub n_iterations:    usize,
    pub solution_length: usize,   //  rows      +    cols       × 2
                                //e.g. (  10      +     10      ) × 2  =  40 moves  ← for 10×10 grid


    pub alpha:  f64,   // pheromone influence
    pub beta: f64,   // heuristic influence
    pub rho:  f64,   // evaporation rate (0–1)
    pub tau_init: f64,   // starting pheromone on every edge
    pub q0: f64,   // probability to exploit best move (ACS-style)
}

impl AcoConfig {
    pub fn default_for(grid: &Grid) -> Self {
        AcoConfig {
            n_ants:          50,
            n_iterations:    1500,
            solution_length: free_cells(grid) * 2,
            alpha:           1.0,
            beta:            3.0,
            rho:             0.05,
            tau_init:        1.0,
            q0:              0.9,
        }
    }
}

//  Per-iteration log ─────────────────────────────────────────

// Best ant of each iteration , used for CSV export 
pub struct IterationLog {
    pub iteration: usize,
    pub fitness:   f64,
    pub distance:  f64,
    pub revisits:  usize,
    pub unvisited: usize,
    pub moves:     Vec<Move>,
}

// ── Result ────────────────────────────────────────────────────

pub struct AcoResult {
    pub best_moves:   Vec<Move>,
    pub best_fitness: Fitness,
    pub history:      Vec<IterationLog>, // one entry per iteration
}

// ── Pheromone ─────────────────────────────────────────────────

// τ[row][col][move_idx]  how good is move m when standing at cell (r,c)?
type Pheromone = Vec<Vec<[f64; 8]>>;

fn init_pheromone(grid: &Grid, tau: f64) -> Pheromone {
    vec![vec![[tau; 8]; grid[0].len()]; grid.len()]
}

// ── Heuristic ─────────────────────────────────────────────────

// New free cells reachable by move m from pos. +1 avoids zero weight
fn coverage_gain(pos: Pos, mv: Move, grid: &Grid, visited: &HashSet<Pos>) -> f64 {
    apply(pos, mv, grid)
        .iter()
        .filter(|p| !visited.contains(p))
        .count() as f64
        + 1.0
}

// ── Ant construction ──────────────────────────────────────────

fn build_solution(
    grid:      &Grid,
    pheromone: &Pheromone,
    cfg:       &AcoConfig,
    rng:       &mut impl Rng,
) -> Vec<Move> {
    let mut moves   = Vec::with_capacity(cfg.solution_length);
    let mut pos     = (0usize, 0usize);
    let mut visited = HashSet::new();
    visited.insert(pos);

    for _ in 0..cfg.solution_length {
        let (r, c) = pos;

        let weights: Vec<f64> = (0..8)
            .map(|i| {
                pheromone[r][c][i].powf(cfg.alpha)
                    * coverage_gain(pos, ALL_MOVES[i], grid, &visited).powf(cfg.beta)
            })
            .collect();

        let mv_idx = if rng.r#gen::<f64>() < cfg.q0 {
            // exploit: take highest score
            weights.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap()
        } else {
            // explore: roulette wheel
            let total: f64 = weights.iter().sum();
            let mut pick   = rng.r#gen::<f64>() * total;
            let mut chosen = 7;
            for (i, &w) in weights.iter().enumerate() {
                pick -= w;
                if pick <= 0.0 { chosen = i; break; }
            }
            chosen
        };

        let mv = ALL_MOVES[mv_idx];
        moves.push(mv);
        let new_cells = apply(pos, mv, grid);
        if let Some(&last) = new_cells.last() { pos = last; }
        for p in new_cells { visited.insert(p); }
    }

    moves
}


//  Main ACO loop //

pub fn run(grid: &Grid, cfg: &AcoConfig) -> AcoResult {
    let mut rng       = rand::thread_rng();
    let mut pheromone = init_pheromone(grid, cfg.tau_init);
    let mut best_moves: Vec<Move> = Vec::new();
    let mut best_fit  = f64::MAX;
    let mut history   = Vec::new();

    for iter in 0..cfg.n_iterations {
        // Build solutions, track best ant this iteration (full fitness)
        let mut iter_best: Option<(Vec<Move>, Fitness)> = None;

        for _ in 0..cfg.n_ants {
            let moves   = build_solution(grid, &pheromone, cfg, &mut rng);
            let fitness = evaluate(&decode(&moves, grid, (0, 0)), grid);
            if iter_best.as_ref().map_or(true, |(_, b)| fitness.total < b.total) {
                iter_best = Some((moves, fitness));
            }
        }

        // Evaporate
        for row in &mut pheromone {
            for cell in row {
                for tau in cell.iter_mut() { *tau *= 1.0 - cfg.rho; }
            }
        }

        if let Some((mvs, fit)) = iter_best {
            // Deposit on best ant's trail
            let deposit = 1.0 / fit.total;
            let mut pos = (0usize, 0usize);
            for &mv in &mvs {
                let mv_idx = ALL_MOVES.iter().position(|&m| m == mv).unwrap();
                pheromone[pos.0][pos.1][mv_idx] += deposit;
                let new_cells = apply(pos, mv, grid);
                if let Some(&last) = new_cells.last() { pos = last; }
            }

            if fit.total < best_fit {
                best_fit   = fit.total;
                best_moves = mvs.clone();
            }

            // Log every iteration
            history.push(IterationLog {
                iteration: iter,
                fitness:   fit.total,
                distance:  fit.distance,
                revisits:  fit.revisits,
                unvisited: fit.unvisited,
                moves:     mvs,
            });
        }
    }

    let best_fitness = evaluate(&decode(&best_moves, grid, (0, 0)), grid);
    AcoResult { best_moves, best_fitness, history }
}
