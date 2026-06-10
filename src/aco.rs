use crate::shared::*;
use rand::Rng;
use std::collections::HashSet;

pub struct AcoConfig {
    pub n_ants: usize,
    pub n_iterations: usize,
    pub solution_length: usize, // (rows + cols) * 2
    pub alpha: f64,    // pheromone weight
    pub beta: f64,     // heuristic weight
    pub rho: f64,      // evaporation rate
    pub tau_init: f64, // initial pheromone value
    pub q0: f64,       // exploitation probability
}

impl AcoConfig {
    pub fn default_for(grid: &Grid) -> Self {
        AcoConfig {
            n_ants: 50,
            n_iterations: 1500,
            solution_length: free_cells(grid) * 2,
            alpha: 1.0,
            beta: 3.0,
            rho: 0.05,
            tau_init: 1.0,
            q0: 0.9,
        }
    }
    pub fn to_json_map(&self) -> String {
        format!(
            "\"n_ants\": {}, \"n_iterations\": {}, \"alpha\": {}, \"beta\": {}, \"rho\": {}",
            self.n_ants, self.n_iterations, self.alpha, self.beta, self.rho
        )
    }
}

// tau[row][col][move_idx]: pheromone strength for each move at each cell
type Pheromone = Vec<Vec<[f64; 8]>>;

fn init_pheromone(grid: &Grid, tau: f64) -> Pheromone {
    vec![vec![[tau; 8]; grid[0].len()]; grid.len()]
}

// +1 avoids zero weight when no new cells are reachable
fn coverage_gain(pos: Position, mv: Move, grid: &Grid, visited: &HashSet<Position>) -> f64 {
    apply_move(pos, mv, grid)
        .iter()
        .filter(|p| !visited.contains(p))
        .count() as f64
        + 1.0
}

fn build_solution(
    grid: &Grid,
    pheromone: &Pheromone,
    cfg: &AcoConfig,
    rng: &mut impl Rng,
) -> Vec<Move> {
    let mut moves = Vec::with_capacity(cfg.solution_length);
    let mut pos = (0usize, 0usize);
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
            // exploit: pick highest score
            weights
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        } else {
            // explore: roulette wheel selection
            let total: f64 = weights.iter().sum();
            let mut pick = rng.r#gen::<f64>() * total;
            let mut chosen = 7;
            for (i, &w) in weights.iter().enumerate() {
                pick -= w;
                if pick <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            chosen
        };

        let mv = ALL_MOVES[mv_idx];
        moves.push(mv);
        let new_cells = apply_move(pos, mv, grid);
        if let Some(&last) = new_cells.last() {
            pos = last;
        }
        for p in new_cells {
            visited.insert(p);
        }
    }

    moves
}

pub fn aco_run(grid: &Grid, cfg: &AcoConfig) -> Result {
    let mut rng = rand::thread_rng();
    let mut pheromone = init_pheromone(grid, cfg.tau_init);
    let mut best_moves: Vec<Move> = Vec::new();
    let mut best_fit = f64::MAX;
    let mut history = Vec::new();

    for iter in 0..cfg.n_iterations {
        let mut iter_best: Option<(Vec<Move>, Fitness)> = None;

        for _ in 0..cfg.n_ants {
            let moves = build_solution(grid, &pheromone, cfg, &mut rng);
            let fitness = evaluate(&decode(&moves, grid), grid);
            if iter_best
                .as_ref()
                .map_or(true, |(_, b)| fitness.total < b.total)
            {
                iter_best = Some((moves, fitness));
            }
        }

        // evaporate
        for row in &mut pheromone {
            for cell in row {
                for tau in cell.iter_mut() {
                    *tau *= 1.0 - cfg.rho;
                }
            }
        }

        if let Some((mvs, fit)) = iter_best {
            // deposit on best ant's path
            let deposit = 1.0 / fit.total;
            let mut pos = (0usize, 0usize);
            for &mv in &mvs {
                let mv_idx = ALL_MOVES.iter().position(|&m| m == mv).unwrap();
                pheromone[pos.0][pos.1][mv_idx] += deposit;
                let new_cells = apply_move(pos, mv, grid);
                if let Some(&last) = new_cells.last() {
                    pos = last;
                }
            }

            if fit.total < best_fit {
                best_fit = fit.total;
                best_moves = mvs.clone();
            }

            history.push(IterationLog {
                iteration: iter,
                fitness: fit.total,
                distance: fit.distance,
                revisits: fit.revisits,
                unvisited: fit.unvisited,
                moves: mvs,
            });
        }
    }

    let best_fitness = evaluate(&decode(&best_moves, grid), grid);
    Result {
        best_moves,
        best_fitness,
        history,
    }
}
