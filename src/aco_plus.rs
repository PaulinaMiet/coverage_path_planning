// ACO+ (Improved Ant Colony Optimisation)

use crate::shared::*;
use rand::Rng;
use std::collections::{HashSet, VecDeque};


#[derive(Clone)]
pub struct AcoConfig {
    pub n_ants:          usize,
    pub n_iterations:    usize,
    pub solution_length: usize,

    pub alpha:     f64,   // pheromone influence
    pub beta:      f64,   // heuristic influence
    pub rho:       f64,   // global evaporation rate (0–1)
    pub local_rho: f64,   // ACS local update strength ξ — pulls used edges toward τ₀
    pub q0:        f64,   // exploitation probability (ACS-style)
    pub tabu_size: usize, // short-term memory — blocks last N visited cells


    pub restart_after: usize,
}

impl AcoConfig {
    pub fn default_for(grid: &Grid) -> Self {
        AcoConfig {
            n_ants:          100,
            n_iterations:    1500,
            solution_length: free_cells(grid) * 2,
            alpha:           1.0,
            beta:            5.0,
            rho:             0.05,
            local_rho:       0.1,
            q0:              0.9,
            tabu_size:       6,
            restart_after:   250,
        }
    }

    pub fn to_json_map(&self) -> String {
        format!(
            "\"n_ants\": {}, \"n_iterations\": {}, \"alpha\": {}, \"beta\": {}, \
             \"rho\": {}, \"local_rho\": {}, \"q0\": {}, \"restart_after\": {}",
            self.n_ants, self.n_iterations, self.alpha, self.beta,
            self.rho, self.local_rho, self.q0, self.restart_after
        )
    }
}


type Pheromone = Vec<Vec<[f64; 8]>>;

fn init_pheromone(grid: &Grid, tau: f64) -> Pheromone {
    vec![vec![[tau; 8]; grid[0].len()]; grid.len()]
}


fn greedy_cost(
    grid:          &Grid,
    neighbour_map: &NeighbourMap,
    cfg:           &AcoConfig,
    rng:           &mut impl Rng,
) -> f64 {
    let greedy_cfg = AcoConfig { q0: 1.0, local_rho: 0.0, ..cfg.clone() };
    let mut uniform = init_pheromone(grid, 1.0);
    let moves = build_solution(grid, &mut uniform, neighbour_map, &greedy_cfg, 1.0, rng);
    evaluate(&decode(&moves, grid), grid).total.max(1.0)
}


fn mmas_bounds(best_cost: f64, n_free: f64, rho: f64) -> (f64, f64) {
    let tau_max = 1.0 / (rho * best_cost);
    let tau_min = tau_max / (2.0 * n_free);
    (tau_min, tau_max)
}



type NeighbourMap = Vec<Vec<Vec<(usize, Position)>>>;

pub fn build_neighbour_map(grid: &Grid) -> NeighbourMap {
    let rows = grid.len();
    let cols = grid[0].len();
    let mut map: NeighbourMap = vec![vec![Vec::new(); cols]; rows];


    let deltas: [(isize, isize); 8] = [
        (-1,  0), ( 1,  0), ( 0, -1), ( 0,  1),
        (-1, -1), (-1,  1), ( 1, -1), ( 1,  1),
    ];

    for r in 0..rows {
        for c in 0..cols {
            if grid[r][c] == 1 { continue; }
            for (mv_idx, &(dr, dc)) in deltas.iter().enumerate() {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if is_free(nr, nc, grid) {
                    map[r][c].push((mv_idx, (nr as usize, nc as usize)));
                }
            }
        }
    }

    map
}


fn flood_fill_check(
    grid: &Grid,
    start: Position,
) -> std::result::Result<usize, usize> {
    let total = free_cells(grid);
    let mut visited: HashSet<Position> = HashSet::new();
    let mut queue: VecDeque<Position> = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    let deltas: [(isize, isize); 8] = [
        (-1,  0), ( 1,  0), ( 0, -1), ( 0,  1),
        (-1, -1), (-1,  1), ( 1, -1), ( 1,  1),
    ];

    while let Some(pos) = queue.pop_front() {
        for (dr, dc) in deltas {
            let nr = pos.0 as isize + dr;
            let nc = pos.1 as isize + dc;
            if is_free(nr, nc, grid) {
                let npos = (nr as usize, nc as usize);
                if visited.insert(npos) { queue.push_back(npos); }
            }
        }
    }

    let reachable = visited.len();
    if reachable == total { Ok(reachable) } else { Err(total - reachable) }
}



fn coverage_gain(pos: Position, mv: Move, grid: &Grid, visited: &HashSet<Position>) -> f64 {
    let dest_cells = apply_move(pos, mv, grid);
    if dest_cells.is_empty() { return 1.0; }

    let dest = dest_cells[0];
    let base = if !visited.contains(&dest) { 1.0 } else { 0.0 };

    let cost = match mv {
        Move::UpLeft | Move::UpRight | Move::DownLeft | Move::DownRight =>
            std::f64::consts::SQRT_2,
        _ => 1.0,
    };

    (base + 1.0) / cost
}


fn build_solution(
    grid:          &Grid,
    pheromone:     &mut Pheromone,
    neighbour_map: &NeighbourMap,
    cfg:           &AcoConfig,
    tau_min:       f64,
    rng:           &mut impl Rng,
) -> Vec<Move> {
    let mut moves   = Vec::with_capacity(cfg.solution_length);
    let mut pos     = START;
    let mut visited: HashSet<Position> = HashSet::new();
    visited.insert(pos);

    let mut tabu: VecDeque<Position> = VecDeque::new();

    let total_free = free_cells(grid);

    for _ in 0..cfg.solution_length {
        if visited.len() == total_free { break; }

        let (r, c) = pos;

        let options = &neighbour_map[r][c];
        if options.is_empty() { break; }

        let weights: Vec<f64> = options.iter()
            .map(|&(mv_idx, dest)| {
                if tabu.contains(&dest) {
                    return 0.0;
                }
                let mv = ALL_MOVES[mv_idx];
                pheromone[r][c][mv_idx].powf(cfg.alpha)
                    * coverage_gain(pos, mv, grid, &visited).powf(cfg.beta)
            })
            .collect();


        let all_tabu = weights.iter().all(|&w| w == 0.0);
        let effective: Vec<f64> = if all_tabu {
            options.iter()
                .map(|&(mv_idx, _)| {
                    let mv = ALL_MOVES[mv_idx];
                    pheromone[r][c][mv_idx].powf(cfg.alpha)
                        * coverage_gain(pos, mv, grid, &visited).powf(cfg.beta)
                })
                .collect()
        } else {
            weights
        };

        let chosen_idx = if rng.r#gen::<f64>() < cfg.q0 {
            effective.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap()
        } else {
            let total: f64 = effective.iter().sum();
            let mut pick   = rng.r#gen::<f64>() * total;
            let mut chosen = options.len() - 1;
            for (i, &w) in effective.iter().enumerate() {
                pick -= w;
                if pick <= 0.0 { chosen = i; break; }
            }
            chosen
        };

        let (mv_idx, dest) = options[chosen_idx];
        let mv = ALL_MOVES[mv_idx];
        moves.push(mv);

        // ACS local update: τ ← (1−ξ)·τ + ξ·τ_min

        pheromone[r][c][mv_idx] =
            (1.0 - cfg.local_rho) * pheromone[r][c][mv_idx] + cfg.local_rho * tau_min;

        let new_cells = apply_move(pos, mv, grid);
        if let Some(&last) = new_cells.last() { pos = last; }
        for p in new_cells { visited.insert(p); }

        tabu.push_back(dest);
        if tabu.len() > cfg.tabu_size {
            tabu.pop_front();
        }
    }

    moves
}


pub fn run(grid: &Grid, cfg: &AcoConfig) -> Result {
    match flood_fill_check(grid, START) {
        Ok(n)  => println!("[ACO+] flood-fill OK — {} free cells all connected", n),
        Err(n) => println!("[ACO+] WARNING: {} free cells unreachable from start", n),
    }

    let neighbour_map = build_neighbour_map(grid);

    let mut rng = rand::thread_rng();


    let n_free = free_cells(grid) as f64;
    let c_greedy = greedy_cost(grid, &neighbour_map, cfg, &mut rng);
    let (mut tau_min, mut tau_max) = mmas_bounds(c_greedy, n_free, cfg.rho);
    println!("[ACO+] C_greedy = {:.2}  tau_max = {:.4e}  tau_min = {:.4e}", c_greedy, tau_max, tau_min);


    let mut pheromone = init_pheromone(grid, tau_max);
    let mut best_moves: Vec<Move> = Vec::new();
    let mut best_fit  = f64::MAX;
    let mut history   = Vec::new();
    let mut stagnant  = 0usize;


    const GB_EVERY:  usize = 5;
    const GB_WARMUP: usize = 50;
    let mut since_restart = 0usize;

    for iter in 0..cfg.n_iterations {
        let mut iter_best: Option<(Vec<Move>, Fitness)> = None;

        for _ in 0..cfg.n_ants {
            let moves   = build_solution(grid, &mut pheromone, &neighbour_map, cfg, tau_min, &mut rng);
            let fitness = evaluate(&decode(&moves, grid), grid);
            if iter_best.as_ref().map_or(true, |(_, b)| fitness.total < b.total) {
                iter_best = Some((moves, fitness));
            }
        }


        for row in &mut pheromone {
            for cell in row {
                for tau in cell.iter_mut() {
                    *tau = (*tau * (1.0 - cfg.rho)).clamp(tau_min, tau_max);
                }
            }
        }

        if let Some((mvs, fit)) = iter_best {
            if fit.total < best_fit {
                best_fit   = fit.total;
                best_moves = mvs.clone();
                stagnant   = 0;
                (tau_min, tau_max) = mmas_bounds(best_fit, n_free, cfg.rho);
            } else {
                stagnant += 1;
            }


            let use_gb = !best_moves.is_empty()
                && since_restart >= GB_WARMUP
                && iter % GB_EVERY == 0;
            let (trail, cost): (&[Move], f64) = if use_gb {
                (&best_moves, best_fit)
            } else {
                (&mvs, fit.total)
            };
            let deposit = 1.0 / cost;
            let mut pos = START;
            for &mv in trail {
                let mv_idx = ALL_MOVES.iter().position(|&m| m == mv).unwrap();
                pheromone[pos.0][pos.1][mv_idx] =
                    (pheromone[pos.0][pos.1][mv_idx] + deposit).min(tau_max);
                let new_cells = apply_move(pos, mv, grid);
                if let Some(&last) = new_cells.last() { pos = last; }
            }

            history.push(IterationLog {
                iteration: iter,
                fitness:   fit.total,
                distance:  fit.distance,
                revisits:  fit.revisits,
                unvisited: fit.unvisited,
                moves:     mvs,
            });
        }
        since_restart += 1;

        if cfg.restart_after > 0 && stagnant >= cfg.restart_after {
            for row in &mut pheromone {
                for cell in row {
                    for tau in cell.iter_mut() { *tau = tau_max; }
                }
            }
            stagnant = 0;
            since_restart = 0;
            println!("[ACO+] stagnation restart at iteration {}", iter);
        }
    }

    let best_fitness = evaluate(&decode(&best_moves, grid), grid);
    Result { best_moves, best_fitness, history }
}