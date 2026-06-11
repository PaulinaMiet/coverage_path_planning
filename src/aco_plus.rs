// aco_plus.rs — ACO+ (Improved Ant Colony Optimisation)
// 20260607
//
// Revision 20260611 — ACS-grounded pheromone management:
//   * local update is now τ ← (1−ξ)τ + ξτ₀ (was pure decay τ ← (1−ξ)τ,
//     which erased trails toward zero and made ACO+ underperform base ACO
//     on 10×10 instances)
//   * τ₀ auto-computed as 1/(n_free · C_greedy) from one greedy construction,
//     replacing the fixed tau_init = 1.0 that swamped the 1/fitness deposits
//   * global evaporation clamped below at τ₀ so unexplored edges stay reachable
//
// Revision 20260611b — MMAS-style adaptive pheromone bounds (Stützle & Hoos):
//   The fixed τ₀ floor left the exploit/explore ratio unbounded (deposits
//   ~10⁴× above the floor), so the colony committed to one basin within a few
//   iterations — the 20×20 stagnation signature. Now:
//   * τ_max = 1/(ρ · C_best), recomputed whenever the best-so-far improves
//   * τ_min = τ_max / (2 · n_free) — caps the exploit/explore ratio at ~2n
//   * pheromone is initialised AND restart-reset to τ_max (maximum
//     exploration), and clamped into [τ_min, τ_max] every iteration
//   * the ACS local update now pulls used edges toward τ_min
//   * global-best deposit schedule: every 5th iteration (after a 50-iteration
//     post-restart warmup) the best-so-far deposits instead of the
//     iteration-best, so restart epochs refine the incumbent

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

    // MMAS-style stagnation restart: if best-so-far has not improved for this
    // many iterations, reset all pheromone to τ₀ (keeping the incumbent best).
    // 0 disables the restart.
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
            local_rho:       0.1,   // ACS-canonical ξ (Dorigo & Gambardella 1997)
            q0:              0.9,   // high exploitation balances local-update diversification
            tabu_size:       6,
            restart_after:   250,   // stagnation restart; set 0 to disable
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

// ── Pheromone ─────────────────────────────────────────────────

// τ[row][col][move_idx] — 8 directions matching ALL_MOVES index contract
type Pheromone = Vec<Vec<[f64; 8]>>;

fn init_pheromone(grid: &Grid, tau: f64) -> Pheromone {
    vec![vec![[tau; 8]; grid[0].len()]; grid.len()]
}

/// Cost of one heuristic-only greedy construction. Seeds the MMAS bounds
/// before any colony solution exists.
fn greedy_cost(
    grid:          &Grid,
    neighbour_map: &NeighbourMap,
    cfg:           &AcoConfig,
    rng:           &mut impl Rng,
) -> f64 {
    // One greedy ant: uniform pheromone, pure exploitation, no local update
    let greedy_cfg = AcoConfig { q0: 1.0, local_rho: 0.0, ..cfg.clone() };
    let mut uniform = init_pheromone(grid, 1.0);
    let moves = build_solution(grid, &mut uniform, neighbour_map, &greedy_cfg, 1.0, rng);
    evaluate(&decode(&moves, grid), grid).total.max(1.0)
}

/// MMAS bounds (Stützle & Hoos 2000), adapted to this deposit scheme.
/// τ_max is the equilibrium pheromone of an edge deposited every iteration
/// (deposit/ρ = 1/(ρ·C_best)); τ_min keeps every edge selectable, capping
/// the exploit/explore ratio at 2·n_free instead of leaving it unbounded.
fn mmas_bounds(best_cost: f64, n_free: f64, rho: f64) -> (f64, f64) {
    let tau_max = 1.0 / (rho * best_cost);
    let tau_min = tau_max / (2.0 * n_free);
    (tau_min, tau_max)
}

// ── Neighbour map ─────────────────────────────────────────────

// Pre-built once per run. Each cell stores (move_idx, destination) for every
// legal move. Illegal moves are structurally absent — no runtime wall checks.
type NeighbourMap = Vec<Vec<Vec<(usize, Position)>>>;

pub fn build_neighbour_map(grid: &Grid) -> NeighbourMap {
    let rows = grid.len();
    let cols = grid[0].len();
    let mut map: NeighbourMap = vec![vec![Vec::new(); cols]; rows];

    // Deltas in ALL_MOVES order:
    // 0=Up 1=Down 2=Left 3=Right 4=UpLeft 5=UpRight 6=DownLeft 7=DownRight
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

// ── Connectivity check ────────────────────────────────────────

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


// Cost-normalised coverage gain: new coverage per unit of distance.
// A straight move covers 1 new cell for cost 1; a diagonal covers 1 new cell
// for cost √2, so all eight moves now compete on equal footing and straight
// sweeps win when both reach unvisited area.
//
// The previous version added a diagonal "opens up new area" bonus (up to +2),
// which β = 5 amplified into a ~32× preference for diagonals. On 20×20 grids
// this drove diagonal zig-zag paths with ~300 revisits; efficient coverage
// paths are orthogonal sweeps. +1 keeps the weight > 0.
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

// ── Ant construction ──────────────────────────────────────────

fn build_solution(
    grid:          &Grid,
    pheromone:     &mut Pheromone,  // mut: local update is applied in-place
    neighbour_map: &NeighbourMap,
    cfg:           &AcoConfig,
    tau_min:       f64,             // exploration floor — target of the local update
    rng:           &mut impl Rng,
) -> Vec<Move> {
    let mut moves   = Vec::with_capacity(cfg.solution_length);
    let mut pos     = START;
    let mut visited: HashSet<Position> = HashSet::new();
    visited.insert(pos);

    // Short-term tabu window — blocks the last tabu_size destinations
    let mut tabu: VecDeque<Position> = VecDeque::new();

    let total_free = free_cells(grid);

    for _ in 0..cfg.solution_length {
        // Early exit: full coverage reached, no point continuing
        if visited.len() == total_free { break; }

        let (r, c) = pos;

        let options = &neighbour_map[r][c];
        if options.is_empty() { break; }

        // Score each legal neighbour; zero out tabu destinations
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

        // Safety valve: if every neighbour is tabu, lift the restriction
        // so the ant can escape a dead-end corridor
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

        // Select move: exploit (q0) or explore (roulette wheel)
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
        // Pulls the trail just used back toward the exploration floor so
        // subsequent ants in this iteration are nudged away from the same path.
        // Unlike pure decay (τ ← (1−ξ)·τ), this can never erase a trail —
        // heavily used edges converge to τ_min, not zero, preserving the
        // learned signal across iterations.
        pheromone[r][c][mv_idx] =
            (1.0 - cfg.local_rho) * pheromone[r][c][mv_idx] + cfg.local_rho * tau_min;

        // Advance one cell and record coverage
        let new_cells = apply_move(pos, mv, grid);
        if let Some(&last) = new_cells.last() { pos = last; }
        for p in new_cells { visited.insert(p); }

        // Slide tabu window forward
        tabu.push_back(dest);
        if tabu.len() > cfg.tabu_size {
            tabu.pop_front();
        }
    }

    moves
}


pub fn run(grid: &Grid, cfg: &AcoConfig) -> Result {
    // Connectivity check — warn if any free cell is unreachable from (0,0)
    match flood_fill_check(grid, START) {
        Ok(n)  => println!("[ACO+] flood-fill OK — {} free cells all connected", n),
        Err(n) => println!("[ACO+] WARNING: {} free cells unreachable from start", n),
    }

    let neighbour_map = build_neighbour_map(grid);

    let mut rng = rand::thread_rng();

    // MMAS adaptive bounds, seeded from one greedy construction and
    // re-tightened every time the best-so-far improves.
    let n_free = free_cells(grid) as f64;
    let c_greedy = greedy_cost(grid, &neighbour_map, cfg, &mut rng);
    let (mut tau_min, mut tau_max) = mmas_bounds(c_greedy, n_free, cfg.rho);
    println!("[ACO+] C_greedy = {:.2}  tau_max = {:.4e}  tau_min = {:.4e}", c_greedy, tau_max, tau_min);

    // MMAS initialises at τ_max: all edges equally attractive → maximum
    // early exploration, with the bounds preventing early lock-in.
    let mut pheromone = init_pheromone(grid, tau_max);
    let mut best_moves: Vec<Move> = Vec::new();
    let mut best_fit  = f64::MAX;
    let mut history   = Vec::new();
    let mut stagnant  = 0usize; // iterations since best-so-far last improved

    // MMAS global-best deposit schedule: normally the iteration-best ant
    // deposits (exploration), but every GB_EVERY-th iteration the best-so-far
    // deposits instead, steering the colony back toward the incumbent so each
    // restart epoch refines it rather than resampling from scratch. The first
    // GB_WARMUP iterations after a restart stay iteration-best only, so the
    // epoch can explore before being pulled in.
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

        // Global evaporation — applied once per iteration after all ants,
        // with both MMAS bounds enforced: the floor keeps every edge
        // selectable, the ceiling stops any edge running away.
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
                // Better incumbent → tighter bounds around the new cost scale
                (tau_min, tau_max) = mmas_bounds(best_fit, n_free, cfg.rho);
            } else {
                stagnant += 1;
            }

            // Choose the depositing solution: iteration-best by default,
            // best-so-far on the GB schedule (capped at τ_max either way)
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

        // Stagnation restart: reset the trail to τ_max (NOT τ_min) so the
        // colony re-explores from a fully uniform state, while the incumbent
        // best solution is kept.
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