// aco_plus.rs — ACO+ (Improved Ant Colony Optimisation)
//
// Movement: 8-direction single-step (4 cardinal + 4 diagonal)
// Anti-oscillation: short-term tabu window (default size 6) + safety valve
// Pheromone: τ[row][col][move_idx], 8 slots per cell (one per direction)
// Key differentiator vs basic ACO: LOCAL pheromone decay applied after each
// ant step, within the same iteration. This weakens the trail an ant just
// walked, steering subsequent ants in the same round toward unexplored cells.
//   τ[r][c][mv] ← τ[r][c][mv] × (1 − local_rho)

use crate::shared::*;
use rand::Rng;
use std::collections::{HashSet, VecDeque};

// ── ACO+ private types and helpers ───────────────────────────
// These are only needed by aco_plus.rs and live here rather than
// in shared.rs to avoid dead-code warnings for aco and ils.

type Pos = Position;
type NeighbourMap = Vec<Vec<Vec<(usize, Position)>>>;

// Single-step move — identical to apply_move, named `apply` for readability
fn apply(pos: Position, mv: Move, grid: &Grid) -> Vec<Position> {
    apply_move(pos, mv, grid)
}

// BFS connectivity check from `start`.
// Returns Ok(n)  if all n free cells are reachable.
// Returns Err(k) if k free cells are unreachable.
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
                if visited.insert(npos) {
                    queue.push_back(npos);
                }
            }
        }
    }

    let reachable = visited.len();
    if reachable == total { Ok(reachable) } else { Err(total - reachable) }
}

// Pre-computes all legal single-step neighbours for every free cell.
// map[r][c] = Vec of (move_index, destination).
// move_index matches ALL_MOVES and the pheromone [f64; 8] slots.
pub fn build_neighbour_map(grid: &Grid) -> NeighbourMap {
    let rows = grid.len();
    let cols = grid[0].len();
    let mut map: NeighbourMap = vec![vec![Vec::new(); cols]; rows];

    // Deltas in ALL_MOVES index order:
    // 0=Up  1=Down  2=Left  3=Right  4=UpLeft  5=UpRight  6=DownLeft  7=DownRight
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

// ── Config ────────────────────────────────────────────────────

pub struct AcoConfig {
    pub n_ants:          usize,
    pub n_iterations:    usize,
    pub solution_length: usize,

    pub alpha:     f64,   // pheromone influence
    pub beta:      f64,   // heuristic influence
    pub rho:       f64,   // global evaporation rate (0–1)
    pub local_rho: f64,   // local pheromone decay per step — ACO+ only
    pub tau_init:  f64,   // starting pheromone on every edge
    pub q0:        f64,   // exploitation probability (ACS-style)
    pub tabu_size: usize, // short-term memory — blocks last N visited cells
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
            local_rho:       0.01,  // gentle within-iteration diversity
            tau_init:        1.0,
            q0:              0.7,   // lower than basic ACO — less greedy, helps wall escape
            tabu_size:       6,
        }
    }

    pub fn to_json_map(&self) -> String {
        format!(
            "\"n_ants\": {}, \"n_iterations\": {}, \"alpha\": {}, \"beta\": {}, \
             \"rho\": {}, \"local_rho\": {}, \"q0\": {}",
            self.n_ants, self.n_iterations, self.alpha, self.beta,
            self.rho, self.local_rho, self.q0
        )
    }
}

// ── Pheromone ─────────────────────────────────────────────────

// τ[row][col][move_idx] — 8 directions, index matches ALL_MOVES:
// 0=Up  1=Down  2=Left  3=Right  4=UpLeft  5=UpRight  6=DownLeft  7=DownRight
type Pheromone = Vec<Vec<[f64; 8]>>;

fn init_pheromone(grid: &Grid, tau: f64) -> Pheromone {
    vec![vec![[tau; 8]; grid[0].len()]; grid.len()]
}

// ── Heuristic ─────────────────────────────────────────────────

// Returns attractiveness of a move based on coverage gain.
// Base: +1.0 if the destination is unvisited, +0.0 if already seen.
// Diagonal bonus: for diagonal moves, also counts unvisited orthogonal
// neighbours of the destination (×0.5 each) — rewards the positional
// advantage that diagonals give over cardinal steps.
// +1.0 floor ensures weight is always > 0 so blocked moves stay selectable.
fn coverage_gain(pos: Pos, mv: Move, grid: &Grid, visited: &HashSet<Pos>) -> f64 {
    let dest_cells = apply(pos, mv, grid);
    if dest_cells.is_empty() { return 1.0; }

    let dest = dest_cells[0];

    let is_diagonal = matches!(mv,
        Move::UpLeft | Move::UpRight | Move::DownLeft | Move::DownRight
    );

    let base = if !visited.contains(&dest) { 1.0 } else { 0.0 };

    let bonus = if is_diagonal {
        let (dr, dc) = (dest.0 as isize, dest.1 as isize);
        let ortho = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        ortho.iter()
            .filter(|(r, c)| is_free(dr + r, dc + c, grid))
            .filter(|(r, c)| !visited.contains(&((dr + r) as usize, (dc + c) as usize)))
            .count() as f64 * 0.5
    } else {
        0.0
    };

    base + bonus + 1.0
}

// ── Ant construction ──────────────────────────────────────────

fn build_solution(
    grid:          &Grid,
    pheromone:     &mut Pheromone,  // mut: local decay is applied in-place
    neighbour_map: &NeighbourMap,
    cfg:           &AcoConfig,
    rng:           &mut impl Rng,
) -> Vec<Move> {
    let mut moves   = Vec::with_capacity(cfg.solution_length);
    let mut pos     = START;
    let mut visited: HashSet<Pos> = HashSet::new();
    visited.insert(pos);

    // Short-term tabu window — blocks the last tabu_size destinations
    let mut tabu: VecDeque<Pos> = VecDeque::new();

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

        // ── LOCAL pheromone decay (ACO+ distinguishing line) ──────────
        // Weaken the trail just used so subsequent ants in this iteration
        // are nudged away from the same path, increasing intra-round diversity.
        pheromone[r][c][mv_idx] *= 1.0 - cfg.local_rho;

        // Advance one cell and record coverage
        let new_cells = apply(pos, mv, grid);
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

// ── Main ACO+ loop ────────────────────────────────────────────

pub fn run(grid: &Grid, cfg: &AcoConfig) -> Result {
    // Connectivity check — warns if any free cell is unreachable from (0,0)
    match flood_fill_check(grid, START) {
        Ok(n)  => println!("[ACO+] flood-fill OK — {} free cells all connected", n),
        Err(n) => println!("[ACO+] WARNING: {} free cells unreachable from start", n),
    }

    let neighbour_map = build_neighbour_map(grid);

    let mut rng       = rand::thread_rng();
    let mut pheromone = init_pheromone(grid, cfg.tau_init);
    let mut best_moves: Vec<Move> = Vec::new();
    let mut best_fit  = f64::MAX;
    let mut history   = Vec::new();

    for iter in 0..cfg.n_iterations {
        let mut iter_best: Option<(Vec<Move>, Fitness)> = None;

        for _ in 0..cfg.n_ants {
            // build_solution takes &mut pheromone for local decay
            let moves   = build_solution(grid, &mut pheromone, &neighbour_map, cfg, &mut rng);
            let fitness = evaluate(&decode(&moves, grid), grid);
            if iter_best.as_ref().map_or(true, |(_, b)| fitness.total < b.total) {
                iter_best = Some((moves, fitness));
            }
        }

        // Global evaporation — applied once per iteration after all ants
        for row in &mut pheromone {
            for cell in row {
                for tau in cell.iter_mut() { *tau *= 1.0 - cfg.rho; }
            }
        }

        if let Some((mvs, fit)) = iter_best {
            // Deposit on the best ant's trail for this iteration
            let deposit = 1.0 / fit.total;
            let mut pos = START;
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

    let best_fitness = evaluate(&decode(&best_moves, grid), grid);
    Result { best_moves, best_fitness, history }
}