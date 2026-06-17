// shared.rs — types, grid, decoder, and fitness. can be used by all algorithm files.

use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

pub const START: Position = (0, 0);
pub const INSTANCE: &str = "instances/cpp_20x20_sparse.txt";

pub type Grid = Vec<Vec<u8>>; // 0 = free, 1 = obstacle
pub type Position = (usize, usize); // (row, col)

/// The move alphabet. Copy so we can pass by value freely.
/// All 8 moves are single-step: 4 cardinal + 4 diagonal.
/// If the destination cell is blocked or out of bounds the robot stays put.
#[derive(Clone, Copy, PartialEq)]
pub enum Move {
    Up,
    Down,
    Left,
    Right,
    UpLeft,
    UpRight,
    DownLeft,
    DownRight,
}

/// ALL_MOVES index contract (shared by pheromone arrays and NeighbourMap):
///   0=Up  1=Down  2=Left  3=Right  4=UpLeft  5=UpRight  6=DownLeft  7=DownRight
pub const ALL_MOVES: [Move; 8] = [
    Move::Up,
    Move::Down,
    Move::Left,
    Move::Right,
    Move::UpLeft,
    Move::UpRight,
    Move::DownLeft,
    Move::DownRight,
];

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Move::Up => "U",
                Move::Down => "D",
                Move::Left => "L",
                Move::Right => "R",
                Move::UpLeft => "UL",
                Move::UpRight => "UR",
                Move::DownLeft => "DL",
                Move::DownRight => "DR",
            }
        )
    }
}

/// Best solution of each iteration for CSV export
pub struct IterationLog {
    pub iteration: usize,
    pub fitness: f64,
    pub distance: f64,
    pub revisits: usize,
    pub unvisited: usize,
    pub moves: Vec<Move>,
}

pub struct Result {
    pub best_moves: Vec<Move>,
    pub best_fitness: Fitness,
    pub history: Vec<IterationLog>, // one entry per iteration
}

/// Turns an array of moves into a single string.
pub fn fmt_moves(moves: &[Move]) -> String {
    moves
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Reads grid from file to Grid (Vec<Vec<u8>>)
pub fn parse_grid(path: &str) -> Grid {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect())
        .collect()
}

/// Counts how many cells are not obstacles (the target to cover)
pub fn free_cells(grid: &Grid) -> usize {
    grid.iter()
        .flat_map(|r| r.iter())
        .filter(|&&c| c == 0)
        .count()
}

/// Returns true if a cell is within bounds and not an obstacle
pub fn is_free(r: isize, c: isize, grid: &Grid) -> bool {
    r >= 0 && r < grid.len() as isize //check if row is correct
        && c >= 0 && c < grid[0].len() as isize // check if column is correct
        && grid[r as usize][c as usize] == 0 // check if cell is without obstacle
}

/// Prints the grid to console. \
/// S=start  E=end  *=visited  .=missed  #=obstacle
pub fn display_grid(grid: &Grid, path: &[Position]) {
    let visited: HashSet<Position> = path.iter().cloned().collect();
    for (r, row) in grid.iter().enumerate() {
        let s: String = (0..row.len())
            .map(|c| match () {
                _ if grid[r][c] == 1 => '#',
                _ if path.first() == Some(&(r, c)) => 'S',
                _ if path.last() == Some(&(r, c)) => 'E',
                _ if visited.contains(&(r, c)) => '*',
                _ => '.',
            })
            .collect();
        println!("  {}", s);
    }
}

/// Translates a Move into a (row_delta, col_delta) step.
/// Diagonal moves step ±1 on both axes.
pub fn dir_delta(mv: Move) -> (isize, isize) {
    match mv {
        Move::Up => (-1, 0),
        Move::Down => (1, 0),
        Move::Left => (0, -1),
        Move::Right => (0, 1),
        Move::UpLeft => (-1, -1),
        Move::UpRight => (-1, 1),
        Move::DownLeft => (1, -1),
        Move::DownRight => (1, 1),
    }
}

/// Takes current position, attempts to execute a Move, and returns an array of covered cells.
/// Returns a one-element Vec if the destination is free,
/// or an empty Vec if it is blocked or out of bounds.
pub fn apply_move(pos: Position, mv: Move, grid: &Grid) -> Vec<Position> {
    let (r, c) = (pos.0 as isize, pos.1 as isize);
    let (dr, dc) = dir_delta(mv);
    if is_free(r + dr, c + dc, grid) {
        vec![((r + dr) as usize, (c + dc) as usize)]
    } else {
        vec![]
    }
}

/// Generates path from a list of moves, starting from START (0, 0).
pub fn decode(moves: &[Move], grid: &Grid) -> Vec<Position> {
    let mut path = vec![START];
    let mut pos = START;
    for &mv in moves {
        let cells = apply_move(pos, mv, grid);
        if let Some(&last) = cells.last() {
            pos = last;
        }
        path.extend(cells);
    }
    path
}

pub struct Fitness {
    pub total: f64,
    pub distance: f64,
    pub revisits: usize,
    pub unvisited: usize,
}

///Returns revisit and unvisited penalty value. \
/// Penalties scale with grid area so the algorithm always prefers full coverage
/// area=25  : revisit=50,   unvisited=125
/// area=100 : revisit=200,  unvisited=500
/// area=400 : revisit=800,  unvisited=2000
pub fn penalty_weights(grid: &Grid) -> (f64, f64) {
    let area = (grid.len() * grid[0].len()) as f64;
    (area * 2.0, area * 5.0)
}

/// Calculates the euclidean distance.
fn euclidean(a: Position, b: Position) -> f64 {
    let dr = a.0 as f64 - b.0 as f64;
    let dc = a.1 as f64 - b.1 as f64;
    (dr * dr + dc * dc).sqrt()
}

/// Evaluates path fitness.
pub fn evaluate(path: &[Position], grid: &Grid) -> Fitness {
    let distance: f64 = path.windows(2).map(|w| euclidean(w[0], w[1])).sum();
    let mut seen = HashSet::new();
    let mut revisits = 0;
    for &p in path {
        if !seen.insert(p) {
            revisits += 1;
        }
    }
    let unvisited = free_cells(grid).saturating_sub(seen.len());
    let (rw, uw) = penalty_weights(grid);
    let total = distance + revisits as f64 * rw + unvisited as f64 * uw;
    Fitness {
        total,
        distance,
        revisits,
        unvisited,
    }
}

/// Extracts the filename without extension from a path
/// "instances/cpp_10x10_line.txt" → "cpp_10x10_line"
pub fn instance_stem(path: &str) -> String {
    path.split('/')
        .last()
        .unwrap_or("run")
        .trim_end_matches(".txt")
        .to_string()
}

/// Generates a unique tag based on the instance name and current time
pub fn generate_run_tag(instance_path: &str) -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{}_{}", instance_stem(instance_path), ts)
}

/// Ensures the results directory exists
pub fn ensure_results_dir() {
    fs::create_dir_all("results").expect("could not create results/");
}

/// Saves csv with iteration, fitness, distance, revisits, unvisited, solution
/// to results dir
pub fn save_csv(result: &Result, tag: &str, alg: &str) {
    ensure_results_dir();
    let path = format!("results/{}_{}.csv", tag, alg);
    let mut f = fs::File::create(&path).expect("could not create CSV file");

    writeln!(f, "iteration,fitness,distance,revisits,unvisited,solution").unwrap();
    for log in &result.history {
        writeln!(
            f,
            "{},{:.2},{:.2},{},{},\"{}\"",
            log.iteration,
            log.fitness,
            log.distance,
            log.revisits,
            log.unvisited,
            fmt_moves(&log.moves),
        )
        .unwrap();
    }
    println!("\nsaved csv   → {}", path);
}

// Best solution path as (row,col) pairs in json for grid visualisation
pub fn save_json(result: &Result, grid: &Grid, tag: &str, alg: &str, config_json: &str) {
    ensure_results_dir();
    let path = format!("results/{}_{}.json", tag, alg);

    let f = &result.best_fitness;
    let best_path = decode(&result.best_moves, grid);

    let path_json: String = best_path
        .iter()
        .map(|(r, c)| format!("[{},{}]", r, c))
        .collect::<Vec<_>>()
        .join(",");

    let json = format!(
        "{{\n\
        \t\"algorithm\": \"{alg}\",\n\
        \t\"instance\": \"{tag}\",\n\
        \t\"config\": {{ {} }},\n\
        \t\"best_fitness\": {:.2},\n\
        \t\"distance\": {:.2},\n\
        \t\"revisits\": {},\n\
        \t\"unvisited\": {},\n\
        \t\"best_moves\": \"{}\",\n\
        \t\"best_path\": [{}]\n\
        }}",
        config_json,
        f.total,
        f.distance,
        f.revisits,
        f.unvisited,
        fmt_moves(&result.best_moves),
        path_json,
    );

    fs::write(&path, json).expect("could not write JSON file");
    println!("saved → {}", path);
}
