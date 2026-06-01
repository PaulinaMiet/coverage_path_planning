// shared.rs — types, grid, decoder, and fitness. can be Used by all algorithm files.

use rand::Rng;
use std::collections::HashSet;
use std::fmt;
use std::fs;

// ── Core types ────────────────────────────────────────────────

pub const START: Position = (0, 0);

pub type Grid = Vec<Vec<u8>>; // 0 = free, 1 = obstacle
pub type Position = (usize, usize); // (row, col)

/// The move alphabet. Copy so we can pass by value freely.
/// Plain (Up/Down/Left/Right): move 1 cell, stay put if blocked.
/// Star (U*/D*/L*/R*): slide until the next cell is a wall or obstacle, then stop
#[derive(Clone, Copy, PartialEq)]
pub enum Move {
    Up,
    Down,
    Left,
    Right,
    UpS,
    DownS,
    LeftS,
    RightS,
}

pub const ALL_MOVES: [Move; 8] = [
    Move::Up,
    Move::Down,
    Move::Left,
    Move::Right,
    Move::UpS,
    Move::DownS,
    Move::LeftS,
    Move::RightS,
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
                Move::UpS => "U*",
                Move::DownS => "D*",
                Move::LeftS => "L*",
                Move::RightS => "R*",
            }
        )
    }
}

/// Turns an array of moves into a single string.
pub fn fmt_moves(moves: &[Move]) -> String {
    moves
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Grid ──────────────────────────────────────────────────────

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

// ── Decoder ───────────────────────────────────────────────────

/// Translates a Move into coordinate changes for the row and column.
pub fn dir_delta(mv: Move) -> (isize, isize, bool) {
    match mv {
        Move::Up => (-1, 0, false),
        Move::Down => (1, 0, false),
        Move::Left => (0, -1, false),
        Move::Right => (0, 1, false),
        Move::UpS => (-1, 0, true),
        Move::DownS => (1, 0, true),
        Move::LeftS => (0, -1, true),
        Move::RightS => (0, 1, true),
    }
}

/// Takes current position, attempts to execute a Move, and returns an array of covered cells.
/// Uses [`is_free`] to prevent crashing.
pub fn apply_move(pos: Position, mv: Move, grid: &Grid) -> Vec<Position> {
    let (r, c) = (pos.0 as isize, pos.1 as isize);
    let (dr, dc, star) = dir_delta(mv);

    let mut cells = vec![];
    let (mut cr, mut cc) = (r, c);

    if star {
        while is_free(cr + dr, cc + dc, grid) {
            cr += dr;
            cc += dc;
            cells.push((cr as usize, cc as usize));
        }
    } else {
        if is_free(r + dr, c + dc, grid) {
            cells.push(((r + dr) as usize, (c + dc) as usize));
        }
    }
    cells
}

/// Generates path form a list of moves.
pub fn decode(moves: &[Move], grid: &Grid, start: Position) -> Vec<Position> {
    let mut path = vec![start];
    let mut pos = start;
    for &mv in moves {
        let cells = apply_move(pos, mv, grid);
        if let Some(&last) = cells.last() {
            pos = last;
        }
        path.extend(cells);
    }
    path
}

// ── Fitness ───────────────────────────────────────────────────

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

// // ── Utility (this is for ILS algorithm initialization that i tested with random, we proposed spanning tree..) ───────────────────────────────────────────────────

// /// Random move sequence. Length scales with grid size.
// pub fn random_solution(grid: &Grid, rng: &mut impl Rng) -> Vec<Move> {
//     let len = (grid.len() + grid[0].len()) * 2;
//     (0..len).map(|_| ALL_MOVES[rng.gen_range(0..8)]).collect()
// }
