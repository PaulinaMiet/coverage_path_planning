# shared.rs

This is the shared foundation for all algorithm implementations. Every algorithm imports from here so all solutions are scored the same way.

---

## Core Types

```rust
pub type Grid = Vec<Vec<u8>>;      // the map — 0 = free cell, 1 = obstacle
pub type Position = (usize, usize); // a coordinate as (row, col)
```

The grid is a 2D array of 0s and 1s. A `Position` like `(2, 3)` means row 2, column 3.

---

## Moves

```rust
pub enum Move { Up, Down, Left, Right, UpS, DownS, LeftS, RightS }
```

Eight moves are available. Plain moves (`Up`, `Down`, `Left`, `Right`) move the robot exactly one cell. If the next cell is a wall, the robot stays put. Slide moves (`UpS`, `DownS`, `LeftS`, `RightS`) keep moving the robot in that direction until it hits a wall or obstacle, like a hockey puck.

A solution is a sequence of these moves, for example `[Right, DownS, Left, Up, ...]`.

```rust
pub const ALL_MOVES: [Move; 8] = [...]; // all 8 moves — useful for random sampling
```

---

## Grid Utilities

| Function | What it does |
|---|---|
| `parse_grid(path)` | Loads a grid from a `.txt` file |
| `free_cells(grid)` | Counts cells that are not obstacles |
| `is_free(r, c, grid)` | Returns true if a cell is in bounds and not an obstacle |
| `display_grid(grid, path)` | Prints the grid — `S`=start, `E`=end, `*`=visited, `.`=missed, `#`=wall |

---

## Decoder

```rust
pub fn decode(moves: &[Move], grid: &Grid, start: Position) -> Vec<Position>
```

Takes a move sequence and simulates the robot walking through the grid. Returns the list of positions visited in order. Call this before evaluating — it converts your solution into an actual route.

`apply(pos, mv, grid)` is the single-step version. Given a position and one move, it returns the cells the robot lands on.

---

## Fitness

```rust
pub fn evaluate(path: &[Position], grid: &Grid) -> Fitness
```

The one function every algorithm must use to score a solution. Returns a `Fitness` struct:

```rust
pub struct Fitness {
    pub total:     f64,   // final score — lower is better
    pub distance:  f64,   // total travel distance (Euclidean)
    pub revisits:  usize, // cells stepped on more than once
    pub unvisited: usize, // free cells never reached
}
```

Score is calculated as: total = distance + (revisits × revisit_penalty) + (unvisited × unvisited_penalty)

Both penalties scale with grid area (rows × cols):
revisit_penalty  = area × 2.0
unvisited_penalty = area × 5.0

On a 10×10 grid each revisit costs 200 points and each missed cell costs 500. Missing a cell is penalised harder than backtracking, and both penalties grow with map size.

`penalty_weights(grid)` returns `(revisit_penalty, unvisited_penalty)` if you need them directly.

---

## Writing Your Own Algorithm

1. Import shared at the top of your file:
```rust
   mod shared;
   use shared::*;
```

2. Load the grid and penalties:
```rust
   let grid = parse_grid(INSTANCE);
   let (revis_pen, unvis_pen) = penalty_weights(&grid);
```

3. Your algorithm produces a `Vec<Move>`. The starting position is `(0, 0)`.

4. Decode and evaluate using the shared functions:
```rust
   let path    = decode(&my_moves, &grid, start);
   let fitness = evaluate(&path, &grid);
   println!("Score: {:.2}", fitness.total); // lower is better
```

5. Create `main_yourname.rs` as the entry point for your algorithm.

Scores are directly comparable to everyone else's.
