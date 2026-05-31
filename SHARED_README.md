# shared.rs

shared.rs is the shared foundation for all algorithm implementations.  
Every algorithm file should import from here so we all score routes the same way.

---

## Core Types

```rust
pub type Grid = Vec<Vec<u8>>;   // the map — 0 = free cell, 1 = obstacle
pub type Position  = (usize, usize); // a position on the map as (row, col)
```

The grid is just a 2D array of 0s and 1s. A `Position` is a coordinate like `(2, 3)` meaning row 2, column 3.

---

## Moves — the "alphabet" of solutions

```rust
pub enum Move { Up, Down, Left, Right, UpS, DownS, LeftS, RightS }
```

There are **8 possible moves**:

- **Plain moves** (`Up`, `Down`, `Left`, `Right`) — move exactly **1 cell** in that direction. If the next cell is a wall, the robot stays put.
- **Slide moves** (`UpS`, `DownS`, `LeftS`, `RightS`) — the robot **slides** in that direction until it hits a wall or obstacle, then stops. Like a hockey puck.

A **solution** is a list (sequence) of these moves, e.g. `[Right, DownS, Left, Up, ...]`.

```rust
pub const ALL_MOVES: [Move; 8] = [...]; // array of all 8 moves — useful for random sampling
```

---

## Grid Utilities

| Function | What it does |
|---|---|
| `parse_grid(path)` | Loads a grid from a `.txt` file |
| `free_cells(grid)` | Counts how many cells are not obstacles (the target to cover) |
| `is_free(r, c, grid)` | Returns true if a cell is within bounds and not an obstacle |
| `display_grid(grid, path)` | Prints the grid to console — `S`=start, `E`=end, `*`=visited, `.`=missed, `#`=wall |

---

## Decoder — turning a move list into a path

```rust
pub fn decode(moves: &[Move], grid: &Grid, start: Position) -> Vec<Position>
```

This takes your solution (a list of moves) and simulates the robot walking through the grid.  
It returns the **list of positions visited**, in order.

**You will always call this before evaluating.** It's the bridge between your solution representation and the actual route.

`apply(pos, mv, grid)` is the single-step version — given a position and one move, returns the cells the robot lands on.

---

## Fitness — how we score a solution

```rust
pub fn evaluate(path: &[Position], grid: &Grid) -> Fitness
```

This is the **one function everyone must use** to score their solutions. It returns a `Fitness` struct:

```rust
pub struct Fitness {
    pub total:     f64,   // ← the final score (LOWER = BETTER)
    pub distance:  f64,   // total travel distance (Euclidean)
    pub revisits:  usize, // how many times the robot stepped on an already-visited cell
    pub unvisited: usize, // how many free cells were never visited
}
```

### How the score is calculated

```
total = distance + (revisits × revisit_penalty) + (unvisited × unvisited_penalty)
```

The penalties scale with the **grid area** (rows × cols):

```
revisit_penalty  = area × 2.0
unvisited_penalty = area × 5.0
```

**Why scaling?** A missed cell is much worse than a revisit, and both should matter more on a bigger map.

**Example** on a 10×10 grid (area = 100):
- Each revisit costs **200 points**
- Each unvisited cell costs **500 points**

So the algorithm is pushed hard to cover everything, and secondarily to avoid backtracking.

`penalty_weights(grid)` returns `(revisit_penalty, unvisited_penalty)` if you need them directly.

---

## How to write your own algorithm file

1. **Import shared** at the top of your file:
   ```rust
   mod shared;
   use shared::*;
   ```

2. **Load the grid** (or accept it as a parameter):
   ```rust
   let grid = parse_grid("path/to/map.txt");
   let start: Position = (0, 0); // or wherever your robot starts
   ```

3. **Your algorithm produces a `Vec<Move>`** — a sequence of moves. How you find that sequence is up to you (ACO, ILS, GA, greedy, etc.).

4. **Decode and evaluate** using shared functions:
   ```rust
   let path    = decode(&my_moves, &grid, start);
   let fitness = evaluate(&path, &grid);
   println!("Score: {:.2}", fitness.total); // lower is better
   ```

5. **Create your own `main_yourname.rs`** as the entry point for your algorithm.

That's it — as long as you use `evaluate()` from shared, your scores are directly comparable to everyone else's.
