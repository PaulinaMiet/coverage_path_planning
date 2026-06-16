# Coverage Path Planning (CPP) Optimization

This project implements and compares three metaheuristic algorithms for the Coverage Path Planning problem: Ant Colony Optimization (ACO), a hybrid ACO with local pheromone update and MMAS-style bounds (ACO+), and Iterated Local Search (ILS) with Minimum Spanning Tree initialization. The goal is to find a path through a grid that visits all free cells while minimizing total distance and revisits.

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (latest stable version)

### Installation

```bash
git clone <repository-url>
cd coverage_path_planning
cargo build --release
```

### Running the Algorithms

```bash
cargo run --bin aco
cargo run --bin aco_plus
cargo run --bin ils
```

## Project Structure

- `src/shared.rs`: Core logic shared by all algorithms — grid parsing, fitness evaluation, move decoding, and data export.
- `src/aco.rs`: Standard ACO implementation.
- `src/aco_plus.rs`: ACO+ implementation with local pheromone update and MMAS pheromone bounds.
- `src/ils.rs`: ILS implementation including MST initialization logic.
- `src/main_aco.rs` / `src/main_aco_plus.rs` / `src/main_ils.rs`: Entry points for each algorithm.
- `instances/`: Grid files representing different environments (chunk, line, sparse) across three sizes (5×5, 10×10, 20×20).
- `results/`: Auto-generated directory where CSV logs and JSON path data are saved.

## Algorithms

### Ant Colony Optimization (ACO)

Ants build paths by balancing pheromone intensity (learned behaviour) and a coverage-gain heuristic (greedy exploration). Global pheromone evaporation applies after each iteration and reinforcement is applied across all ants.

### ACO+ (Hybrid ACO)

Extends ACO with two additional mechanisms. A local pheromone update weakens trail intensity on an edge immediately after an ant crosses it, steering following ants in the same iteration toward alternative paths. MMAS-style bounds restrict global reinforcement to the best ant only and keep all pheromone values within a fixed min-max range, preventing stagnation. A reset triggers after 250 rounds without improvement.

### Iterated Local Search (ILS)

1. **MST Initialization**: Builds an initial path using a Minimum Spanning Tree to guarantee full coverage from the start.
2. **Local Search**: Stochastic hill-climbing to reduce distance and revisits.
3. **Perturbation**: Randomly alters path segments to escape local optima.

## Results and Visualization

Every run writes two files to the `results/` directory:

- `.csv`: Per-iteration log of fitness metrics (distance, revisits, unvisited cells).
- `.json`: Best solution metadata including a full list of `[row, col]` coordinates for visualization.

## Configuration

Grid instances are defined in `.txt` files under `instances/`. To change the active instance, modify the `INSTANCE` constant in `src/shared.rs`.
