# Coverage Path Planning (CPP) Optimization

This project implements and compares two metaheuristic algorithms for the **Coverage Path Planning** problem: Ant Colony Optimization (ACO) and Iterated Local Search (ILS) with Minimum Spanning Tree (MST) initialization.

The goal is to find a path through a grid that visits all free cells while minimizing total distance and revisits.

##  Getting Started

### Prerequisites
*   [Rust](https://www.rust-lang.org/tools/install) (latest stable version)

### Installation
```bash
git clone <repository-url>
cd coverage_path_planning
cargo build --release
```

### Running the Algorithms
You can run either the ACO or ILS implementation:

**Run ACO:**
```bash
cargo run --bin aco
```

**Run ILS:**
```bash
cargo run --bin ils
```

##  Project Structure

*   `src/shared.rs`: Core logic shared by all algorithms (Grid parsing, fitness evaluation, move decoding, and data export).
*   `src/aco.rs`: Ant Colony Optimization implementation.
*   `src/ils.rs`: Iterated Local Search implementation (includes MST logic).
*   `src/main_aco.rs` / `src/main_ils.rs`: Main entry points for the respective algorithms.
*   `instances/`: Text-based grid files representing different environments (sparse, line, chunk).
*   `results/`: Automatically generated directory where CSV logs and JSON path data are saved.

##  Algorithms

### Ant Colony Optimization (ACO)
Uses a pheromone-based approach where "ants" build paths by balancing pheromone intensity (learned behavior) and a coverage-gain heuristic (greedy exploration).

### Iterated Local Search (ILS)
Focuses on refining a high-quality initial solution:
1.  **MST Initialization**: Generates a skeleton path using a Minimum Spanning Tree to guarantee 100% coverage from the start.
2.  **Local Search**: Stochastic hill-climbing to reduce distance and revisits.
3.  **Perturbation**: Randomly alters path segments to escape local optima.

## Results & Visualization

Every run generates two types of files in the `results/` directory:
*   `.csv`: A per-iteration log of fitness metrics (distance, revisits, unvisited cells).
*   `.json`: The metadata for the best solution, including a complete list of `[row, col]` coordinates for visualization.

##  Configuration

Grids are defined in the `.txt` files in `instances/`. You can change the active instance by modifying the `INSTANCE` constant in `src/main_aco.rs` or `src/main_ils.rs`.
