// main_aco_plus.rs — Entry point for ACO+

mod shared;
mod aco_plus;

use shared::*;
use aco_plus::AcoConfig;

fn main() {
    let grid = parse_grid(INSTANCE);
    let (rw, uw) = penalty_weights(&grid);

    println!("\n=== {} ===", INSTANCE);
    println!("free cells: {}  |  weights — revisit: {:.0}  unvisited: {:.0}\n",
        free_cells(&grid), rw, uw);

    print_neighbour_summary(&grid);

    let cfg = AcoConfig::default_for(&grid);
    println!("Algorithm      : ACO+ (step-only, local pheromone decay)");
    println!("q0             : {}  |  local_rho: {}", cfg.q0, cfg.local_rho);
    println!("solution_length: {}\n", cfg.solution_length);

    let result = aco_plus::run(&grid, &cfg);

    // Print convergence — only lines where fitness improved
    let mut last = f64::MAX;
    for log in &result.history {
        if log.fitness < last {
            println!("  iter {:5}  fitness: {:.2}", log.iteration, log.fitness);
            last = log.fitness;
        }
    }

    let f = &result.best_fitness;
    println!("\nbest fitness : {:.2}", f.total);
    println!("  distance   : {:.2}", f.distance);
    println!("  revisits   : {}", f.revisits);
    println!("  unvisited  : {}", f.unvisited);
    println!("  solution   : [{}]", fmt_moves(&result.best_moves));
    println!();
    display_grid(&grid, &decode(&result.best_moves, &grid));

    let run_tag = generate_run_tag(INSTANCE);
    save_csv(&result, &run_tag, "aco_plus");
    save_json(&result, &grid, &run_tag, "aco_plus", &cfg.to_json_map());
}

fn print_neighbour_summary(grid: &Grid) {
    let map = aco_plus::build_neighbour_map(grid);
    println!("Neighbour map — legal step count per cell (# = obstacle):");
    for r in 0..grid.len() {
        print!("  ");
        for c in 0..grid[0].len() {
            if grid[r][c] == 1 { print!("#  "); }
            else               { print!("{}  ", map[r][c].len()); }
        }
        println!();
    }
    println!();
}