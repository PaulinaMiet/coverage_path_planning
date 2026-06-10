mod ils;
mod shared;

use ils::IlsConfig;
use shared::*;

fn main() {
    let mut rng = rand::thread_rng();
    let grid = parse_grid(INSTANCE);
    let (rw, uw) = penalty_weights(&grid);
    let alg = "ils";

    println!("\n=== ILS RUN: {} ===", INSTANCE);
    println!(
        "free cells: {}  |  weights — revisit: {:.0}  unvisited: {:.0}\n",
        free_cells(&grid),
        rw,
        uw
    );

    // 1. Run Algorithm
    let mut cfg = IlsConfig::default_for();
    cfg.strategy = ils::StartingStrategy::Random;
    let result = ils::ils_run(&grid, &cfg, &mut rng);

    // 2. Print Convergence (Progress)
    println!("--- Convergence ---");
    let mut last = f64::MAX;
    for log in &result.history {
        if log.fitness < last {
            println!("  iter {:5}  fitness: {:.2}", log.iteration, log.fitness);
            last = log.fitness;
        }
    }

    // 3. Print Final Summary
    let f = &result.best_fitness;
    println!("\n--- Results ---");
    println!("best fitness : {:.2}", f.total);
    println!("  distance   : {:.2}", f.distance);
    println!("  revisits   : {}", f.revisits);
    println!("  unvisited  : {}", f.unvisited);
    println!("  solution   : [{}]", fmt_moves(&result.best_moves));
    println!("\n--- Visualisation ---");
    display_grid(&grid, &decode(&result.best_moves, &grid));

    // 4. Save Outputs
    let run_tag = generate_run_tag(INSTANCE);
    save_csv(&result, &run_tag, alg);
    save_json(&result, &grid, &run_tag, alg, &cfg.to_json_map());
}
