mod aco;
mod shared;

use aco::AcoConfig;
use shared::*;

fn main() {
    let alg = "aco";
    let grid = parse_grid(INSTANCE);
    let (rw, uw) = penalty_weights(&grid);

    println!("\n=== {} ===", INSTANCE);
    println!(
        "free cells: {}  |  weights — revisit: {:.0}  unvisited: {:.0}\n",
        free_cells(&grid),
        rw,
        uw
    );

    let cfg = AcoConfig::default_for(&grid);
    let result = aco::aco_run(&grid, &cfg);

    // Print convergence (only lines where fitness improved)
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

    save_csv(&result, &run_tag, alg);
    save_json(&result, &grid, &run_tag, alg, &cfg.to_json_map());
}
