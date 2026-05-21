
mod shared;
mod aco;

use shared::*;
use aco::{AcoConfig, AcoResult};
use std::fs;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

const INSTANCE: &str = "instances/cpp_10x10_line.txt";

fn main() {
    let grid = parse_grid(INSTANCE);
    let (rw, uw) = penalty_weights(&grid);

    println!("\n=== {} ===", INSTANCE);
    println!("free cells: {}  |  weights — revisit: {:.0}  unvisited: {:.0}\n",
        free_cells(&grid), rw, uw);

    let cfg    = AcoConfig::default_for(&grid);
    let result = aco::run(&grid, &cfg);

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
    display_grid(&grid, &decode(&result.best_moves, &grid, (0, 0)));

    // Unique timestamp so each run gets its own file
    let ts      = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let stem    = instance_stem(INSTANCE);
    let run_tag = format!("{}_{}", stem, ts);

    save_csv(&result, &run_tag);
    save_json(&result, &grid, &run_tag);
}

// ── Output helpers (for loggin) ────────────────────────────────────────────

// "instances/cpp_10x10_line.txt" → "cpp_10x10_line"
fn instance_stem(path: &str) -> String {
    path.split('/').last().unwrap_or("run")
        .trim_end_matches(".txt")
        .to_string()
}

fn ensure_results_dir() {
    fs::create_dir_all("results").expect("could not create results/");
}

// Every iteration logged — one row per iteration
fn save_csv(result: &AcoResult, tag: &str) {
    ensure_results_dir();
    let path = format!("results/{}_aco.csv", tag);
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
        ).unwrap();
    }

    println!("\nsaved → {}", path);
}

// Best solution path as (row,col) pairs , in json,  for future grid visualisation
fn save_json(result: &AcoResult, grid: &Grid, tag: &str) {
    ensure_results_dir();
    let path = format!("results/{}_aco.json", tag);

    let f         = &result.best_fitness;
    let best_path = decode(&result.best_moves, grid, (0, 0));

    let path_json: String = best_path
        .iter()
        .map(|(r, c)| format!("[{},{}]", r, c))
        .collect::<Vec<_>>()
        .join(",");

    let json = format!(
        "{{\n\
        \t\"instance\": \"{tag}\",\n\
        \t\"best_fitness\": {:.2},\n\
        \t\"distance\": {:.2},\n\
        \t\"revisits\": {},\n\
        \t\"unvisited\": {},\n\
        \t\"best_moves\": \"{}\",\n\
        \t\"best_path\": [{}]\n\
        }}",
        f.total, f.distance, f.revisits, f.unvisited,
        fmt_moves(&result.best_moves),
        path_json,
    );

    fs::write(&path, json).expect("could not write JSON file");
    println!("saved → {}", path);
}
