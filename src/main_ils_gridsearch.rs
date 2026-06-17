mod ils;
mod shared;

use ils::{IlsConfig, StartingStrategy};
use shared::*;
use std::fs;
use std::io::Write;

const REPS: usize = 10; // replications per combination
const TOTAL_BUDGET: usize = 75000; // budget

fn main() {
    let instances = [
        "instances/cpp_5x5_chunk.txt",
        "instances/cpp_5x5_line.txt",
        "instances/cpp_5x5_sparse.txt",
        "instances/cpp_10x10_chunk.txt",
        "instances/cpp_10x10_line.txt",
        "instances/cpp_10x10_sparse.txt",
        "instances/cpp_20x20_chunk.txt",
        "instances/cpp_20x20_line.txt",
        "instances/cpp_20x20_sparse.txt",
    ];

    for instance in instances {
        run_gridsearch(instance);
    }
}

fn run_gridsearch(instance_path: &str) {
    let grid = parse_grid(instance_path);
    let stem = instance_stem(instance_path);

    let strategy_lvl = [
        StartingStrategy::Mst,
        StartingStrategy::Snake,
        StartingStrategy::Random,
    ];
    let ls_iters_lvl = [100_usize, 250, 500];
    let perturb_lvl = [2_usize, 5, 10];

    ensure_results_dir();

    let csv_path = format!("results/gridsearch_ils_{}_convergence.csv", stem);
    let mut csv_file = fs::File::create(&csv_path).expect("could not create convergence CSV");
    writeln!(csv_file, "combination,iteration,mean_fitness").unwrap();

    let mut json_rows: Vec<String> = Vec::new();

    println!(
        "\n=== ILS Grid Search  |  {}  |  {} reps/combo ===\n",
        stem, REPS
    );

    let mut combo_idx = 0;
    for &strategy in &strategy_lvl {
        for &ls_iterations in &ls_iters_lvl {
            for &perturb_size in &perturb_lvl {
                combo_idx += 1;

                // budget = n_iterations * ls_iterations
                let n_iterations = TOTAL_BUDGET / ls_iterations;

                let cfg = IlsConfig {
                    n_iterations,
                    ls_iterations,
                    perturb_size,
                    strategy,
                };

                let mut rep_fitnesses: Vec<f64> = Vec::with_capacity(REPS);
                let mut best_overall = f64::MAX;
                let mut best_moves: Vec<Move> = Vec::new();

                let mut conv_sum = vec![0.0_f64; n_iterations];

                for _ in 0..REPS {
                    let mut rng = rand::thread_rng();
                    let result = ils::ils_run(&grid, &cfg, &mut rng);

                    let mut running = f64::MAX;
                    for (i, log) in result.history.iter().enumerate() {
                        if i < n_iterations {
                            running = running.min(log.fitness);
                            conv_sum[i] += running;
                        }
                    }

                    let final_fit = result.best_fitness.total;
                    rep_fitnesses.push(final_fit);

                    if final_fit < best_overall {
                        best_overall = final_fit;
                        best_moves = result.best_moves.clone();
                    }
                }

                let mean_fit = rep_fitnesses.iter().sum::<f64>() / REPS as f64;
                let variance = rep_fitnesses
                    .iter()
                    .map(|x| (x - mean_fit).powi(2))
                    .sum::<f64>()
                    / REPS as f64;
                let std_dev = variance.sqrt();
                let mse = rep_fitnesses.iter().map(|x| x * x).sum::<f64>() / REPS as f64;
                let sn_ratio = -10.0 * mse.log10();

                for (i, &sum) in conv_sum.iter().enumerate() {
                    writeln!(csv_file, "{},{},{:.4}", combo_idx, i, sum / REPS as f64).unwrap();
                }

                let best_path = decode(&best_moves, &grid);
                let best_fit = evaluate(&best_path, &grid);

                let rep_vals: String = rep_fitnesses
                    .iter()
                    .map(|x| format!("{:.4}", x))
                    .collect::<Vec<_>>()
                    .join(", ");

                let path_json: String = best_path
                    .iter()
                    .map(|(r, c)| format!("[{},{}]", r, c))
                    .collect::<Vec<_>>()
                    .join(", ");

                let row_json = format!(
                    "    {{\n\
                     \t\t\"combination\": {},\n\
                     \t\t\"params\": {{ \"strategy\": \"{:?}\", \"ls_iterations\": {}, \"perturb_size\": {}, \"n_iterations\": {} }},\n\
                     \t\t\"rep_fitnesses\": [{}],\n\
                     \t\t\"mean_fitness\": {:.4},\n\
                     \t\t\"std_dev\": {:.4},\n\
                     \t\t\"sn_ratio\": {:.4},\n\
                     \t\t\"best_fitness\": {:.4},\n\
                     \t\t\"best_distance\": {:.4},\n\
                     \t\t\"best_revisits\": {},\n\
                     \t\t\"best_unvisited\": {},\n\
                     \t\t\"best_moves\": \"{}\",\n\
                     \t\t\"best_path\": [{}]\n\
                     \t}}",
                    combo_idx,
                    strategy,
                    ls_iterations,
                    perturb_size,
                    n_iterations,
                    rep_vals,
                    mean_fit,
                    std_dev,
                    sn_ratio,
                    best_fit.total,
                    best_fit.distance,
                    best_fit.revisits,
                    best_fit.unvisited,
                    fmt_moves(&best_moves),
                    path_json,
                );

                json_rows.push(row_json);

                println!(
                    "  combo {:2}/27  |  mean={:.2}  best={:.2}  S/N={:.2}",
                    combo_idx, mean_fit, best_overall, sn_ratio
                );
            }
        }
    }

    let json_path = format!("results/gridsearch_ils_{}.json", stem);
    let json_content = format!(
        "{{\n\
         \t\"instance\": \"{}\",\n\
         \t\"reps_per_combo\": {},\n\
         \t\"total_budget\": {},\n\
         \t\"combinations\": [\n{}\n\t]\n}}",
        stem,
        REPS,
        TOTAL_BUDGET,
        json_rows.join(",\n")
    );
    fs::write(&json_path, json_content).expect("could not write Grid Search JSON");

    println!("\nGrid Search done for {}.", stem);
    println!("  summary     → {}", json_path);
    println!("  convergence → {}", csv_path);
}
