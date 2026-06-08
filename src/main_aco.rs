mod aco;
mod shared;

use aco::AcoConfig;
use shared::*;
use std::fs;
use std::io::Write;

// ── Mode toggle ───────────────────────────────────────────────
// true  → Taguchi L18 experiment (no normal CSV/JSON output)
// false → normal single run (default behaviour)
const TAGUCHI_MODE: bool = true;

// Number of replications per L18 row (ACO is stochastic)
const REPS: usize = 10;

fn main() {
    if TAGUCHI_MODE {
        run_taguchi();
    } else {
        run_normal();
    }
}

// ── Normal run ────────────────────────────────────────────────

fn run_normal() {
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

// ── Taguchi L18 run ───────────────────────────────────────────

fn run_taguchi() {
    let grid = parse_grid(INSTANCE);
    let stem = instance_stem(INSTANCE);

    // ── L18 orthogonal array ──────────────────────────────────
    // Each row is [n_ants_lvl, alpha_lvl, beta_lvl, rho_lvl, q0_lvl].
    // Values are level indices (0/1/2) into the level arrays below.
    // Derived from L18(2^1 × 3^7) using columns B–F.
    // Verified orthogonal: every pair of level combinations appears exactly twice.
    let l18: [[usize; 5]; 18] = [
        [0, 0, 0, 0, 0], // row  1
        [0, 1, 1, 1, 1], // row  2
        [0, 2, 2, 2, 2], // row  3
        [1, 0, 0, 1, 1], // row  4
        [1, 1, 1, 2, 2], // row  5
        [1, 2, 2, 0, 0], // row  6
        [2, 0, 1, 0, 2], // row  7
        [2, 1, 2, 1, 0], // row  8
        [2, 2, 0, 2, 1], // row  9
        [0, 0, 2, 2, 1], // row 10
        [0, 1, 0, 0, 2], // row 11
        [0, 2, 1, 1, 0], // row 12
        [1, 0, 1, 2, 0], // row 13
        [1, 1, 2, 0, 1], // row 14
        [1, 2, 0, 1, 2], // row 15
        [2, 0, 2, 1, 2], // row 16
        [2, 1, 0, 2, 0], // row 17
        [2, 2, 1, 0, 1], // row 18
    ];

    // Parameter levels
    let n_ants_lvl = [20_usize, 50, 100];
    let alpha_lvl  = [0.5_f64,  1.0, 2.0];
    let beta_lvl   = [1.0_f64,  3.0, 5.0];
    let rho_lvl    = [0.01_f64, 0.05, 0.1];
    let q0_lvl     = [0.5_f64,  0.7, 0.9];

    let n_iters: usize = 1500;

    ensure_results_dir();

    // Convergence CSV — one averaged running-best curve per L18 row
    let csv_path = format!("results/taguchi_{}_convergence.csv", stem);
    let mut csv_file = fs::File::create(&csv_path).expect("could not create convergence CSV");
    writeln!(csv_file, "row,iteration,mean_fitness").unwrap();

    let mut json_rows: Vec<String> = Vec::with_capacity(18);

    println!(
        "\n=== Taguchi L18  |  {}  |  {} reps/row ===\n",
        stem, REPS
    );

    for (row_idx, row) in l18.iter().enumerate() {
        let n_ants = n_ants_lvl[row[0]];
        let alpha  = alpha_lvl[row[1]];
        let beta   = beta_lvl[row[2]];
        let rho    = rho_lvl[row[3]];
        let q0     = q0_lvl[row[4]];

        let cfg = AcoConfig {
            n_ants,
            n_iterations: n_iters,
            solution_length: free_cells(&grid) * 2,
            alpha,
            beta,
            rho,
            tau_init: 1.0,
            q0,
        };

        let mut rep_fitnesses: Vec<f64> = Vec::with_capacity(REPS);
        let mut best_overall             = f64::MAX;
        let mut best_moves: Vec<Move>   = Vec::new();

        // Accumulates running-best at each iteration across all reps (for averaging)
        let mut conv_sum = vec![0.0_f64; n_iters];

        for _ in 0..REPS {
            let result = aco::aco_run(&grid, &cfg);

            // Build running-best convergence curve from iteration history
            let mut running = f64::MAX;
            for (i, log) in result.history.iter().enumerate() {
                running = running.min(log.fitness);
                conv_sum[i] += running;
            }

            let final_fit = result.best_fitness.total;
            rep_fitnesses.push(final_fit);

            if final_fit < best_overall {
                best_overall = final_fit;
                best_moves   = result.best_moves.clone();
            }
        }

        // ── Statistics ────────────────────────────────────────
        let mean_fit = rep_fitnesses.iter().sum::<f64>() / REPS as f64;
        let variance = rep_fitnesses
            .iter()
            .map(|x| (x - mean_fit).powi(2))
            .sum::<f64>()
            / REPS as f64;
        let std_dev  = variance.sqrt();
        // S/N ratio — smaller-is-better: SN = -10 × log10( mean(yi²) )
        let mse      = rep_fitnesses.iter().map(|x| x * x).sum::<f64>() / REPS as f64;
        let sn_ratio = -10.0 * mse.log10();

        // Write averaged convergence rows to CSV
        for (i, &sum) in conv_sum.iter().enumerate() {
            writeln!(csv_file, "{},{},{:.4}", row_idx + 1, i, sum / REPS as f64).unwrap();
        }

        // Best path detail (re-evaluated for clean fitness breakdown)
        let best_path = decode(&best_moves, &grid);
        let best_fit  = evaluate(&best_path, &grid);

        // ── Build JSON entry for this row ─────────────────────
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
             \t\t\"row\": {},\n\
             \t\t\"params\": {{ \"n_ants\": {}, \"alpha\": {}, \"beta\": {}, \"rho\": {}, \"q0\": {} }},\n\
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
            row_idx + 1,
            n_ants, alpha, beta, rho, q0,
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
            "  row {:2}/18  |  mean={:.2}  best={:.2}  S/N={:.2}",
            row_idx + 1,
            mean_fit,
            best_overall,
            sn_ratio
        );
    }

    // ── Write summary JSON ────────────────────────────────────
    let json_path    = format!("results/taguchi_{}.json", stem);
    let json_content = format!(
        "{{\n\
         \t\"instance\": \"{}\",\n\
         \t\"reps_per_row\": {},\n\
         \t\"n_iterations\": {},\n\
         \t\"rows\": [\n{}\n\t]\n}}",
        stem,
        REPS,
        n_iters,
        json_rows.join(",\n")
    );
    fs::write(&json_path, json_content).expect("could not write Taguchi JSON");

    println!("\nTaguchi done.");
    println!("  summary     → {}", json_path);
    println!("  convergence → {}", csv_path);
}
