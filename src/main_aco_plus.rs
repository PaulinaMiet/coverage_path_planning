mod shared;
mod aco_plus;

use shared::*;
use aco_plus::AcoConfig;
use std::fs;
use std::io::Write;


const TAGUCHI_MODE: bool = true;

const REPS: usize = 10;

fn main() {
    if TAGUCHI_MODE {
        run_taguchi();
    } else {
        run_normal();
    }
}

fn run_normal() {
    let grid = parse_grid(INSTANCE);
    let (rw, uw) = penalty_weights(&grid);

    println!("\n=== {} ===", INSTANCE);
    println!("free cells: {}  |  weights — revisit: {:.0}  unvisited: {:.0}\n",
        free_cells(&grid), rw, uw);

    print_neighbour_summary(&grid);


    const N_ITERATIONS:  usize = 10_000;
    const RESTART_AFTER: usize = 250;

    let mut cfg = AcoConfig::default_for(&grid);
    cfg.n_ants        = 100;
    cfg.alpha         = 0.5;
    cfg.beta          = 5.0;
    cfg.rho           = 0.05;
    cfg.q0            = 0.9;
    cfg.local_rho     = 0.1;
    cfg.tabu_size     = 0;
    cfg.n_iterations  = N_ITERATIONS;
    cfg.restart_after = RESTART_AFTER;

    println!("Algorithm      : ACO+ (ACS local update, MMAS adaptive bounds)");
    println!("q0             : {}  |  local_rho: {}  |  tabu_size: {}", cfg.q0, cfg.local_rho, cfg.tabu_size);
    println!("n_iterations   : {}  |  restart_after: {}", cfg.n_iterations, cfg.restart_after);
    println!("solution_length: {}\n", cfg.solution_length);

    let result = aco_plus::run(&grid, &cfg);

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


fn run_taguchi() {
    let grid = parse_grid(INSTANCE);
    let stem = instance_stem(INSTANCE);

    let l18: [[usize; 6]; 18] = [
        [0, 0, 0, 0, 0, 0], // row  1
        [1, 1, 1, 1, 1, 1], // row  2
        [2, 2, 2, 2, 2, 2], // row  3
        [0, 0, 1, 1, 2, 2], // row  4
        [1, 1, 2, 2, 0, 0], // row  5
        [2, 2, 0, 0, 1, 1], // row  6
        [0, 1, 0, 2, 1, 2], // row  7
        [1, 2, 1, 0, 2, 0], // row  8
        [2, 0, 2, 1, 0, 1], // row  9
        [0, 2, 2, 1, 1, 0], // row 10
        [1, 0, 0, 2, 2, 1], // row 11
        [2, 1, 1, 0, 0, 2], // row 12
        [0, 1, 2, 0, 2, 1], // row 13
        [1, 2, 0, 1, 0, 2], // row 14
        [2, 0, 1, 2, 1, 0], // row 15
        [0, 2, 1, 2, 0, 1], // row 16
        [1, 0, 2, 0, 1, 2], // row 17
        [2, 1, 0, 1, 2, 0], // row 18
    ];

    // Parameter levels
    // Shared with Basic ACO: n_ants, alpha, beta, rho, q0
    // ACO+ only: local_rho
    let n_ants_lvl     = [20_usize,  50,    100  ];
    let alpha_lvl     = [0.5_f64,   1.0,   2.0   ];
    let beta_lvl      = [1.0_f64,   3.0,   5.0   ];
    let rho_lvl       = [0.01_f64,  0.05,  0.1   ];
    let q0_lvl        = [0.5_f64,   0.7,   0.9  ];
    let local_rho_lvl = [0.05_f64,  0.1,   0.2   ];
    let n_iters: usize = 1500;



    ensure_results_dir();

    let csv_path = format!("results/taguchi_aco_plus_{}_convergence.csv", stem);
    let mut csv_file = fs::File::create(&csv_path).expect("could not create convergence CSV");
    writeln!(csv_file, "row,iteration,mean_fitness").unwrap();

    let mut json_rows: Vec<String> = Vec::with_capacity(18);

    println!(
        "\n=== Taguchi L18  |  ACO+  |  {}  |  {} reps/row ===\n",
        stem, REPS
    );
    println!("  Parameters tested:");
    println!("    n_ants     : {:?}", n_ants_lvl);
    println!("    alpha      : {:?}", alpha_lvl);
    println!("    beta       : {:?}", beta_lvl);
    println!("    rho        : {:?}", rho_lvl);
    println!("    q0         : {:?}", q0_lvl);
    println!("    local_rho  : {:?}  (ACO+ only)", local_rho_lvl);
    println!();

    for (row_idx, row) in l18.iter().enumerate() {
        let n_ants    = n_ants_lvl[row[0]];
        let alpha     = alpha_lvl[row[1]];
        let beta      = beta_lvl[row[2]];
        let rho       = rho_lvl[row[3]];
        let q0        = q0_lvl[row[4]];
        let local_rho = local_rho_lvl[row[5]];

        let cfg = AcoConfig {
            n_ants,
            n_iterations: n_iters,
            solution_length: free_cells(&grid) * 2,
            alpha,
            beta,
            rho,
            local_rho,
            q0,
            tabu_size: 0,
            restart_after: 0,
        };

        let mut rep_fitnesses: Vec<f64> = Vec::with_capacity(REPS);
        let mut best_overall            = f64::MAX;
        let mut best_moves: Vec<Move>   = Vec::new();

        let mut conv_sum = vec![0.0_f64; n_iters];

        for _ in 0..REPS {
            let result = aco_plus::run(&grid, &cfg);

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

        let mean_fit = rep_fitnesses.iter().sum::<f64>() / REPS as f64;
        let variance = rep_fitnesses
            .iter()
            .map(|x| (x - mean_fit).powi(2))
            .sum::<f64>()
            / REPS as f64;
        let std_dev  = variance.sqrt();
        let mse      = rep_fitnesses.iter().map(|x| x * x).sum::<f64>() / REPS as f64;
        let sn_ratio = -10.0 * mse.log10();

        for (i, &sum) in conv_sum.iter().enumerate() {
            writeln!(csv_file, "{},{},{:.4}", row_idx + 1, i, sum / REPS as f64).unwrap();
        }

        let best_path = decode(&best_moves, &grid);
        let best_fit  = evaluate(&best_path, &grid);

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
             \t\t\"params\": {{ \"n_ants\": {}, \"alpha\": {}, \"beta\": {}, \"rho\": {}, \"q0\": {}, \"local_rho\": {} }},\n\
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
            n_ants, alpha, beta, rho, q0, local_rho,
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
            "  row {:2}/18  |  mean={:.2}  best={:.2}  S/N={:.2}  (local_rho={} n_ants={})",
            row_idx + 1, mean_fit, best_overall, sn_ratio, local_rho, n_ants,
        );
    }

    let json_path    = format!("results/taguchi_aco_plus_{}.json", stem);
    let json_content = format!(
        "{{\n\
         \t\"algorithm\": \"aco_plus\",\n\
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