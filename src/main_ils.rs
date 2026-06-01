mod ils;
mod shared;

use ils::{IlsConfig, IlsResult};
use shared::*;

const INSTANCE: &str = "instances/cpp_10x10_line.txt";

fn main() {
    let grid = parse_grid(INSTANCE);
    let (revis_pen, unvis_pen) = penalty_weights(&grid);

    println!("\n=== {} ===", INSTANCE);
    println!(
        "free cells: {}  |  weights — revisit: {:.0}  unvisited: {:.0}\n",
        free_cells(&grid),
        revis_pen,
        unvis_pen
    );

    let cfg = IlsConfig::default_for(&grid);

    println!("Computing MST...");
    let mst = ils::compute_mst(&grid);
    println!("MST computed with {} edges.", mst.len());
    if let Some(edge) = mst.first() {
        println!(
            "Sample edge: {:?} -> {:?} (weight: {})",
            edge.u, edge.v, edge.weight
        );
    }

    let _result = ils::ils_run(&grid, &cfg);
}
