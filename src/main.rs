use log::info;
use lsmip::{UNBOUNDED, USE_EXPONENTIAL_WEIGHT_DECAY};
use mpsparser::{MPSInstance, Number};
use std::collections::HashMap;
use std::{path::PathBuf, sync::RwLock};
use structopt::StructOpt;

#[cfg(feature = "gurobi")]
const USE_PRESOLVE: bool = true;
#[cfg(not(feature = "gurobi"))]
const USE_PRESOLVE: bool = false;

const NUM_THREADS: usize = 8;

#[cfg(feature = "gurobi")]
fn presolve(
    path: &std::path::Path,
    python_path: &Option<String>,
    convert_script_path: &str,
    temp_path: &str,
) -> (std::path::PathBuf, MPSInstance) {
    println!("instance:{}", path.file_name().unwrap().to_string_lossy());

    let mut filename = path.file_name().unwrap().to_string_lossy().to_string();
    filename.push_str("_presolved.mps");
    let mut other_path = PathBuf::from(temp_path);
    other_path.push(filename);

    info!("Presolving");
    let python_path = if let Some(python_path) = python_path {
        PathBuf::from(python_path)
    } else {
        which::which("python").unwrap()
    };

    let output = std::process::Command::new(python_path)
        .args([
            convert_script_path,
            "--presolve",
            &path.to_string_lossy(),
            &other_path.to_string_lossy(),
        ])
        .output()
        .expect("failed to execute process");
    println!("python output: {:?}", output);
    info!("Presolved. File exists: {}", other_path.exists());

    let mps_content = std::fs::read(&other_path).unwrap();
    (
        other_path,
        mpsparser::parse(mps_content.as_slice()).unwrap(),
    )
}

#[derive(Debug, StructOpt)]
#[structopt()]
struct Opts {
    #[structopt(name = "INPUT")]
    input_file: String,

    #[structopt(name = "OUTPUT")]
    output_file: String,

    #[cfg(feature = "gurobi")]
    #[structopt(long)]
    python_path: Option<String>,

    #[cfg(feature = "gurobi")]
    #[structopt(long)]
    convert_script_path: String,

    #[cfg(feature = "gurobi")]
    #[structopt(long)]
    temp_path: String,
}

fn main() {
    pretty_env_logger::env_logger::Builder::from_env(
        pretty_env_logger::env_logger::Env::default().default_filter_or("trace"),
    )
    .init();

    let _p = hprof::enter("solver");
    let opt = Opts::from_args();
    println!("{:?}", opt);

    let path = std::path::Path::new(&opt.input_file);

    println!("instance:{}", path.file_name().unwrap().to_string_lossy());
    let original_mps_obj = {
        let mps_content = std::fs::read(path).unwrap();
        mpsparser::parse(mps_content.as_slice()).unwrap()
    };
    let original_mps = &original_mps_obj;

    let original_path = path;
    let output_file = &opt.output_file;

    let presolved: RwLock<Option<(PathBuf, MPSInstance)>> = RwLock::new(None);
    let presolved_borrowed = &presolved;
    let mut presolved_output_path = original_path.to_owned();

    let mut presolved_filename = original_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    presolved_filename.push_str("_presolved.mps");
    presolved_output_path.set_file_name(presolved_filename);

    let mut presolved_solution_path = presolved_output_path.to_owned();
    presolved_solution_path.set_extension("sol");

    let (sol_tx, sol_rx) = std::sync::mpsc::channel::<(bool, Vec<f64>)>();

    crossbeam::scope(|s| {
        // If we have gurobi access, run the presolve on a separate thread.
        #[cfg(feature = "gurobi")]
        s.spawn(move |_| {
            let mut mps = presolved_borrowed.write().unwrap();
            let (presolved_path, presolved_mps) = presolve(
                path,
                &opt.python_path,
                &opt.convert_script_path,
                &opt.temp_path,
            );
            *mps = Some((presolved_path, presolved_mps));
        });

        // Reserve 1 thread for presolve.
        let mut thread_idxs = (1..NUM_THREADS).collect::<Vec<_>>();
        if NUM_THREADS == 1 || !USE_PRESOLVE {
            thread_idxs.push(NUM_THREADS);
        }

        for thread_idx in thread_idxs {
            let seed = thread_idx as u8;
            let relax_continuous = thread_idx % 2 == 0;
            let use_presolved = USE_PRESOLVE && (thread_idx / 4) % 2 == 1;
            let decay_factor = if (thread_idx / 2) % 2 == 0 || !USE_EXPONENTIAL_WEIGHT_DECAY {
                1.0
            } else {
                0.9999
            };

            let sol_tx = sol_tx.clone();
            s.spawn(move |_| {
                let lock = use_presolved.then(|| presolved_borrowed.read().unwrap());

                let mps = if use_presolved {
                    &lock.as_ref().unwrap().as_ref().unwrap().1
                } else {
                    original_mps
                };

                let mut solver =
                    lsmip::solver::Solver::with_seed(thread_idx, seed, decay_factor);

                translate_mps(&mut solver, mps, relax_continuous);

                solver.solve(|solution| {
                    let solution = solution.iter().map(|v| *v as f64).collect::<Vec<_>>();
                    sol_tx.send((use_presolved, solution)).unwrap();
                });
            });
        }

        drop(sol_tx);

        let mut presolved_lock = None;
        let mut best_objective = f64::INFINITY;
        for (use_presolved, solution) in sol_rx.into_iter() {
            if use_presolved && presolved_lock.is_none() {
                presolved_lock = Some(presolved.read().unwrap());
            }

            let mps = if use_presolved {
                &presolved_lock.as_ref().unwrap().as_ref().unwrap().1
            } else {
                original_mps
            };

            let path = if use_presolved {
                &presolved_lock.as_ref().unwrap().as_ref().unwrap().0
            } else {
                path
            };

            #[cfg(feature = "gurobi")]
            lsmip::lp::repair_continuous_and_save(
                path,
                original_path,
                std::path::Path::new(&output_file),
                &presolved_output_path,
                &presolved_solution_path,
                &mps.variables,
                &solution,
                &mut best_objective,
            );

            #[cfg(not(feature = "gurobi"))]
            {
                let objective = mps.calculate_objective(&solution);

                println!("Thread found solution with objective: {}", objective);

                if objective < best_objective {
                    best_objective = objective;
                    println!("New best objective: {}", best_objective);
                    if let Err(e) = mps.save_solution(&opt.output_file, &solution, best_objective) {
                        eprintln!("Error saving solution: {}", e);
                    } else {
                        println!("Solution saved to: {}", opt.output_file);
                    }
                }
            }
        }
    })
    .unwrap();

    hprof::profiler().print_timing();
}

/// Add the variables and constraints from the MPSInstance to the Feasibility Jump solver instance.
fn translate_mps(solver: &mut lsmip::solver::Solver, mps: &MPSInstance, relax_continuous: bool) {
    let objective_coeffs = mps
        .objective()
        .iter()
        .flat_map(|(_, v)| v.iter())
        .map(|c| (c.var, c.coeff.as_f64() as f32))
        .collect::<HashMap<usize, f32>>();

    // Add variables
    for (var_idx, var) in mps.variables.iter().enumerate() {
        let lb = var.lb.map(Number::as_f64).unwrap_or(0.) as f32;
        let ub = var.ub.map(Number::as_f64).unwrap_or(UNBOUNDED) as f32;
        solver.add_var(
            var.var_type,
            lb,
            ub,
            objective_coeffs.get(&var_idx).copied().unwrap_or(0.),
        );
    }

    // Add constraints
    for constraint in mps.constraints.iter() {
        if matches!(constraint.rowtype, mpsparser::RowType::None) {
            continue;
        }
        solver.add_constraint(
            constraint.rowtype,
            constraint.rhs.map(Number::as_f64).unwrap_or(0.) as f32,
            &constraint
                .cells
                .iter()
                .map(|c| (c.var, c.coeff.as_f64() as f32))
                .collect::<Vec<_>>(),
            relax_continuous,
        );
    }
}
