use std::path::Path;
use log::{info, warn};

#[allow(clippy::too_many_arguments)] // In a hurry :)
pub fn repair_continuous_and_save(
    mps_file: &Path,
    original_mps_file: &Path,
    output_file: &Path,
    presolved_output_path: &Path,
    presolved_solution_path: &Path,
    var_info: &[mpsparser::Variable],
    solution: &[f64],
    best_objective: &mut f64,
) {
    use grb::prelude::*;

    let mut env = grb::Env::new("").unwrap();
    env.set(grb::param::Threads, 1).unwrap();
    env.set(grb::param::OutputFlag, 0).unwrap();
    let mut model = Model::read_from(&mps_file.to_string_lossy(), &env).unwrap();
    let vars = model.get_vars().unwrap();

    for (var_idx, value) in solution.iter().enumerate() {
        let var = model
            .get_var_by_name(&var_info[var_idx].name)
            .unwrap()
            .unwrap();
        assert!(var == vars[var_idx]);
        let var = vars[var_idx];

        if matches!(var_info[var_idx].var_type, mpsparser::VarType::Integer) {
            model.set_obj_attr(attr::LB, &var, *value).unwrap();
            model.set_obj_attr(attr::UB, &var, *value).unwrap();
        }
    }

    model.optimize().unwrap();
    if model.status().unwrap() == Status::Optimal {
        let lp_objective = model.get_attr(attr::ObjVal).unwrap();
        info!("LP solution found with objective{}", lp_objective);

        let is_presolved = mps_file != original_mps_file;
        if !is_presolved {
            if lp_objective < *best_objective {
                *best_objective = lp_objective;
                info!(
                    "Writing objective {} from original mps solution {}",
                    lp_objective,
                    output_file.to_string_lossy()
                );
                model.write(&output_file.to_string_lossy()).unwrap();
            }
        } else {
            // try to insert it into the original model file by variable name just in case
            // the presolve did major changes.

            let saved_through_original_mps = if is_presolved {
                let orig_save = try_saving_from_original_model(
                    &model,
                    original_mps_file,
                    env,
                    solution,
                    var_info,
                    output_file,
                    *best_objective,
                );
                if let Ok(original_objective) = orig_save {
                    if let Some(original_objective) = original_objective {
                        *best_objective = original_objective;
                    }
                    true
                } else {
                    warn!("Could not transfer solution to original MPS input file. Saving presolved model and solution.");
                    false
                }
            } else {
                false
            };

            if !saved_through_original_mps && lp_objective < *best_objective {
                *best_objective = lp_objective;
                info!("Copying presolved model to the input mps directory.");
                std::fs::copy(mps_file, presolved_output_path).unwrap();
                info!(
                    "Writing objective {} presolved mps solution {}",
                    lp_objective,
                    presolved_solution_path.to_string_lossy()
                );
                model
                    .write(&presolved_solution_path.to_string_lossy())
                    .unwrap();
            }
        }
    } else {
        warn!("No LP solution found.");
    }
}

fn try_saving_from_original_model(
    _possibly_presolved_model: &grb::Model,
    original_mps_file: &Path,
    env: grb::Env,
    solution: &[f64],
    var_info: &[mpsparser::Variable],
    output_file: &Path,
    best_objective: f64,
) -> Result<Option<f64>, grb::Error> {
    use grb::prelude::*;

    info!("loading model");

    let mut original_model = Model::read_from(&original_mps_file.to_string_lossy(), &env)?;
    info!("setting values");
    for (var_idx, value) in solution.iter().enumerate() {
        if let Some(var) = original_model.get_var_by_name(&var_info[var_idx].name)? {
            // let old_bound_ok = if let Some(possibly_presolved_var) =
            //     possibly_presolved_model.get_var_by_name(&var_info[var_idx].name)?
            // {
            //     let possibly_presolved_lb =
            //         possibly_presolved_model.get_obj_attr(attr::LB, &possibly_presolved_var)?;
            //     let possibly_presolved_ub =
            //         possibly_presolved_model.get_obj_attr(attr::UB, &possibly_presolved_var)?;
            //     *value + 1.0e-6 >= possibly_presolved_lb && *value - 1.0e-6 <= possibly_presolved_ub
            // } else {
            //     true
            // };

            // let new_bound_ok = {
            //     let lb = original_model.get_obj_attr(attr::LB, &var)?;
            //     let ub = original_model.get_obj_attr(attr::UB, &var)?;
            //     *value + 1.0e-6 >= lb && *value - 1.0e-6 <= ub
            // };

            // if old_bound_ok && new_bound_ok {
            if matches!(var_info[var_idx].var_type, mpsparser::VarType::Integer) {
                original_model.set_obj_attr(attr::LB, &var, *value)?;
                original_model.set_obj_attr(attr::UB, &var, *value)?;
            }
            // } else {
            //     warn!(
            //         "Variable {} does not map back to original model",
            //         var_info[var_idx].name
            //     );
            // }
        } else {
            warn!(
                "Unknown variable in original model: {:?}",
                var_info[var_idx].name
            )
        }
    }
    info!("starting optimize");
    original_model.optimize()?;
    if original_model.status()? == Status::Optimal {
        let objective = original_model.get_attr(attr::ObjVal)?;
        if objective < best_objective {
            info!(
                "Writing objective {} original mps solution {}",
                objective,
                output_file.to_string_lossy()
            );
            original_model.write(&output_file.to_string_lossy())?;
            Ok(Some(objective))
        } else {
            info!("Solution was worse than the previous best.");
            Ok(None)
        }
    } else {
        warn!(
            "Solution in original mps failed {:?}",
            original_model.status()?
        );
        Err(grb::Error::ModelObjectMismatch)
    }
}
