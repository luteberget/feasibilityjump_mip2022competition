use log::{error, info};
use mpsparser::{RowType, VarType};
use ordered_float::OrderedFloat;
use rand::{prelude::SliceRandom, Rng, SeedableRng};
use std::{
    cmp::Reverse,
    time::{Duration, Instant},
};

use crate::{interval::Interval, USE_DAMPING, USE_SIZE_WEIGHTING};

pub struct Solver {
    vars: Vec<Var>,
    constraints: Vec<Constraint>,
    solution: Vec<f32>,

    unsat_constraints: Vec<u32>,
    good_choices: Vec<Choice>,
    best_shift_buffer: Vec<(f32, f32)>,

    weight_bumps: usize,
    rng: rand_xoshiro::SplitMix64,
    best_unsat: usize,

    incumbent_objective: f64,
    best_objective: f64,
    objective_weight: f32,
    n_nonzeros: usize,
    last_improvement_time: usize,
    last_dampen_time: usize,
    time: usize,

    weight_update_decay: f64,
    weight_update_increment: f64,

    solver_id: usize,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
struct Choice {
    var: u32,
    dir: Direction,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
enum Direction {
    Up,
    Down,
    Jump,
}

struct Var {
    vartype: VarType,
    lb: f32,
    ub: f32,
    time_stamp: usize,
    good_up_idx: i32,
    good_down_idx: i32,
    good_jump_idx: i32,
    up_score: f32,
    down_score: f32,
    jump_score: f32,
    jump_value: f32,
    coeffs: Vec<(u32, f32)>,
    step_size: f32,
    back_and_forth: Option<(i32, f32)>,
    default_value: f32,
    objective_coeff: f32,
}

#[derive(Debug)]
struct Constraint {
    sense: RowType,
    rhs: f32,
    coeffs: Vec<(u32, f32)>,
    weight: f32,
    incumbent_lhs: f32,
    unsat_idx: i32,
}

impl Constraint {
    pub fn score(&self, lhs: f32) -> f32 {
        match self.sense {
            mpsparser::RowType::None => 0.,
            mpsparser::RowType::Equal => -(lhs - self.rhs).abs(),
            mpsparser::RowType::Lte => -((lhs - self.rhs).max(0.)),
            mpsparser::RowType::Gte => -((self.rhs - lhs).max(0.)),
        }
    }
}

const RANDOM_VAR: f32 = 0.001;
const RANDOM_CELL: f32 = 0.01;
const MAX_GOOD_CHOICES_TO_CHECK: usize = 15;

const USE_RECOMPUTE_JUMP: bool = true;

// const CONSTRAINT_WEIGHT_INCREASE: f32 = 1.0;
// const OBJECTIVE_WEIGHT_INCREASE: f32 = 1.0;

const VIOLATION_TOLERANCE: f32 = 1.0e-5;
const EQ_TOLERANCE: f32 = 1.0e-5;

impl Solver {
    pub fn new(solver_id: usize, weight_update_decay: f64) -> Solver {
        Solver {
            vars: Vec::new(),
            constraints: Vec::new(),
            solution: Vec::new(),
            unsat_constraints: Vec::new(),
            good_choices: Vec::new(),
            best_shift_buffer: Default::default(),
            weight_bumps: 0,
            rng: rand_xoshiro::SplitMix64::from_seed([1, 0, 0, 0, 0, 0, 0, 0]),
            best_unsat: usize::MAX,
            best_objective: f64::INFINITY,
            objective_weight: 0.0,
            incumbent_objective: 0.,
            n_nonzeros: 0,
            last_improvement_time: 0,
            last_dampen_time: 0,
            time: 0,

            weight_update_decay,
            weight_update_increment: 1.0,
            solver_id,
        }
    }

    pub fn with_seed(solver_id: usize, seed: u8, weight_update_decay: f64) -> Solver {
        let mut s = Self::new(solver_id, weight_update_decay);
        s.rng = rand_xoshiro::SplitMix64::from_seed([seed, 0, 0, 0, 0, 0, 0, 0]);
        s
    }

    pub fn add_var(&mut self, vartype: VarType, lb: f32, ub: f32, objective_coeff: f32) -> usize {
        let idx = self.vars.len();
        let is_int = matches!(vartype, VarType::Integer);
        let big_domain = (ub - lb) > 265.0;

        let step_size = if is_int && !big_domain {
            1.0
        } else {
            (0.1 * (ub - lb)).max(2.0 * EQ_TOLERANCE)
        };

        let step_size = if is_int { step_size.ceil() } else { step_size };

        self.vars.push(Var {
            step_size,
            vartype,
            lb,
            ub,
            time_stamp: 0,
            up_score: f32::NAN,
            down_score: f32::NAN,
            jump_score: f32::NAN,
            good_jump_idx: -1,
            good_down_idx: -1,
            good_up_idx: -1,
            coeffs: Vec::new(),
            back_and_forth: None,
            default_value: lb,
            jump_value: f32::NAN,
            objective_coeff,
        });
        idx
    }

    pub fn set_initial_value(&mut self, var: usize, value: f32) {
        self.vars[var].default_value = value;
    }

    pub fn step_down(&self, var: usize) -> Option<f32> {
        let curr = self.solution[var];
        let new = match self.vars[var].vartype {
            VarType::Integer => (curr - self.vars[var].step_size)
                .min(curr - 1.0)
                .max(self.vars[var].lb),
            VarType::Continuous => (curr - self.vars[var].step_size).max(self.vars[var].lb),
        };
        ((new - curr).abs() > 0.1 * EQ_TOLERANCE).then_some(new)
    }

    pub fn step_up(&self, var: usize) -> Option<f32> {
        let curr = self.solution[var];
        let new = match self.vars[var].vartype {
            VarType::Integer => (curr + self.vars[var].step_size)
                .max(curr + 1.0)
                .min(self.vars[var].ub),
            VarType::Continuous => (curr + self.vars[var].step_size).min(self.vars[var].ub),
        };
        ((new - curr).abs() > 0.1 * EQ_TOLERANCE).then_some(new)
    }

    pub fn jump_value(&self, var: usize) -> Option<f32> {
        if self.vars[var].jump_value.is_nan() {
            return None;
        }

        ((self.solution[var] - self.vars[var].jump_value).abs() > 0.1 * EQ_TOLERANCE)
            .then(|| self.vars[var].jump_value)
    }

    pub fn add_constraint(
        &mut self,
        sense: RowType,
        mut rhs: f32,
        coeffs: &[(usize, f32)],
        relax_continuous: bool,
    ) -> usize {
        if relax_continuous
            && matches!(sense, RowType::Equal)
            && coeffs
                .iter()
                .any(|(v, _)| matches!(self.vars[*v].vartype, VarType::Continuous))
        {
            self.add_constraint(RowType::Gte, rhs, coeffs, relax_continuous);
            self.add_constraint(RowType::Lte, rhs, coeffs, relax_continuous);
            return usize::MAX;
        }

        let coeffs = coeffs
            .iter()
            .filter_map(|(v, c)| {
                if relax_continuous && matches!(self.vars[*v].vartype, VarType::Continuous) {
                    match sense {
                        RowType::None => panic!(),
                        RowType::Equal => panic!(),
                        RowType::Lte => {
                            if *c >= 0. {
                                rhs -= *c * self.vars[*v].lb;
                            } else {
                                rhs -= *c * self.vars[*v].ub;
                            }
                        }
                        RowType::Gte => {
                            if *c >= 0. {
                                rhs -= *c * self.vars[*v].ub;
                            } else {
                                rhs -= *c * self.vars[*v].lb;
                            }
                        }
                    }
                    None
                } else {
                    Some((*v as _, *c))
                }
            })
            .collect::<Vec<_>>();

        if coeffs.is_empty() {
            // assert!(match sense {
            //     RowType::None => true,
            //     RowType::Equal => (rhs - 0.).abs() < EQ_TOLERANCE,
            //     RowType::Lte => 0. <= rhs,
            //     RowType::Gte => 0. >= rhs,
            // });
            return usize::MAX;
        }

        // let constraint_initial_weight = 1.0 / (coeffs.iter().map(|(_var,coeff)| coeff.abs()).sum::<f32>());
        let constraint_initial_weight = 1.0;

        self.n_nonzeros += coeffs.len();
        let constraint_idx = self.constraints.len();
        self.constraints.push(Constraint {
            sense,
            rhs,
            coeffs,
            weight: constraint_initial_weight,
            incumbent_lhs: f32::NAN,
            unsat_idx: -1,
        });

        for (v, c) in self.constraints[constraint_idx].coeffs.iter() {
            self.vars[*v as usize]
                .coeffs
                .push((constraint_idx as u32, *c));
        }

        constraint_idx
    }

    pub fn dampen_weights(&mut self) {
        info!("id:{} Damping weights.", self.solver_id);
        let sum_weight = self
            .constraints
            .iter()
            .map(|c| c.weight as f64)
            .sum::<f64>();
        let average_weight = sum_weight / self.constraints.len() as f64;
        for c in self.constraints.iter_mut() {
            c.weight = c.weight.min(average_weight as f32);
        }
        self.weight_update_increment = 1.0;
        for var_idx in 0..self.vars.len() {
            self.reset_var_cost(var_idx);
        }
        self.last_dampen_time = self.time;
    }

    pub fn solve(&mut self, mut output: impl FnMut(&[f32])) {
        const MAX_NONIMPROVING_TIME: f64 = 10.0;

        info!("id:{} Starting local search.", self.solver_id);
        let start = Instant::now();
        let mut last_output = Instant::now();
        // let max_dur = Duration::from_secs(600);
        self.init(self.vars.iter().map(|v| v.default_value).collect());

        // let mut prev_choice: Option<Choice> = None;
        for step in 0.. {
            // self.time as f64 / 1_000_000.0 / start.elapsed().as_secs_f64() == 35

            let time_since_last_improvement =
                (self.time - self.last_improvement_time) as f64 / 35_000_000.0;

            if USE_DAMPING
                && (self.time - self.last_improvement_time.max(self.last_dampen_time)) as f64
                    / 35_000_000.0
                    > MAX_NONIMPROVING_TIME / 2.0
            {
                self.dampen_weights();
            }

            if time_since_last_improvement > MAX_NONIMPROVING_TIME {
                // warn!("Breaking because time{} last{} num{}", self.time, self.last_improvement_time, 10_000_000);
                break;
            }

            if self.unsat_constraints.len() < self.best_unsat {
                self.last_improvement_time = self.time;
                self.best_unsat = self.unsat_constraints.len();
            }

            if self.unsat_constraints.is_empty() && self.incumbent_objective < self.best_objective {
                self.last_improvement_time = self.time;
                self.best_objective = self.incumbent_objective;
                info!(
                    "id:{} new best objective {}",
                    self.solver_id, self.best_objective
                );
                output(&self.solution);
            }

            if step % 1_000 == 0 {
                // if start.elapsed() > max_dur {
                //     // panic!("timeout");
                //     return;
                // }

                if last_output.elapsed() > Duration::from_secs(1) {
                    info!(
                        "id:{} step:{} best_viol:{} best_obj:{} weight_bumps:{} unsat:{} good:{} sumstep:{} mtps:{:.2} time:{:.2}",
                        self.solver_id,
                        step,
                        self.best_unsat,self.best_objective,
                        self.weight_bumps,
                        self.unsat_constraints.len(),
                        self.good_choices.len(),
                        self.vars.iter().map(|v| v.step_size).sum::<f32>(),
                        self.time as f64 / 1_000_000.0 / start.elapsed().as_secs_f64(),
                        MAX_NONIMPROVING_TIME - time_since_last_improvement
                    );

                    last_output = Instant::now();
                }
            }

            // debug!("Check_all 1");
            // self.check_all();

            let choice = self.decide();
            // debug!("choice {:?}", choice);
            // if prev_choice.as_ref().map(|v| v.var) == Some(choice.var) {
            //     debug!(
            //         "went back on decision #{} {:?} {:?}",
            //         step, prev_choice, choice
            //     );
            //     panic!();
            // }
            // prev_choice = Some(choice);
            // debug!("Check_all 2");
            // self.check_all();
            self.set_value(choice);
            // debug!("Check_all 3");
            // self.check_all();
            self.vars[choice.var as usize].time_stamp = step;
        }
        // None
    }

    // fn check_all(&self) {
    //     for c in 0..self.constraints.len() {
    //         self.check_constraint(c);
    //     }
    //     for v in 0..self.vars.len() {
    //         self.check_var(v);
    //     }
    // }

    // fn check_constraint(&self, c_idx: usize) {
    //     if matches!(self.constraints[c_idx].sense, RowType::None) {
    //         return;
    //     }

    //     let lhs = self.constraints[c_idx]
    //         .coeffs
    //         .iter()
    //         .map(|(v, c)| *c * self.solution[*v as usize])
    //         .sum::<f32>();
    //     assert!((lhs - self.constraints[c_idx].incumbent_lhs).abs() < EQ_TOLERANCE);
    //     if self.constraints[c_idx].score(self.constraints[c_idx].incumbent_lhs)
    //         < -VIOLATION_TOLERANCE
    //     {
    //         assert!(self.constraints[c_idx].unsat_idx != -1);
    //         assert!(
    //             self.unsat_constraints[self.constraints[c_idx].unsat_idx as usize] == c_idx as _
    //         );
    //     } else {
    //         assert!(self.constraints[c_idx].unsat_idx == -1);
    //         assert!(!self.unsat_constraints.iter().any(|c| *c as usize == c_idx));
    //     }
    // }

    // fn check_var(&self, var: usize) {
    //     // debug!(
    //     //     "var:{} lb:{} ub:{} val:{} up:{:?} down:{:?}",
    //     //     var,
    //     //     self.vars[var].lb,
    //     //     self.vars[var].ub,
    //     //     self.solution[var],
    //     //     self.step_up(var),
    //     //     self.step_down(var)
    //     // );

    //     let up_score = self.vars[var]
    //         .coeffs
    //         .iter()
    //         .map(|(c_idx, coeff)| {
    //             self.check_constraint(*c_idx as usize);
    //             if let Some(var_up) = self.step_up(var) {
    //                 assert!(var_up > self.solution[var]);
    //                 let lhs_old = self.constraints[*c_idx as usize].incumbent_lhs;
    //                 let lhs_up = lhs_old - (*coeff * self.solution[var]) + (*coeff * var_up);

    //                 // debug!(
    //                 //     "constraint {} var {} from {} to {} score {}\n   {:?}",
    //                 //     c_idx,
    //                 //     var,
    //                 //     self.solution[var],
    //                 //     var_up,
    //                 //     self.constraints[*c_idx as usize].score(lhs_up)
    //                 //         - self.constraints[*c_idx as usize].score(lhs_old),
    //                 //     self.constraints[*c_idx as usize]
    //                 // );

    //                 self.objective_weight
    //                     * self.vars[var].objective_coeff
    //                     * (var_up - self.solution[var])
    //                     + self.constraints[*c_idx as usize].weight
    //                         * (self.constraints[*c_idx as usize].score(lhs_up)
    //                             - self.constraints[*c_idx as usize].score(lhs_old))
    //             } else {
    //                 0.
    //             }
    //         })
    //         .sum::<f32>();

    //     if (self.vars[var].up_score - up_score).abs() >= EQ_TOLERANCE {
    //         error!(
    //             "var {} has up score {} should be {}",
    //             var, self.vars[var].up_score, up_score
    //         );
    //         assert!(false);
    //     }

    //     if self.vars[var].up_score > 0. {
    //         assert!(self.vars[var].good_up_idx != -1);
    //         assert!(
    //             self.good_choices[self.vars[var].good_up_idx as usize]
    //                 == Choice {
    //                     var: var as u32,
    //                     dir: Direction::Up
    //                 }
    //         );
    //     } else {
    //         assert!(self.vars[var].good_up_idx == -1);
    //         assert!(!self
    //             .good_choices
    //             .iter()
    //             .any(|c| c.var as usize == var && c.dir == Direction::Up));
    //     }

    //     if self.step_up(var).is_none() {
    //         assert!((self.solution[var] - self.vars[var].ub).abs() < EQ_TOLERANCE);
    //         assert!(self.vars[var].good_up_idx == -1);
    //     }

    //     let down_score = self.vars[var]
    //         .coeffs
    //         .iter()
    //         .map(|(c_idx, coeff)| {
    //             self.check_constraint(*c_idx as usize);
    //             if let Some(var_down) = self.step_down(var) {
    //                 assert!(var_down < self.solution[var]);
    //                 let lhs_old = self.constraints[*c_idx as usize].incumbent_lhs;
    //                 let lhs_down = lhs_old - (*coeff * self.solution[var]) + (*coeff * var_down);

    //                 self.objective_weight
    //                     * self.vars[var].objective_coeff
    //                     * (var_down - self.solution[var])
    //                     + self.constraints[*c_idx as usize].weight
    //                         * (self.constraints[*c_idx as usize].score(lhs_down)
    //                             - self.constraints[*c_idx as usize].score(lhs_old))
    //             } else {
    //                 0.
    //             }
    //         })
    //         .sum::<f32>();

    //     assert!((self.vars[var].down_score - down_score).abs() < EQ_TOLERANCE);
    //     if self.vars[var].down_score > 0. {
    //         assert!(self.vars[var].good_down_idx != -1);
    //         assert!(
    //             self.good_choices[self.vars[var].good_down_idx as usize]
    //                 == Choice {
    //                     var: var as u32,
    //                     dir: Direction::Down
    //                 }
    //         );
    //     } else {
    //         assert!(self.vars[var].good_down_idx == -1);
    //         assert!(!self
    //             .good_choices
    //             .iter()
    //             .any(|c| c.var as usize == var && c.dir == Direction::Down));
    //     }

    //     if self.step_down(var).is_none() {
    //         assert!((self.solution[var] - self.vars[var].lb).abs() < EQ_TOLERANCE);
    //         assert!(self.vars[var].good_down_idx == -1);
    //     }

    //     assert!(!self.vars[var].jump_value.is_nan());
    //     let jump_score = self.vars[var]
    //         .coeffs
    //         .iter()
    //         .map(|(c_idx, coeff)| {
    //             self.check_constraint(*c_idx as usize);
    //             if let Some(var_jump_value) = self.jump_value(var) {
    //                 assert!((var_jump_value - self.solution[var]).abs() > 0.1 * EQ_TOLERANCE);
    //                 let lhs_old = self.constraints[*c_idx as usize].incumbent_lhs;
    //                 let lhs_down =
    //                     lhs_old - (*coeff * self.solution[var]) + (*coeff * var_jump_value);

    //                 self.objective_weight
    //                     * self.vars[var].objective_coeff
    //                     * (var_jump_value - self.solution[var])
    //                     + self.constraints[*c_idx as usize].weight
    //                         * (self.constraints[*c_idx as usize].score(lhs_down)
    //                             - self.constraints[*c_idx as usize].score(lhs_old))
    //             } else {
    //                 0.
    //             }
    //         })
    //         .sum::<f32>();

    //     assert!((self.vars[var].jump_score - jump_score).abs() < EQ_TOLERANCE);
    //     if self.vars[var].jump_score > 0. {
    //         assert!(self.vars[var].good_jump_idx != -1);
    //         assert!(
    //             self.good_choices[self.vars[var].good_jump_idx as usize]
    //                 == Choice {
    //                     var: var as u32,
    //                     dir: Direction::Jump
    //                 }
    //         );
    //     } else {
    //         assert!(self.vars[var].good_jump_idx == -1);
    //         assert!(!self
    //             .good_choices
    //             .iter()
    //             .any(|c| c.var as usize == var && c.dir == Direction::Jump));
    //     }

    //     if self.jump_value(var).is_none() {
    //         assert!((self.solution[var] - self.vars[var].jump_value).abs() < EQ_TOLERANCE);
    //         assert!(self.vars[var].good_jump_idx == -1);
    //     }
    // }

    pub fn init(&mut self, solution: Vec<f32>) {
        self.solution = solution;
        self.incumbent_objective = (0..self.vars.len())
            .map(|v| self.vars[v].objective_coeff as f64 * self.solution[v] as f64)
            .sum();
        self.reset_lhs();

        self.good_choices.clear();
        for var_idx in 0..self.vars.len() {
            self.reset_var_cost(var_idx);
        }
        info!(
            "Inited, {} unsat {} good.",
            self.unsat_constraints.len(),
            self.good_choices.len()
        );
    }

    fn reset_lhs(&mut self) {
        self.unsat_constraints.clear();
        for constraint_idx in 0..self.constraints.len() {
            let constraint = &mut self.constraints[constraint_idx];
            constraint.unsat_idx = -1;
            constraint.incumbent_lhs = 0.;
            for (v, c) in constraint.coeffs.iter() {
                constraint.incumbent_lhs += *c * self.solution[*v as usize];
            }

            if constraint.score(constraint.incumbent_lhs) < -VIOLATION_TOLERANCE {
                constraint.unsat_idx = self.unsat_constraints.len() as i32;
                self.unsat_constraints.push(constraint_idx as u32);
            }
        }
        self.time += self.n_nonzeros;
    }

    fn remove_unsat(&mut self, c_idx: usize) {
        debug_assert!(self.constraints[c_idx].unsat_idx != -1);
        let last_unsat_idx = self.unsat_constraints.len() - 1;
        let last_constraint_idx = self.unsat_constraints[last_unsat_idx] as usize;
        let this_unsat_idx = self.constraints[c_idx].unsat_idx as usize;

        self.unsat_constraints.swap(this_unsat_idx, last_unsat_idx);
        debug_assert!(self.constraints[last_constraint_idx].unsat_idx == last_unsat_idx as i32);

        self.constraints[last_constraint_idx].unsat_idx = this_unsat_idx as i32;
        self.constraints[c_idx].unsat_idx = -1;
        self.unsat_constraints.pop();
    }

    fn decide(&mut self) -> Choice {
        // if self.good_choices.is_empty() {
        //     let mut vars_update = HashSet::new();
        //     for c in self.unsat_constraints.iter() {
        //         for (v, _) in self.constraints[*c as usize].coeffs.iter() {
        //             vars_update.insert(*v);
        //         }
        //     }
        //     for v in vars_update {
        //         self.reset_var_cost(v as usize);
        //     }
        // }

        if !self.good_choices.is_empty() {
            if self.rng.gen::<f32>() < RANDOM_VAR {
                // debug!("Decide random");
                return *self.good_choices.choose(&mut self.rng).unwrap();
            }

            self.time += MAX_GOOD_CHOICES_TO_CHECK;
            // debug!("Decide best");
            return *self
                .good_choices
                .choose_multiple(&mut self.rng, MAX_GOOD_CHOICES_TO_CHECK)
                .max_by_key(|v| {
                    let score = match v.dir {
                        Direction::Up => self.vars[v.var as usize].up_score,
                        Direction::Down => self.vars[v.var as usize].down_score,
                        Direction::Jump => self.vars[v.var as usize].jump_score,
                    };
                    debug_assert!(score > 0.);
                    (
                        OrderedFloat(score),
                        Reverse(self.vars[v.var as usize].time_stamp),
                    )
                })
                .unwrap();
        }

        debug_assert!(self
            .vars
            .iter()
            .all(|v| v.up_score <= 0. && v.down_score <= 0.));

        self.update_weights();

        // if self.good_choices.is_empty() {
        //     // info!("Reduced step size.");
        //     // self.step_size *= 0.5;
        //     for var_idx in 0..self.vars.len() {
        //         self.reset_var_cost(var_idx);
        //     }
        // }

        // assert!(!self.unsat_constraints.is_empty());

        // debug!("Decide best violated constraint");
        if !self.unsat_constraints.is_empty() {
            let c = *self.unsat_constraints.choose(&mut self.rng).unwrap() as usize;

            if self.rng.gen::<f32>() < RANDOM_CELL {
                // debug!("Decide randmo cell");
                let var = self.constraints[c].coeffs.choose(&mut self.rng).unwrap().0;
                let choice = [Direction::Up, Direction::Down, Direction::Jump]
                    .iter()
                    .copied()
                    .map(|dir| Choice { var, dir })
                    .filter(|c| match c.dir {
                        Direction::Up => self.step_up(c.var as usize).is_some(),
                        Direction::Down => self.step_down(c.var as usize).is_some(),
                        Direction::Jump => self.jump_value(c.var as usize).is_some(),
                    })
                    .max_by_key(|v| {
                        let score = match v.dir {
                            Direction::Up => self.vars[v.var as usize].up_score,
                            Direction::Down => self.vars[v.var as usize].down_score,
                            Direction::Jump => self.vars[v.var as usize].jump_score,
                        };
                        (
                            OrderedFloat(score),
                            Reverse(self.vars[v.var as usize].time_stamp),
                        )
                    });
                if let Some(choice) = choice {
                    // println!("Random cell {:?}", choice);
                    return choice;
                }
            }

            if let Some(ch) = self.constraints[c]
                .coeffs
                .iter()
                .flat_map(move |(v, _)| {
                    std::iter::once(Choice {
                        var: *v,
                        dir: Direction::Up,
                    })
                    .chain(std::iter::once(Choice {
                        var: *v,
                        dir: Direction::Down,
                    }))
                    .chain(std::iter::once(Choice {
                        var: *v,
                        dir: Direction::Jump,
                    }))
                })
                .filter(|c| match c.dir {
                    Direction::Up => self.step_up(c.var as usize).is_some(),
                    Direction::Down => self.step_down(c.var as usize).is_some(),
                    Direction::Jump => self.jump_value(c.var as usize).is_some(),
                })
                .max_by_key(|v| {
                    let score = match v.dir {
                        Direction::Up => self.vars[v.var as usize].up_score,
                        Direction::Down => self.vars[v.var as usize].down_score,
                        Direction::Jump => self.vars[v.var as usize].jump_score,
                    };
                    (
                        OrderedFloat(score),
                        Reverse(self.vars[v.var as usize].time_stamp),
                    )
                })
            {
                return ch;
            } else {
                let vs = self.constraints[c]
                    .coeffs
                    .iter()
                    .flat_map(move |(v, _)| {
                        std::iter::once(Choice {
                            var: *v,
                            dir: Direction::Up,
                        })
                        .chain(std::iter::once(Choice {
                            var: *v,
                            dir: Direction::Down,
                        }))
                        .chain(std::iter::once(Choice {
                            var: *v,
                            dir: Direction::Jump,
                        }))
                    })
                    .map(|c| {
                        let value = match c.dir {
                            Direction::Up => self.step_up(c.var as usize),
                            Direction::Down => self.step_down(c.var as usize),
                            Direction::Jump => self.jump_value(c.var as usize),
                        };
                        format!(
                            "v{} lb{} ub{} step{} val:{} nxval:{:?}",
                            c.var,
                            self.vars[c.var as usize].lb,
                            self.vars[c.var as usize].ub,
                            self.vars[c.var as usize].step_size,
                            self.solution[c.var as usize],
                            value
                        )
                    })
                    .collect::<Vec<_>>();
                error!("constraint {:?}", self.constraints[c]);
                for v in vs {
                    error!("  {:?}", v);
                }
                panic!();
            }
        }

        // Completely random?
        for _ in 0..1000 {
            let c = self.constraints.choose(&mut self.rng).unwrap();
            let var = c.coeffs.choose(&mut self.rng).unwrap().0;
            let choice = [Direction::Up, Direction::Down, Direction::Jump]
                .iter()
                .copied()
                .map(|dir| Choice { var, dir })
                .filter(|c| match c.dir {
                    Direction::Up => self.step_up(c.var as usize).is_some(),
                    Direction::Down => self.step_down(c.var as usize).is_some(),
                    Direction::Jump => self.jump_value(c.var as usize).is_some(),
                })
                .max_by_key(|v| {
                    let score = match v.dir {
                        Direction::Up => self.vars[v.var as usize].up_score,
                        Direction::Down => self.vars[v.var as usize].down_score,
                        Direction::Jump => self.vars[v.var as usize].jump_score,
                    };
                    (
                        OrderedFloat(score),
                        Reverse(self.vars[v.var as usize].time_stamp),
                    )
                });
            if let Some(choice) = choice {
                // println!("Completely random cell {:?}", choice);
                return choice;
            }
        }
        panic!("No choices available.");
    }

    fn update_weights(&mut self) {
        // debug!("update weights.");
        let mut reset_all_weights = false;

        self.weight_bumps += 1;
        let any_unsat = !self.unsat_constraints.is_empty();
        let mut unsat_idx = 0;
        let mut dt = 0;
        while unsat_idx < self.unsat_constraints.len() {
            let c_idx = self.unsat_constraints[unsat_idx] as usize;
            unsat_idx += 1;

            let increment = if USE_SIZE_WEIGHTING {
                let num_coeffs = self.constraints[c_idx].coeffs.len();
                let size_factor = if num_coeffs <= 2 {
                    0.3
                } else if num_coeffs <= 6 {
                    1.0
                } else {
                    1.7
                };
                (1.0 / size_factor) * self.weight_update_increment as f32
            } else {
                self.weight_update_increment as f32
            };

            self.constraints[c_idx].weight += increment;
            if self.constraints[c_idx].weight > 1.0e20 {
                reset_all_weights = true;
            }

            let mut cell_idx = 0;
            dt += self.constraints[c_idx].coeffs.len();
            while cell_idx < self.constraints[c_idx].coeffs.len() {
                let constraint = &self.constraints[c_idx];
                let (var_idx, coeff) = constraint.coeffs[cell_idx];
                cell_idx += 1;
                let var_idx = var_idx as usize;
                if let Some(var_up) = self.step_up(var_idx) {
                    let candidate_lhs =
                        constraint.incumbent_lhs + coeff * (var_up - self.solution[var_idx]);
                    self.vars[var_idx].up_score += self.weight_update_increment as f32
                        * (constraint.score(candidate_lhs)
                            - constraint.score(constraint.incumbent_lhs));
                }
                if let Some(var_down) = self.step_down(var_idx) {
                    let candidate_lhs =
                        constraint.incumbent_lhs + coeff * (var_down - self.solution[var_idx]);
                    self.vars[var_idx].down_score += self.weight_update_increment as f32
                        * (constraint.score(candidate_lhs)
                            - constraint.score(constraint.incumbent_lhs));
                }
                if let Some(var_jump_value) = self.jump_value(var_idx) {
                    let candidate_lhs = constraint.incumbent_lhs
                        + coeff * (var_jump_value - self.solution[var_idx]);
                    self.vars[var_idx].jump_score += self.weight_update_increment as f32
                        * (constraint.score(candidate_lhs)
                            - constraint.score(constraint.incumbent_lhs));
                }

                self.update_good_choice_lists(var_idx);
            }
        }

        if !any_unsat && self.incumbent_objective >= self.best_objective {
            // TODO try: average_soft - average_hard < 100

            self.objective_weight += self.weight_update_increment as f32;
            if self.objective_weight > 1.0e20 {
                reset_all_weights = true;
            }

            dt += self.vars.len();
            for var_idx in 0..self.vars.len() {
                if let Some(var_up) = self.step_up(var_idx) {
                    self.vars[var_idx].up_score += self.weight_update_increment as f32
                        * self.vars[var_idx].objective_coeff
                        * (var_up - self.solution[var_idx]);
                }
                if let Some(var_down) = self.step_down(var_idx) {
                    self.vars[var_idx].down_score += self.weight_update_increment as f32
                        * self.vars[var_idx].objective_coeff
                        * (var_down - self.solution[var_idx]);
                }
                if let Some(var_jump_value) = self.jump_value(var_idx) {
                    self.vars[var_idx].jump_score += self.weight_update_increment as f32
                        * self.vars[var_idx].objective_coeff
                        * (var_jump_value - self.solution[var_idx]);
                }
                self.update_good_choice_lists(var_idx);
            }
        }

        self.weight_update_increment *= 1.0 / self.weight_update_decay;

        if reset_all_weights {
            for c in self.constraints.iter_mut() {
                c.weight *= 1.0e-20;
            }
            self.weight_update_increment *= 1.0e-20;
            self.objective_weight *= 1.0e-20;
            for var_idx in 0..self.vars.len() {
                self.reset_var_cost(var_idx);
            }
        }

        self.time += dt;
    }

    fn var_best_jump_value(&mut self, var: usize) -> f32 {
        let _p = hprof::enter("center value");
        self.best_shift_buffer.clear();

        let var_incumbent_value = self.solution[var];
        let is_integer = matches!(self.vars[var].vartype, VarType::Integer);
        let var_bounds = Interval(self.vars[var].lb, self.vars[var].ub);

        let mut current_value = var_bounds.0;
        let mut current_score = 0.0;
        let mut current_slope = 0.0;

        self.time += self.vars[var].coeffs.len();
        for (c_idx, coeff) in self.vars[var].coeffs.iter() {
            let constraint = &self.constraints[*c_idx as usize];
            let constraint_bounds = match constraint.sense {
                RowType::None => continue,
                RowType::Equal => vec![
                    Interval(constraint.rhs, f32::INFINITY),
                    Interval(f32::NEG_INFINITY, constraint.rhs),
                    Interval::singleton(constraint.rhs),
                ],
                RowType::Lte => vec![Interval(f32::NEG_INFINITY, constraint.rhs)],
                RowType::Gte => vec![Interval(constraint.rhs, f32::INFINITY)],
            };
            for constraint_bounds in constraint_bounds {
                let residual_incumbent =
                    Interval::singleton(constraint.incumbent_lhs - *coeff * var_incumbent_value);
                let valid_range = (1.0 / *coeff) * (constraint_bounds + -1.0 * residual_incumbent);
                let valid_range = valid_range.tighten(is_integer);

                // if var == 69 {
                //     debug!("Constraint {:?}", constraint);
                //     debug!("Valid range {:?}", valid_range);
                // }

                if valid_range.is_empty() {
                    continue;
                }

                if valid_range.0 > current_value {
                    current_score += constraint.weight * (valid_range.0 - current_value);
                    current_slope -= constraint.weight;
                    if valid_range.0 < var_bounds.1 {
                        self.best_shift_buffer
                            .push((valid_range.0, constraint.weight));
                    }
                }

                if valid_range.1 <= current_value {
                    current_score += constraint.weight * (valid_range.1 - current_value);
                    current_slope += constraint.weight;
                } else if valid_range.1 < var_bounds.1 {
                    self.best_shift_buffer
                        .push((valid_range.1, constraint.weight));
                }
            }
        }

        self.best_shift_buffer.push((var_bounds.0, 0.0));
        self.best_shift_buffer.push((var_bounds.1, 0.0));
        self.best_shift_buffer
            .sort_by_key(|(v, _)| (OrderedFloat(*v)));

        // if var == 69 {
        //     info!(
        //         " start_score:{} start_value:{} start_slope:{}",
        //         current_score, current_value, current_slope
        //     );
        // }

        let (mut best_score, mut best_value) = (current_score, current_value);
        for (new_value, slope_delta) in self.best_shift_buffer.iter().copied() {
            current_score += (new_value - current_value) * current_slope;
            current_slope += slope_delta;
            current_value = new_value;

            let is_incumbent = (current_value - self.solution[var]).abs() < EQ_TOLERANCE;
            let best_is_incumbent = (best_value - self.solution[var]).abs() < EQ_TOLERANCE;

            if best_is_incumbent || (!is_incumbent && current_score < best_score) {
                best_score = current_score;
                best_value = current_value;
            }
        }

        // if var == 69 {
        //     info!(
        //         "best value for {} {}->{}\n   buffer {:?}",
        //         var, self.solution[var], best_value, self.best_shift_buffer
        //     );
        //     panic!();
        // }

        best_value
    }

    fn reset_var_cost(&mut self, var_idx: usize) {
        self.vars[var_idx].jump_value = self.var_best_jump_value(var_idx);

        let var = &self.vars[var_idx];
        let mut up_score = 0.0;
        let mut down_score = 0.0;
        let mut jump_score = 0.0;

        // debug!(
        //     "var {} value {} up {:?} down {:?}",
        //     var_idx,
        //     self.solution[var_idx],
        //     self.step_up(var_idx),
        //     self.step_down(var_idx)
        // );
        if let Some(var_up) = self.step_up(var_idx) {
            up_score += self.objective_weight
                * self.vars[var_idx].objective_coeff
                * (var_up - self.solution[var_idx]);
        }
        if let Some(var_down) = self.step_down(var_idx) {
            down_score += self.objective_weight
                * self.vars[var_idx].objective_coeff
                * (var_down - self.solution[var_idx]);
        }
        if let Some(var_jump_value) = self.jump_value(var_idx) {
            jump_score += self.objective_weight
                * self.vars[var_idx].objective_coeff
                * (var_jump_value - self.solution[var_idx]);
        }

        self.time += var.coeffs.len();
        for (constraint_idx, coeff) in var.coeffs.iter() {
            let constraint = &self.constraints[*constraint_idx as usize];
            if let Some(var_up) = self.step_up(var_idx) {
                let candidate_lhs =
                    constraint.incumbent_lhs + *coeff * (var_up - self.solution[var_idx]);
                // debug!(
                //     "  cstr:{} rhs:{} lhs:{} newlhs:{} sense:{:?} ",
                //     constraint_idx,
                //     constraint.rhs,
                //     constraint.incumbent_lhs,
                //     candidate_lhs,
                //     constraint.sense
                // );
                // debug!("up_score before:{}", up_score);

                up_score += constraint.weight
                    * (constraint.score(candidate_lhs)
                        - constraint.score(constraint.incumbent_lhs));
                // debug!("up_score after:{}", up_score);
            }
            if let Some(var_down) = self.step_down(var_idx) {
                let candidate_lhs =
                    constraint.incumbent_lhs + *coeff * (var_down - self.solution[var_idx]);
                // debug!(
                //     "  cstr:{} rhs:{} lhs:{} newlhs:{} sense:{:?} ",
                //     constraint_idx,
                //     constraint.rhs,
                //     constraint.incumbent_lhs,
                //     candidate_lhs,
                //     constraint.sense
                // );
                // debug!("downscore before:{}", down_score);

                down_score += constraint.weight
                    * (constraint.score(candidate_lhs)
                        - constraint.score(constraint.incumbent_lhs));
                // debug!("downscore after:{}", down_score);
            }
            if let Some(var_jump_value) = self.jump_value(var_idx) {
                let candidate_lhs =
                    constraint.incumbent_lhs + *coeff * (var_jump_value - self.solution[var_idx]);
                // debug!(
                //     "  cstr:{} rhs:{} lhs:{} newlhs:{} sense:{:?} ",
                //     constraint_idx,
                //     constraint.rhs,
                //     constraint.incumbent_lhs,
                //     candidate_lhs,
                //     constraint.sense
                // );
                // debug!("downscore before:{}", down_score);

                jump_score += constraint.weight
                    * (constraint.score(candidate_lhs)
                        - constraint.score(constraint.incumbent_lhs));
                // debug!("downscore after:{}", down_score);
            }
        }

        let var = &mut self.vars[var_idx];
        var.up_score = up_score;
        var.down_score = down_score;
        var.jump_score = jump_score;

        // debug!(
        //     "  updated score var:{} up:{} down:{} jump:{} jumpvalue:{}",
        //     var_idx, up_score, &down_score,jump_score, jump_value
        // );

        self.update_good_choice_lists(var_idx);
    }

    fn update_good_choice_lists(&mut self, var_idx: usize) {
        // debug!("* good choice {:?}", self.good_choices);
        let var = &mut self.vars[var_idx];

        if var.up_score > 0. && var.good_up_idx == -1 {
            // debug!("  adding {} up", var_idx);
            var.good_up_idx = self.good_choices.len() as _;
            self.good_choices.push(Choice {
                var: var_idx as u32,
                dir: Direction::Up,
            });
        }
        if var.up_score <= 0. && var.good_up_idx != -1 {
            // debug!("  removing {} up", var_idx);
            let idx = var.good_up_idx;
            self.pop_good_var_idx(idx);
            self.vars[var_idx].good_up_idx = -1;
        }

        let var = &mut self.vars[var_idx];

        if var.down_score > 0. && var.good_down_idx == -1 {
            // debug!("  adding {} down", var_idx);
            var.good_down_idx = self.good_choices.len() as _;
            self.good_choices.push(Choice {
                var: var_idx as u32,
                dir: Direction::Down,
            });
        }
        if var.down_score <= 0. && var.good_down_idx != -1 {
            // debug!("  removing {} down", var_idx);
            let idx = var.good_down_idx;
            self.pop_good_var_idx(idx);
            self.vars[var_idx].good_down_idx = -1;
        }
        // debug!("  --> good choice {:?}", self.good_choices);

        let var = &mut self.vars[var_idx];

        if var.jump_score > 0. && var.good_jump_idx == -1 {
            // debug!("  adding {} down", var_idx);
            var.good_jump_idx = self.good_choices.len() as _;
            self.good_choices.push(Choice {
                var: var_idx as u32,
                dir: Direction::Jump,
            });
        }
        if var.jump_score <= 0. && var.good_jump_idx != -1 {
            // debug!("  removing {} down", var_idx);
            let idx = var.good_jump_idx;
            self.pop_good_var_idx(idx);
            self.vars[var_idx].good_jump_idx = -1;
        }
    }

    fn pop_good_var_idx(&mut self, idx: i32) {
        let last_idx = self.good_choices.len() - 1;
        let last_choice = self.good_choices[last_idx];
        self.good_choices.swap(last_idx, idx as usize);

        match last_choice.dir {
            Direction::Up => {
                debug_assert!(self.vars[last_choice.var as usize].good_up_idx == last_idx as i32);
                self.vars[last_choice.var as usize].good_up_idx = idx;
            }
            Direction::Down => {
                debug_assert!(self.vars[last_choice.var as usize].good_down_idx == last_idx as i32);
                self.vars[last_choice.var as usize].good_down_idx = idx;
            }
            Direction::Jump => {
                debug_assert!(self.vars[last_choice.var as usize].good_jump_idx == last_idx as i32);
                self.vars[last_choice.var as usize].good_jump_idx = idx;
            }
        }
        self.good_choices.pop();
    }

    fn set_value(&mut self, choice: Choice) {
        // self.check_all();
        let var_idx = choice.var as usize;
        // self.reset_var_cost(var_idx);

        let choice = if !USE_RECOMPUTE_JUMP {
            choice
        } else {
            self.reset_var_cost(var_idx);

            [Direction::Up, Direction::Down, Direction::Jump]
                .iter()
                .copied()
                .map(|dir| Choice { dir, ..choice })
                .filter(|c| match c.dir {
                    Direction::Up => self.step_up(c.var as usize).is_some(),
                    Direction::Down => self.step_down(c.var as usize).is_some(),
                    Direction::Jump => self.jump_value(c.var as usize).is_some(),
                })
                .max_by_key(|v| {
                    let score = match v.dir {
                        Direction::Up => self.vars[v.var as usize].up_score,
                        Direction::Down => self.vars[v.var as usize].down_score,
                        Direction::Jump => self.vars[v.var as usize].jump_score,
                    };
                    (
                        OrderedFloat(score),
                        Reverse(self.vars[v.var as usize].time_stamp),
                    )
                })
                .unwrap()
        };

        let dir = [Direction::Up, Direction::Down, Direction::Jump]
            .iter()
            .filter(|n| match n {
                Direction::Up => self.step_up(var_idx).is_some(),
                Direction::Down => self.step_down(var_idx).is_some(),
                Direction::Jump => self.jump_value(var_idx).is_some(),
            })
            .max_by_key(|n| match n {
                Direction::Up => OrderedFloat(self.vars[var_idx].up_score),
                Direction::Down => OrderedFloat(self.vars[var_idx].down_score),
                Direction::Jump => OrderedFloat(self.vars[var_idx].jump_score),
            })
            .unwrap();

        let new_value = match dir {
            Direction::Up => self.step_up(var_idx).unwrap(),
            Direction::Down => self.step_down(var_idx).unwrap(),
            Direction::Jump => self.jump_value(var_idx).unwrap(),
        };

        // let old_score = match dir {
        //     Direction::Up => self.vars[var_idx].up_score,
        //     Direction::Down => self.vars[var_idx].down_score,
        //     Direction::Jump => self.vars[var_idx].jump_score,
        // };
        // assert!(old_score > 0.);

        let old_value = self.solution[var_idx];
        self.solution[var_idx] = new_value;
        let delta_value = new_value as f64 - old_value as f64;

        self.incumbent_objective +=
            self.vars[choice.var as usize].objective_coeff as f64 * delta_value;
        // if self.incumbent_objective < 0. {
        //     println!(
        //         "incumbent bug: setting var {} from {} to {} objcoeff {} delta {} delta_obj {}",
        //         var_idx,
        //         old_value,
        //         new_value,
        //         self.vars[choice.var as usize].objective_coeff,
        //         delta_value,
        //         self.vars[choice.var as usize].objective_coeff as f64 * delta_value
        //     );
        // }

        if let Some((back_and_forth_n, back_and_forth_value)) =
            self.vars[choice.var as usize].back_and_forth.as_mut()
        {
            if (new_value - *back_and_forth_value).abs() < EQ_TOLERANCE {
                *back_and_forth_n += 1;
                *back_and_forth_value = old_value;
            }

            if *back_and_forth_n >= 2 {
                self.vars[var_idx].step_size *= 0.5;
                // if var_idx == 1220 {
                //     warn!(
                //         "Setting v{} to step {}",
                //         var_idx, self.vars[var_idx].step_size
                //     );
                // }

                // if !(self.vars[var_idx].step_size > EQ_TOLERANCE) {
                //     let v = &self.vars[var_idx];
                //     error!(
                //         "var {} lb {} ub {} step {}",
                //         var_idx, v.lb, v.ub, v.step_size
                //     );
                //     panic!();
                // }
                if matches!(self.vars[var_idx].vartype, VarType::Integer) {
                    self.vars[var_idx].step_size = self.vars[var_idx].step_size.ceil();
                }
                // debug!(
                //     "Decreasing step size of var:{} backnf:{:?} lb:{} ub:{} new_step:{}",
                //     var_idx,
                //     self.vars[var_idx as usize].back_and_forth,
                //     self.vars[var_idx as usize].lb,
                //     self.vars[var_idx as usize].ub,
                //     self.vars[var_idx as usize].step_size,
                // );
                self.vars[var_idx].back_and_forth = None;
            }
        } else if (matches!(self.vars[choice.var as usize].vartype, VarType::Continuous)
            && self.vars[choice.var as usize].step_size > 2.0 * EQ_TOLERANCE)
            || self.vars[choice.var as usize].step_size > 1.0
        {
            self.vars[choice.var as usize].back_and_forth = Some((0, old_value));
        }

        // debug!(
        //     "Setting {:?} from {} to {}, delta:{}  (SCORE:{})",
        //     choice, old_value, new_value, delta_value, old_score,
        // );

        // now the incumbent lhs of all relevant constraints has changed

        let mut var_cstr_idx = 0;
        // for (c_idx, coeff) in self.vars[choice.var as usize].coeffs.iter().copied() {
        let mut dt = 0;
        while var_cstr_idx < self.vars[var_idx].coeffs.len() {
            let (c_idx, coeff) = self.vars[var_idx].coeffs[var_cstr_idx];
            var_cstr_idx += 1;

            let old_lhs = self.constraints[c_idx as usize].incumbent_lhs;
            let new_lhs = old_lhs + coeff * delta_value as f32;

            self.constraints[c_idx as usize].incumbent_lhs = new_lhs;
            let new_cost = self.constraints[c_idx as usize].score(new_lhs);

            // debug!(
            //     "Changing incumbnet lhs {} to {}  score {} to {}",
            //     old_lhs,
            //     new_lhs,
            //     self.constraints[c_idx as usize].score(old_lhs),
            //     new_cost
            // );

            if new_cost < -VIOLATION_TOLERANCE && self.constraints[c_idx as usize].unsat_idx == -1 {
                // Became unsat.
                self.constraints[c_idx as usize].unsat_idx = self.unsat_constraints.len() as i32;
                self.unsat_constraints.push(c_idx);
            }
            if new_cost >= -VIOLATION_TOLERANCE && self.constraints[c_idx as usize].unsat_idx != -1
            {
                // Became sat.
                debug_assert!(new_cost.abs() < VIOLATION_TOLERANCE);
                self.remove_unsat(c_idx as usize);
            }

            // debug!("Updating scores for constraint {}", c_idx);

            // self.check_constraint(c_idx as usize);

            // Now update scores for all variables in the constraint.
            let mut cstr_var_idx = 0;
            dt += self.constraints[c_idx as usize].coeffs.len();
            while cstr_var_idx < self.constraints[c_idx as usize].coeffs.len() {
                let (other_var, coeff) = self.constraints[c_idx as usize].coeffs[cstr_var_idx];
                cstr_var_idx += 1;

                if other_var as usize == var_idx {
                    continue;
                }

                if let Some(var_up) = self.step_up(other_var as usize) {
                    let constraint = &self.constraints[c_idx as usize];

                    let old_modified_lhs =
                        old_lhs + coeff * (var_up - self.solution[other_var as usize]);

                    let old_score_term = constraint.weight
                        * (constraint.score(old_modified_lhs) - constraint.score(old_lhs));

                    let new_modified_lhs =
                        new_lhs + coeff * (var_up - self.solution[other_var as usize]);

                    let new_score_term = constraint.weight
                        * (constraint.score(new_modified_lhs) - constraint.score(new_lhs));

                    self.vars[other_var as usize].up_score += new_score_term - old_score_term;
                }

                if let Some(var_down) = self.step_down(other_var as usize) {
                    let constraint = &self.constraints[c_idx as usize];

                    let old_modified_lhs =
                        old_lhs + coeff * (var_down - self.solution[other_var as usize]);

                    let old_score_term = constraint.weight
                        * (constraint.score(old_modified_lhs) - constraint.score(old_lhs));

                    let new_modified_lhs =
                        new_lhs + coeff * (var_down - self.solution[other_var as usize]);

                    let new_score_term = constraint.weight
                        * (constraint.score(new_modified_lhs) - constraint.score(new_lhs));

                    self.vars[other_var as usize].down_score += new_score_term - old_score_term;
                }

                if let Some(var_jump_value) = self.jump_value(other_var as usize) {
                    let constraint = &self.constraints[c_idx as usize];

                    let old_modified_lhs =
                        old_lhs + coeff * (var_jump_value - self.solution[other_var as usize]);

                    let old_score_term = constraint.weight
                        * (constraint.score(old_modified_lhs) - constraint.score(old_lhs));

                    let new_modified_lhs =
                        new_lhs + coeff * (var_jump_value - self.solution[other_var as usize]);

                    let new_score_term = constraint.weight
                        * (constraint.score(new_modified_lhs) - constraint.score(new_lhs));

                    self.vars[other_var as usize].jump_score += new_score_term - old_score_term;
                }

                self.update_good_choice_lists(other_var as usize);
            }
        }

        self.reset_var_cost(var_idx);

        // let inverse_score = if choice.dir == 1 {
        //     self.vars[var_idx].down_score
        // } else {
        //     self.vars[var_idx].up_score
        // };

        // if (inverse_score + old_score).abs() > 1e-6 {
        //     println!(
        //         "choice {:?} old score {} inverse {}",
        //         choice, old_score, inverse_score
        //     );
        //     panic!();
        // }
        self.time += dt;
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::new(0, 1.0)
    }
}
