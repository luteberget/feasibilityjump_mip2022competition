mod interval;

#[cfg(feature = "gurobi")]
pub mod lp;

pub mod solver;

pub const UNBOUNDED: f64 = 1.0e5;

pub const USE_EXPONENTIAL_WEIGHT_DECAY: bool = true;
pub const USE_DAMPING: bool = false;
pub const USE_SIZE_WEIGHTING: bool = false;
