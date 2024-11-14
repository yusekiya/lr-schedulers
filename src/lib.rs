pub mod constant;
pub mod linear;
pub mod exponential;
pub mod cosine_annealing;
pub mod cosine_annealing_warm_restarts;

pub trait Scheduler {
    /// Proceed the step of scheduler.
    fn step(&mut self, loss: f64);
    /// Get a learning rate for the current step.
    fn get_lr(&self, loss: f64) -> f64;
}