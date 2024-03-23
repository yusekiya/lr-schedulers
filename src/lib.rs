pub mod constant;
pub mod linear;
pub mod exponential;
pub mod cosine_annealing;

pub trait Scheduler {
    /// Proceed the step of scheduler.
    fn step(&mut self, loss: f64);
    /// Get a learning rate for the current step.
    fn get_lr(&self) -> f64;
}