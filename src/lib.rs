pub mod constant;
pub mod linear;
pub mod exponential;
pub mod cosine_annealing;
pub mod cosine_annealing_warm_restarts;

pub trait Scheduler {
    /// Proceeds the step of scheduler.
    fn step(&mut self, loss: f64);
    /// Returns a learning rate for the current step.
    fn get_lr(&self, loss: f64) -> f64; // The argument `loss` is for schedulers such as ReduceLROnPlateau.
}