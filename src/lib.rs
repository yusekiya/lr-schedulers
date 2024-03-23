pub mod constantlr;
pub mod linearlr;
pub mod exponentiallr;
pub mod cosineannealinglr;

pub trait Scheduler {
    /// Proceed the step of scheduler.
    fn step(&mut self, loss: f64);
    /// Get a learning rate for the current step.
    fn get_lr(&self) -> f64;
}