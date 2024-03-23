pub mod constantlr;

pub trait Scheduler {
    /// Proceed the step of scheduler.
    fn step(&mut self, loss: f64);
    /// Get a learning rate for the current step.
    fn get_lr(&self) -> f64;
}