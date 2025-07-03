pub mod constant;
pub mod cosine_annealing;
pub mod cosine_annealing_warm_restarts;
pub mod exponential;
pub mod linear;
pub mod multiplicative;
pub mod multistep;
pub mod polynomial;
pub mod reduce_lr_on_plateau;
pub mod step;

pub trait Scheduler {
    /// Proceeds the step of scheduler.
    fn step(&mut self, loss: f64);
    /// Returns a learning rate for the current step.
    ///
    /// The argument `loss` is for schedulers that require history of loss value such as ReduceLROnPlateau.
    fn get_lr(&self, loss: f64) -> f64;
}
