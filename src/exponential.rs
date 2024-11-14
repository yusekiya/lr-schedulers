use crate::Scheduler;

/// Change the learning rate geometrically.
/// 
/// # Examples
/// 
/// This scheduler generates geometrically changing learning rates:
/// 
/// ```
/// # use lr_schedulers::exponential::ExponentialLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = ExponentialLR::new(2.0, 0.5, 0);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 5 {
///     // Note: loss value is not used in this scheduler.
///     learning_rates.push(scheduler.get_lr(0.01));
///     scheduler.step(0.01);
/// }
/// assert_eq!(learning_rates, [2.0, 1.0, 0.5, 0.25, 0.125]);
/// ```
/// 
/// Starting point can be changed with `init_step`:
/// 
/// ```
/// # use lr_schedulers::exponential::ExponentialLR;
/// # use lr_schedulers::Scheduler;
/// let init_step = 1;
/// let mut scheduler = ExponentialLR::new(2.0, 0.5, init_step);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 5 {
///     // Note: loss value is not used in this scheduler.
///     learning_rates.push(scheduler.get_lr(0.01));
///     scheduler.step(0.01);
/// }
/// assert_eq!(learning_rates, [1.0, 0.5, 0.25, 0.125, 0.0625]);
/// ```
/// 
/// The `get_lr` method returns the same value unless the `step` method is invoked.
/// 
/// ```no_run
/// # use lr_schedulers::exponential::ExponentialLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = ExponentialLR::new(2.0, 0.5, 0);
/// // Note: loss value is not used in this scheduler.
/// let lr = scheduler.get_lr(0.01);
/// assert_eq!(lr, scheduler.get_lr(0.01));
/// scheduler.step(0.01);
/// let lr = scheduler.get_lr(0.01);
/// assert_ne!(lr, scheduler.get_lr(0.01));
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    lr: f64,
    gamma: f64,
}

impl ExponentialLR {
    /// Construct a ExponentialLR instance.
    /// 
    /// This scheduler returns learning rate at a step i as
    /// lr_i = `gamma` * lr_{i-1}.
    /// 
    /// Starting step can be specified by `init_step`. Use `init_step=0` to train a model from the beginning.
    pub fn new(base_lr: f64, gamma: f64, init_step: usize) -> Self {
        let lr = base_lr * gamma.powi(init_step as i32);
        ExponentialLR { lr, gamma }
    }
}

impl Scheduler for ExponentialLR {
    fn step(&mut self, _loss: f64) {
        self.lr *= self.gamma;
    }

    fn get_lr(&self, _loss: f64) -> f64 {
        self.lr
    }
}

#[cfg(test)]
mod tests {
    use crate::Scheduler;
    use super::*;

    #[test]
    fn decrease_lr() {
        let base_lr = 2.0;
        let gamma = 0.5;
        let init_step = 0;
        let mut scheduler = ExponentialLR::new(
            base_lr, gamma, init_step
        );
        let expected_lrs = [2.0, 1.0, 0.5, 0.25, 0.125];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr(0.0);
            assert_eq!(lr, *exp_lr, "Step {}", i);
            // Process a step with dummy loss
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_midway() {
        let base_lr = 2.0;
        let gamma = 0.5;
        let init_step = 2;
        let mut scheduler = ExponentialLR::new(
            base_lr, gamma, init_step
        );
        let expected_lrs = [0.5, 0.25, 0.125, 0.0625];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr(0.0);
            assert_eq!(lr, *exp_lr, "Step {}", i);
            // Process a step with dummy loss
            scheduler.step(0.0);
        }
    }
}