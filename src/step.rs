use crate::Scheduler;

/// Decays the learning rate by gamma every step_size epochs.
/// 
/// # Examples
/// 
/// This scheduler generates step-wise decaying learning rates:
/// 
/// ```
/// # use lr_schedulers::step::StepLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = StepLR::new(1.0, 3, 0.1, 0);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 10 {
///     // Note: loss value is not used in this scheduler.
///     learning_rates.push(scheduler.get_lr(0.01));
///     scheduler.step(0.01);
/// }
/// assert_eq!(learning_rates, [1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001]);
/// ```
/// 
/// Starting point can be changed with `init_step`:
/// 
/// ```
/// # use lr_schedulers::step::StepLR;
/// # use lr_schedulers::Scheduler;
/// let init_step = 2;
/// let mut scheduler = StepLR::new(1.0, 3, 0.1, init_step);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 5 {
///     // Note: loss value is not used in this scheduler.
///     learning_rates.push(scheduler.get_lr(0.01));
///     scheduler.step(0.01);
/// }
/// assert_eq!(learning_rates, [1.0, 0.1, 0.1, 0.1, 0.01]);
/// ```
/// 
/// The `get_lr` method returns the same value unless the `step` method is invoked.
/// 
/// ```no_run
/// # use lr_schedulers::step::StepLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = StepLR::new(1.0, 3, 0.1, 0);
/// // Note: loss value is not used in this scheduler.
/// let lr = scheduler.get_lr(0.01);
/// assert_eq!(lr, scheduler.get_lr(0.01));
/// scheduler.step(0.01);
/// let lr = scheduler.get_lr(0.01);
/// assert_ne!(lr, scheduler.get_lr(0.01));
/// ```
#[derive(Debug, Clone)]
pub struct StepLR {
    lr: f64,
    base_lr: f64,
    step: usize,
    step_size: usize,
    gamma: f64,
}

impl StepLR {
    /// Constructs a StepLR instance.
    /// 
    /// This scheduler returns learning rate that is decayed by `gamma` every `step_size` epochs.
    /// For epoch n, the learning rate is calculated as:
    /// lr = base_lr * gamma^(floor(n / step_size))
    /// 
    /// Starting step can be specified by `init_step`. Use `init_step=0` to train a model from the beginning.
    pub fn new(base_lr: f64, step_size: usize, gamma: f64, init_step: usize) -> Self {
        let lr = base_lr * gamma.powi((init_step / step_size) as i32);
        StepLR {
            lr,
            base_lr,
            step: init_step,
            step_size,
            gamma,
        }
    }
}

impl Scheduler for StepLR {
    fn step(&mut self, _loss: f64) {
        self.step += 1;
        self.lr = self.base_lr * self.gamma.powi((self.step / self.step_size) as i32);
    }

    fn get_lr(&self, _loss: f64) -> f64 {
        self.lr
    }
}

#[cfg(test)]
mod tests {
    use crate::Scheduler;
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn basic_decay() {
        let base_lr = 1.0;
        let step_size = 3;
        let gamma = 0.1;
        let init_step = 0;
        let mut scheduler = StepLR::new(base_lr, step_size, gamma, init_step);
        
        let expected_lrs = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001];
        for (_i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr(0.0);
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_midway() {
        let base_lr = 1.0;
        let step_size = 3;
        let gamma = 0.1;
        let init_step = 2;
        let mut scheduler = StepLR::new(base_lr, step_size, gamma, init_step);
        
        let expected_lrs = [1.0, 0.1, 0.1, 0.1, 0.01];
        for (_i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr(0.0);
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_at_boundary() {
        let base_lr = 1.0;
        let step_size = 3;
        let gamma = 0.1;
        let init_step = 3;
        let mut scheduler = StepLR::new(base_lr, step_size, gamma, init_step);
        
        let expected_lrs = [0.1, 0.1, 0.1, 0.01, 0.01];
        for (_i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr(0.0);
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn step_size_one() {
        let base_lr = 1.0;
        let step_size = 1;
        let gamma = 0.5;
        let init_step = 0;
        let mut scheduler = StepLR::new(base_lr, step_size, gamma, init_step);
        
        let expected_lrs = [1.0, 0.5, 0.25, 0.125, 0.0625];
        for (_i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr(0.0);
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn large_step_size() {
        let base_lr = 1.0;
        let step_size = 100;
        let gamma = 0.1;
        let init_step = 0;
        let mut scheduler = StepLR::new(base_lr, step_size, gamma, init_step);
        
        // Learning rate should remain constant for the first 100 steps
        for _ in 0 .. 100 {
            let lr = scheduler.get_lr(0.0);
            assert_relative_eq!(lr, base_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
        
        // After 100 steps, learning rate should be decayed
        let lr = scheduler.get_lr(0.0);
        assert_relative_eq!(lr, base_lr * gamma, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn gamma_greater_than_one() {
        let base_lr = 0.1;
        let step_size = 2;
        let gamma = 2.0;
        let init_step = 0;
        let mut scheduler = StepLR::new(base_lr, step_size, gamma, init_step);
        
        let expected_lrs = [0.1, 0.1, 0.2, 0.2, 0.4];
        for (_i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr(0.0);
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn init_step_multiple_decays() {
        let base_lr = 1.0;
        let step_size = 2;
        let gamma = 0.1;
        let init_step = 7;
        let mut scheduler = StepLR::new(base_lr, step_size, gamma, init_step);
        
        // At init_step=7, we should have decayed floor(7/2)=3 times
        // So lr = 1.0 * 0.1^3 = 0.001
        let lr = scheduler.get_lr(0.0);
        assert_relative_eq!(lr, 0.001, epsilon = 1e-10, max_relative = 1e-10);
        
        // Step 8 (still in the same decay period)
        scheduler.step(0.0);
        let lr = scheduler.get_lr(0.0);
        assert_relative_eq!(lr, 0.0001, epsilon = 1e-10, max_relative = 1e-10);
    }
}