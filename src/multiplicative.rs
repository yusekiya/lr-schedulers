use crate::Scheduler;

/// Multiplies the learning rate by a factor given by a lambda function at each epoch.
///
/// # Examples
///
/// This scheduler generates multiplicatively decaying learning rates:
///
/// ```
/// # use lr_schedulers::multiplicative::MultiplicativeLR;
/// # use lr_schedulers::Scheduler;
/// # use approx::assert_relative_eq;
/// let mut scheduler = MultiplicativeLR::new(1.0, |_| 0.95, 0);
/// let expected = [1.0, 0.95, 0.9025, 0.857375, 0.81450625];
/// for exp_lr in expected.iter() {
///     assert_relative_eq!(scheduler.get_lr(), exp_lr, epsilon = 1e-10);
///     scheduler.step(0.01);
/// }
/// ```
///
/// Custom lambda function with different decay rates:
///
/// ```
/// # use lr_schedulers::multiplicative::MultiplicativeLR;
/// # use lr_schedulers::Scheduler;
/// # use approx::assert_relative_eq;
/// let lambda = |epoch| if epoch < 3 { 0.9 } else { 0.95 };
/// let mut scheduler = MultiplicativeLR::new(1.0, lambda, 0);
/// let expected = [1.0, 0.9, 0.81, 0.729, 0.69255, 0.6579225];
/// for exp_lr in expected.iter() {
///     assert_relative_eq!(scheduler.get_lr(), exp_lr, epsilon = 1e-10);
///     scheduler.step(0.01);
/// }
/// ```
///
/// Starting point can be changed with `init_step`:
///
/// ```
/// # use lr_schedulers::multiplicative::MultiplicativeLR;
/// # use lr_schedulers::Scheduler;
/// # use approx::assert_relative_eq;
/// let init_step = 2;
/// let mut scheduler = MultiplicativeLR::new(1.0, |_| 0.95, init_step);
/// let expected = [0.9025, 0.857375, 0.81450625];
/// for exp_lr in expected.iter() {
///     assert_relative_eq!(scheduler.get_lr(), exp_lr, epsilon = 1e-10);
///     scheduler.step(0.01);
/// }
/// ```
///
/// The `get_lr` method returns the same value unless the `step` method is invoked.
///
/// ```no_run
/// # use lr_schedulers::multiplicative::MultiplicativeLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = MultiplicativeLR::new(1.0, |_| 0.95, 0);
/// let lr = scheduler.get_lr();
/// assert_eq!(lr, scheduler.get_lr());
/// scheduler.step(0.01);
/// let new_lr = scheduler.get_lr();
/// assert_ne!(lr, new_lr);
/// ```
#[derive(Debug, Clone)]
pub struct MultiplicativeLR<F>
where
    F: Fn(usize) -> f64,
{
    lr: f64,
    step: usize,
    lr_lambda: F,
}

impl<F> MultiplicativeLR<F>
where
    F: Fn(usize) -> f64,
{
    /// Constructs a MultiplicativeLR instance.
    ///
    /// This scheduler multiplies the learning rate by the factor returned by `lr_lambda` function
    /// at each step. The multiplication is applied to the previous learning rate, not the base learning rate.
    ///
    /// Starting step can be specified by `init_step`. Use `init_step=0` to train a model from the beginning.
    pub fn new(base_lr: f64, lr_lambda: F, init_step: usize) -> Self {
        let mut lr = base_lr;
        // Apply lambda function for each step up to init_step
        for i in 0..init_step {
            lr *= lr_lambda(i);
        }
        MultiplicativeLR {
            lr,
            step: init_step,
            lr_lambda,
        }
    }
}

impl<F> Scheduler for MultiplicativeLR<F>
where
    F: Fn(usize) -> f64,
{
    fn step(&mut self, _loss: f64) {
        self.lr *= (self.lr_lambda)(self.step);
        self.step += 1;
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scheduler;
    use approx::assert_relative_eq;

    #[test]
    fn constant_decay() {
        let base_lr = 1.0;
        let lambda = |_| 0.95;
        let init_step = 0;
        let mut scheduler = MultiplicativeLR::new(base_lr, lambda, init_step);

        let expected_lrs = [1.0, 0.95, 0.9025, 0.857375, 0.81450625];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-9, max_relative = 1e-9);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn variable_decay() {
        let base_lr = 1.0;
        let lambda = |epoch| if epoch < 3 { 0.9 } else { 0.95 };
        let init_step = 0;
        let mut scheduler = MultiplicativeLR::new(base_lr, lambda, init_step);

        let expected_lrs = [1.0, 0.9, 0.81, 0.729, 0.69255, 0.6579225];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-9, max_relative = 1e-9);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn no_decay() {
        let base_lr = 2.0;
        let lambda = |_| 1.0;
        let init_step = 0;
        let mut scheduler = MultiplicativeLR::new(base_lr, lambda, init_step);

        for _ in 0..5 {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, base_lr, epsilon = 1e-9, max_relative = 1e-9);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn increasing_lr() {
        let base_lr = 0.1;
        let lambda = |_| 1.1;
        let init_step = 0;
        let mut scheduler = MultiplicativeLR::new(base_lr, lambda, init_step);

        let expected_lrs = [0.1, 0.11, 0.121, 0.1331, 0.14641];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-9, max_relative = 1e-9);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_midway() {
        let base_lr = 1.0;
        let lambda = |_| 0.95;
        let init_step = 2;
        let mut scheduler = MultiplicativeLR::new(base_lr, lambda, init_step);

        // At init_step=2, lr should be 1.0 * 0.95 * 0.95 = 0.9025
        let expected_lrs = [0.9025, 0.857375, 0.81450625];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-9, max_relative = 1e-9);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn complex_lambda() {
        let base_lr = 1.0;
        let lambda = |epoch| {
            if epoch % 3 == 0 {
                0.5 // Large decay every 3 epochs
            } else {
                0.9 // Small decay otherwise
            }
        };
        let init_step = 0;
        let mut scheduler = MultiplicativeLR::new(base_lr, lambda, init_step);

        // epoch 0: 1.0 * 0.5 = 0.5
        // epoch 1: 0.5 * 0.9 = 0.45
        // epoch 2: 0.45 * 0.9 = 0.405
        // epoch 3: 0.405 * 0.5 = 0.2025
        // epoch 4: 0.2025 * 0.9 = 0.18225
        let expected_lrs = [1.0, 0.5, 0.45, 0.405, 0.2025, 0.18225];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-9, max_relative = 1e-9);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_with_variable_decay() {
        let base_lr = 1.0;
        let lambda = |epoch| if epoch < 2 { 0.8 } else { 0.9 };
        let init_step = 3;
        let mut scheduler = MultiplicativeLR::new(base_lr, lambda, init_step);

        // Apply lambda for epochs 0, 1, 2
        // epoch 0: 1.0 * 0.8 = 0.8
        // epoch 1: 0.8 * 0.8 = 0.64
        // epoch 2: 0.64 * 0.9 = 0.576
        // So at init_step=3, lr = 0.576
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.576, epsilon = 1e-9, max_relative = 1e-9);

        // Next steps should use lambda(3) = 0.9, lambda(4) = 0.9, ...
        scheduler.step(0.0);
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.5184, epsilon = 1e-9, max_relative = 1e-9);
    }
}
