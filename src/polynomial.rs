use crate::Scheduler;

/// Decays the learning rate using a polynomial function.
///
/// # Examples
///
/// This scheduler generates polynomial learning rate decay:
///
/// ```
/// # use lr_schedulers::polynomial::PolynomialLR;
/// # use lr_schedulers::Scheduler;
/// # use approx::assert_relative_eq;
/// let mut scheduler = PolynomialLR::new(1.0, 5, 1.0, 0);
/// let expected = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0];
/// for exp_lr in expected.iter() {
///     assert_relative_eq!(scheduler.get_lr(), exp_lr, epsilon = 1e-10);
///     scheduler.step(0.01);
/// }
/// ```
///
/// Different power values create different decay curves:
///
/// ```
/// # use lr_schedulers::polynomial::PolynomialLR;
/// # use lr_schedulers::Scheduler;
/// # use approx::assert_relative_eq;
/// // Quadratic decay (power=2.0)
/// let mut scheduler = PolynomialLR::new(1.0, 4, 2.0, 0);
/// let expected = [1.0, 0.5625, 0.25, 0.0625, 0.0, 0.0];
/// for exp_lr in expected.iter() {
///     assert_relative_eq!(scheduler.get_lr(), exp_lr, epsilon = 1e-10);
///     scheduler.step(0.01);
/// }
/// ```
///
/// Starting point can be changed with `init_step`:
///
/// ```
/// # use lr_schedulers::polynomial::PolynomialLR;
/// # use lr_schedulers::Scheduler;
/// # use approx::assert_relative_eq;
/// let init_step = 2;
/// let mut scheduler = PolynomialLR::new(1.0, 5, 1.0, init_step);
/// let expected = [0.6, 0.4, 0.2, 0.0];
/// for exp_lr in expected.iter() {
///     assert_relative_eq!(scheduler.get_lr(), exp_lr, epsilon = 1e-10);
///     scheduler.step(0.01);
/// }
/// ```
///
/// The `get_lr` method returns the same value unless the `step` method is invoked.
///
/// ```no_run
/// # use lr_schedulers::polynomial::PolynomialLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = PolynomialLR::new(1.0, 5, 1.0, 0);
/// let lr = scheduler.get_lr();
/// assert_eq!(lr, scheduler.get_lr());
/// scheduler.step(0.01);
/// let new_lr = scheduler.get_lr();
/// assert_ne!(lr, new_lr);
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialLR {
    lr: f64,
    base_lr: f64,
    step: usize,
    total_iters: usize,
    power: f64,
}

impl PolynomialLR {
    /// Constructs a PolynomialLR instance.
    ///
    /// This scheduler decays the learning rate using a polynomial function:
    /// lr = base_lr * ((1 - step / total_iters) ^ power)
    ///
    /// After `total_iters` steps, the learning rate remains at 0.
    ///
    /// Starting step can be specified by `init_step`. Use `init_step=0` to train a model from the beginning.
    pub fn new(base_lr: f64, total_iters: usize, power: f64, init_step: usize) -> Self {
        let lr = if init_step >= total_iters {
            0.0
        } else {
            let factor = 1.0 - (init_step as f64) / (total_iters as f64);
            base_lr * factor.powf(power)
        };

        PolynomialLR {
            lr,
            base_lr,
            step: init_step,
            total_iters,
            power,
        }
    }
}

impl Scheduler for PolynomialLR {
    fn step(&mut self, _loss: f64) {
        self.step += 1;

        if self.step >= self.total_iters {
            self.lr = 0.0;
        } else {
            let factor = 1.0 - (self.step as f64) / (self.total_iters as f64);
            self.lr = self.base_lr * factor.powf(self.power);
        }
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
    fn linear_decay() {
        let base_lr = 1.0;
        let total_iters = 5;
        let power = 1.0;
        let init_step = 0;
        let mut scheduler = PolynomialLR::new(base_lr, total_iters, power, init_step);

        let expected_lrs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn quadratic_decay() {
        let base_lr = 1.0;
        let total_iters = 4;
        let power = 2.0;
        let init_step = 0;
        let mut scheduler = PolynomialLR::new(base_lr, total_iters, power, init_step);

        let expected_lrs = [1.0, 0.5625, 0.25, 0.0625, 0.0, 0.0];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn square_root_decay() {
        let base_lr = 1.0;
        let total_iters = 4;
        let power = 0.5;
        let init_step = 0;
        let mut scheduler = PolynomialLR::new(base_lr, total_iters, power, init_step);

        // Expected values: 1.0, sqrt(0.75), sqrt(0.5), sqrt(0.25), 0.0
        let expected_lrs = [1.0, 0.75f64.sqrt(), 0.5f64.sqrt(), 0.5, 0.0];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_midway() {
        let base_lr = 1.0;
        let total_iters = 5;
        let power = 1.0;
        let init_step = 2;
        let mut scheduler = PolynomialLR::new(base_lr, total_iters, power, init_step);

        // At init_step=2, factor = 1 - 2/5 = 0.6
        let expected_lrs = [0.6, 0.4, 0.2, 0.0];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_at_end() {
        let base_lr = 1.0;
        let total_iters = 5;
        let power = 1.0;
        let init_step = 5;
        let mut scheduler = PolynomialLR::new(base_lr, total_iters, power, init_step);

        // At init_step=5 (>= total_iters), lr should be 0
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.0, epsilon = 1e-10, max_relative = 1e-10);

        // Should remain at 0
        scheduler.step(0.0);
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.0, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn start_step_after_end() {
        let base_lr = 1.0;
        let total_iters = 5;
        let power = 1.0;
        let init_step = 10;
        let mut scheduler = PolynomialLR::new(base_lr, total_iters, power, init_step);

        // At init_step=10 (> total_iters), lr should be 0
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.0, epsilon = 1e-10, max_relative = 1e-10);

        // Should remain at 0
        for _ in 0..5 {
            scheduler.step(0.0);
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, 0.0, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn cubic_decay() {
        let base_lr = 2.0;
        let total_iters = 3;
        let power = 3.0;
        let init_step = 0;
        let mut scheduler = PolynomialLR::new(base_lr, total_iters, power, init_step);

        // Expected values: 2.0, 2.0*(2/3)^3, 2.0*(1/3)^3, 0.0
        let expected_lrs = [
            2.0,
            2.0 * (2.0f64 / 3.0).powf(3.0),
            2.0 * (1.0f64 / 3.0).powf(3.0),
            0.0,
        ];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn total_iters_one() {
        let base_lr = 1.0;
        let total_iters = 1;
        let power = 1.0;
        let init_step = 0;
        let mut scheduler = PolynomialLR::new(base_lr, total_iters, power, init_step);

        // At step 0, factor = 1 - 0/1 = 1.0, so lr = 1.0
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 1.0, epsilon = 1e-10, max_relative = 1e-10);

        // After one step, should be 0
        scheduler.step(0.0);
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.0, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn total_iters_zero() {
        let base_lr = 1.0;
        let total_iters = 0;
        let power = 1.0;
        let init_step = 0;
        let mut scheduler = PolynomialLR::new(base_lr, total_iters, power, init_step);

        // With total_iters=0, lr should be 0 immediately
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.0, epsilon = 1e-10, max_relative = 1e-10);

        scheduler.step(0.0);
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.0, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn different_base_lr() {
        let base_lr = 0.5;
        let total_iters = 4;
        let power = 1.0;
        let init_step = 0;
        let mut scheduler = PolynomialLR::new(base_lr, total_iters, power, init_step);

        let expected_lrs = [0.5, 0.375, 0.25, 0.125, 0.0];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }
}
