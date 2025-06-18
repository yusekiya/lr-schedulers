use crate::Scheduler;

/// Decays the learning rate by a constant factor until the number of steps reaches a given number.
///
/// # Examples
///
/// This scheduler generates modified learning rates until a certain number of steps:
///
/// ```
/// # use lr_schedulers::constant::ConstantLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = ConstantLR::new(1.0, 2.0, 2, 0);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 5 {
///     // Note: loss value is not used in this scheduler.
///     learning_rates.push(scheduler.get_lr(0.01));
///     scheduler.step(0.01);
/// }
/// assert_eq!(learning_rates, [2.0, 2.0, 1.0, 1.0, 1.0]);
/// ```
///
/// Starting point can be changed with `init_step`:
///
/// ```
/// # use lr_schedulers::constant::ConstantLR;
/// # use lr_schedulers::Scheduler;
/// let init_step = 1;
/// let mut scheduler = ConstantLR::new(1.0, 2.0, 2, init_step);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 5 {
///     // Note: loss value is not used in this scheduler.
///     learning_rates.push(scheduler.get_lr(0.01));
///     scheduler.step(0.01);
/// }
/// assert_eq!(learning_rates, [2.0, 1.0, 1.0, 1.0, 1.0]);
/// ```
///
/// The `get_lr` method returns the same value unless the `step` method is invoked.
///
/// ```no_run
/// # use lr_schedulers::constant::ConstantLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = ConstantLR::new(1.0, 2.0, 2, 0);
/// // Note: loss value is not used in this scheduler.
/// let lr = scheduler.get_lr(0.01);
/// assert_eq!(lr, scheduler.get_lr(0.01));
/// scheduler.step(0.01);
/// let lr = scheduler.get_lr(0.01);
/// assert_ne!(lr, scheduler.get_lr(0.01));
/// ```
#[derive(Debug, Clone)]
pub struct ConstantLR {
    lr: f64,
    base_lr: f64,
    step: usize,
    total_iters: usize,
}

impl ConstantLR {
    /// Constructs a ConstantLR instance.
    ///
    /// This scheduler returns `factor * base_lr` before the number of steps is less than `total_iters`, otherwise, returns `base_lr`.
    /// Starting step can be specified by `init_step`. Use `init_step=0` to train a model from the beginning.
    pub fn new(base_lr: f64, factor: f64, total_iters: usize, init_step: usize) -> Self {
        let lr = if init_step < total_iters {
            factor * base_lr
        } else {
            base_lr
        };
        ConstantLR {
            lr,
            base_lr,
            step: init_step,
            total_iters,
        }
    }
}

impl Scheduler for ConstantLR {
    fn step(&mut self, _loss: f64) {
        self.step += 1;
        if self.step == self.total_iters {
            self.lr = self.base_lr
        }
    }

    fn get_lr(&self, _loss: f64) -> f64 {
        self.lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scheduler;

    #[test]
    fn middle_step() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 2;
        let init_step = 0;
        let mut scheduler = ConstantLR::new(base_lr, factor, total_iters, init_step);
        for i in 0..total_steps {
            let lr = scheduler.get_lr(0.0);
            if i < total_iters {
                let expected = factor * base_lr;
                assert_eq!(lr, expected, "Step {}", i);
            } else {
                let expected = base_lr;
                assert_eq!(lr, expected, "Step {}", i);
            }
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn early_step() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 1;
        let init_step = 0;
        let mut scheduler = ConstantLR::new(base_lr, factor, total_iters, init_step);
        for i in 0..total_steps {
            let lr = scheduler.get_lr(0.0);
            if i < total_iters {
                let expected = factor * base_lr;
                assert_eq!(lr, expected, "Step {}", i);
            } else {
                let expected = base_lr;
                assert_eq!(lr, expected, "Step {}", i);
            }
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn later_step() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 4;
        let init_step = 0;
        let mut scheduler = ConstantLR::new(base_lr, factor, total_iters, init_step);
        for i in 0..total_steps {
            let lr = scheduler.get_lr(0.0);
            if i < total_iters {
                let expected = factor * base_lr;
                assert_eq!(lr, expected, "Step {}", i);
            } else {
                let expected = base_lr;
                assert_eq!(lr, expected, "Step {}", i);
            }
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn midway_stop() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 10;
        let init_step = 0;
        let mut scheduler = ConstantLR::new(base_lr, factor, total_iters, init_step);
        for i in 0..total_steps {
            let lr = scheduler.get_lr(0.0);
            let expected = factor * base_lr;
            assert_eq!(lr, expected, "Step {}", i);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn fixed_lr() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 0;
        let init_step = 0;
        let mut scheduler = ConstantLR::new(base_lr, factor, total_iters, init_step);
        for i in 0..total_steps {
            let lr = scheduler.get_lr(0.0);
            let expected = base_lr;
            assert_eq!(lr, expected, "Step {}", i);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_before_step() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 2;
        let init_step = 1;
        let mut scheduler = ConstantLR::new(base_lr, factor, total_iters, init_step);
        for i in init_step..total_steps {
            let lr = scheduler.get_lr(0.0);
            if i < total_iters {
                let expected = factor * base_lr;
                assert_eq!(lr, expected, "Step {}", i);
            } else {
                let expected = base_lr;
                assert_eq!(lr, expected, "Step {}", i);
            }
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_after_step() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 2;
        let init_step = 3;
        let mut scheduler = ConstantLR::new(base_lr, factor, total_iters, init_step);
        for i in init_step..total_steps {
            let lr = scheduler.get_lr(0.0);
            let expected = base_lr;
            assert_eq!(lr, expected, "Step {}", i);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_on_step() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 2;
        let init_step = 2;
        let mut scheduler = ConstantLR::new(base_lr, factor, total_iters, init_step);
        for i in init_step..total_steps {
            let lr = scheduler.get_lr(0.0);
            let expected = base_lr;
            assert_eq!(lr, expected, "Step {}", i);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn fixed_lr_with_init_step() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 0;
        let init_step = 2;
        let mut scheduler = ConstantLR::new(base_lr, factor, total_iters, init_step);
        for i in init_step..total_steps {
            let lr = scheduler.get_lr(0.0);
            let expected = base_lr;
            assert_eq!(lr, expected, "Step {}", i);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }
}
