use crate::Scheduler;

/// Change the learning rate linearly until the number of steps reaches a given number.
/// 
/// # Examples
/// 
/// This scheduler generates linearly changing learning rates until a certain number of steps:
/// 
/// ```
/// # use lr_schedulers::linear::LinearLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = LinearLR::new(1.0, 2.0, 0.5, 2, 0);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 5 {
///     learning_rates.push(scheduler.get_lr());
///     scheduler.step(0.01); // Note: loss value is not used in this scheduler.
/// }
/// assert_eq!(learning_rates, [2.0, 1.25, 0.5, 0.5, 0.5]);
/// ```
/// 
/// Starting point can be changed with `init_step`:
/// 
/// ```
/// # use lr_schedulers::linear::LinearLR;
/// # use lr_schedulers::Scheduler;
/// let init_step = 1;
/// let mut scheduler = LinearLR::new(1.0, 2.0, 0.5, 2, init_step);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 5 {
///     learning_rates.push(scheduler.get_lr());
///     scheduler.step(0.01); // Note: loss value is not used in this scheduler.
/// }
/// assert_eq!(learning_rates, [1.25, 0.5, 0.5, 0.5, 0.5]);
/// ```
/// 
/// The `step` method should be called after training:
/// 
/// ```no_run
/// # use lr_schedulers::linear::LinearLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = LinearLR::new(1.0, 2.0, 0.5, 5, 0);
/// for epoch in 0 .. 10 {
///     let lr = scheduler.get_lr();
///     // Run training with `lr` and calculate loss
/// #    let loss = 0.001;
///     scheduler.step(loss);
/// }
/// ```
#[derive(Debug)]
pub struct LinearLR {
    lr: f64,
    base_lr: f64,
    step: usize,
    total_iters: usize,
    grad: f64,
    start_factor: f64,
    end_factor: f64,
}

impl LinearLR {
    /// Construct a LinearLR instance.
    /// 
    /// This scheduler returns learning rate that interpolates `start_factor * base_lr` and `end_factor * base_lr` when the number of steps is between 0 and `total_iters`.
    /// After that, this returns `end_factor * base_lr`.
    /// Starting step can be specified by `init_step`. Use `init_step=0` to train a model from the beginning.
    pub fn new(
        base_lr: f64,
        start_factor: f64,
        end_factor: f64,
        total_iters: usize,
        init_step: usize
    ) -> Self {
        if init_step >= total_iters {
            LinearLR {
                lr: end_factor * base_lr,
                base_lr,
                step: init_step,
                total_iters,
                grad: 0.0, // Dummy gradient
                start_factor,
                end_factor,
            }
        } else if init_step == 0 {
            let grad = (end_factor - start_factor) / (total_iters as f64);
            LinearLR {
                lr: start_factor * base_lr,
                base_lr,
                step: 0,
                total_iters,
                grad,
                start_factor,
                end_factor,
            }
        } else {
            let grad = (end_factor - start_factor) / (total_iters as f64);
            let lr = base_lr * (init_step as f64).mul_add(grad, start_factor);
            LinearLR {
                lr,
                base_lr,
                step: init_step,
                total_iters,
                grad,
                start_factor,
                end_factor,
            }
        }
    }
}

impl Scheduler for LinearLR {
    fn step(&mut self, _loss: f64) {
        self.step += 1;
        if self.step >= self.total_iters {
            self.lr = self.end_factor * self.base_lr;
        } else {
            self.lr = self.base_lr * (self.step as f64).mul_add(self.grad, self.start_factor);
        }
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }
}

#[cfg(test)]
mod tests {
    use crate::Scheduler;
    use super::*;

    #[test]
    fn decrease_lr() {
        let base_lr = 1.0;
        let start_factor = 2.0;
        let end_factor = 0.5;
        let total_iters = 2;
        let init_step = 0;
        let mut scheduler = LinearLR::new(
            base_lr, start_factor, end_factor, total_iters, init_step
        );
        let expected_lrs = [2.0, 1.25, 0.5, 0.5, 0.5];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr();
            assert_eq!(lr, *exp_lr, "Step {}", i);
            // Proceed a step wit dummy loss
            scheduler.step(0.0);
        }
    }

    #[test]
    fn increase_lr() {
        let base_lr = 1.0;
        let start_factor = 0.5;
        let end_factor = 2.0;
        let total_iters = 2;
        let init_step = 0;
        let mut scheduler = LinearLR::new(
            base_lr, start_factor, end_factor, total_iters, init_step
        );
        let expected_lrs = [0.5, 1.25, 2.0, 2.0, 2.0];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr();
            assert_eq!(lr, *exp_lr, "Step {}", i);
            // Proceed a step wit dummy loss
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_before_total_iters() {
        let base_lr = 1.0;
        let start_factor = 0.5;
        let end_factor = 2.0;
        let total_iters = 2;
        let init_step = 1;
        let mut scheduler = LinearLR::new(
            base_lr, start_factor, end_factor, total_iters, init_step
        );
        let expected_lrs = [1.25, 2.0, 2.0, 2.0];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr();
            assert_eq!(lr, *exp_lr, "Step {}", i);
            // Proceed a step wit dummy loss
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_after_total_iters() {
        let base_lr = 1.0;
        let start_factor = 0.5;
        let end_factor = 2.0;
        let total_iters = 2;
        let init_step = 3;
        let mut scheduler = LinearLR::new(
            base_lr, start_factor, end_factor, total_iters, init_step
        );
        let expected_lrs = [2.0, 2.0];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr();
            assert_eq!(lr, *exp_lr, "Step {}", i);
            // Proceed a step wit dummy loss
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_on_total_iters() {
        let base_lr = 1.0;
        let start_factor = 0.5;
        let end_factor = 2.0;
        let total_iters = 2;
        let init_step = 2;
        let mut scheduler = LinearLR::new(
            base_lr, start_factor, end_factor, total_iters, init_step
        );
        let expected_lrs = [2.0, 2.0, 2.0];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr();
            assert_eq!(lr, *exp_lr, "Step {}", i);
            // Proceed a step wit dummy loss
            scheduler.step(0.0);
        }
    }
}