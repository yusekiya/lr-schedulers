use crate::Scheduler;

const PI: f64 = std::f64::consts::PI;

/// Change the learning rate periodically.
/// 
/// # Examples
/// 
/// This scheduler generates periodically changing learning rates:
/// 
/// ```
/// # use lr_schedulers::cosine_annealing::CosineAnnealingLR;
/// # use lr_schedulers::Scheduler;
/// # use std::iter::zip;
/// let mut scheduler = CosineAnnealingLR::new(1.0, 0.0, 2, 0);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 5 {
///     learning_rates.push(scheduler.get_lr());
///     scheduler.step(0.01); // Note: loss value is not used in this scheduler.
/// }
/// for (target, expected) in zip(learning_rates, [1.0, 0.5, 0.0, 0.5, 1.0]) {
///     assert!((target - expected).abs() < 1e-10);
/// }
/// ```
/// 
/// Starting point can be changed with `init_step`:
/// 
/// ```
/// # use lr_schedulers::cosine_annealing::CosineAnnealingLR;
/// # use lr_schedulers::Scheduler;
/// # use std::iter::zip;
/// let init_step = 1;
/// let mut scheduler = CosineAnnealingLR::new(1.0, 0.0, 2, init_step);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 5 {
///     learning_rates.push(scheduler.get_lr());
///     scheduler.step(0.01); // Note: loss value is not used in this scheduler.
/// }
/// for (target, expected) in zip(learning_rates, [0.5, 0.0, 0.5, 1.0, 0.5]) {
///     assert!((target - expected).abs() < 1e-10);
/// }
/// ```
/// 
/// The `step` method should be called after training:
/// 
/// ```no_run
/// # use lr_schedulers::cosine_annealing::CosineAnnealingLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = CosineAnnealingLR::new(1.0, 0.0, 2, 0);
/// for epoch in 0 .. 10 {
///     let lr = scheduler.get_lr();
///     // Run training with `lr` and calculate loss
/// #    let loss = 0.001;
///     scheduler.step(loss);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    lr: f64,
    eta_0: f64,
    eta_1: f64,
    step: usize,
    t_max: usize,
}

impl CosineAnnealingLR {
    /// Construct a CosineAnnealingLR instance.
    /// 
    /// This scheduler returns learning rate that oscillates between `eta_0` and `eta_1` with a period of `2*t_max`.
    /// The parameter `t_max` must be larger than 0. When 0 is provided, its value is replaced with 1.
    /// Starting step can be specified by `init_step`. Use `init_step=0` to train a model from the beginning.
    pub fn new(
        eta_0: f64,
        eta_1: f64,
        t_max: usize,
        init_step: usize,
    ) -> Self {
        let t_max = t_max.max(1);
        let lr = if init_step == 0 {
            eta_0
        } else {
            let periodic_factor = periodic_factor(init_step, t_max);
            (eta_0 - eta_1).mul_add(periodic_factor, eta_1)
        };
        CosineAnnealingLR { lr, eta_0, eta_1, step: init_step, t_max }
    }
}

impl Scheduler for CosineAnnealingLR {
    fn step(&mut self, _loss: f64) {
        self.step += 1;
        let periodic_factor = periodic_factor(self.step, self.t_max);
        self.lr = (self.eta_0 - self.eta_1).mul_add(periodic_factor, self.eta_1);
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }
}

fn periodic_factor(t: usize, t_max: usize) -> f64 {
    let r = t.rem_euclid(2*t_max);
    let phase = (r as f64) * PI / (t_max as f64);
    0.5 * (1.0 + phase.cos())
}

#[cfg(test)]
mod tests {
    use approx::relative_eq;
    use crate::Scheduler;
    use super::*;

    #[test]
    fn decrease_first_lr() {
        let eta_0 = 1.0;
        let eta_1 = 0.0;
        let t_max = 2;
        let init_step = 0;
        let mut scheduler = CosineAnnealingLR::new(
            eta_0, eta_1, t_max, init_step
        );
        let expected_lrs = [1.0, 0.5, 0.0, 0.5, 1.0];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr();
            assert!(relative_eq!(lr, *exp_lr), "Step {}: left: {}, right: {}", i, lr, *exp_lr);
            // Process a step with dummy loss
            scheduler.step(0.0);
        }
    }

    #[test]
    fn increase_first_lr() {
        let eta_0 = 0.0;
        let eta_1 = 1.0;
        let t_max = 2;
        let init_step = 0;
        let mut scheduler = CosineAnnealingLR::new(
            eta_0, eta_1, t_max, init_step
        );
        let expected_lrs = [0.0, 0.5, 1.0, 0.5, 0.0];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr();
            assert!(relative_eq!(lr, *exp_lr), "Step {}: left: {}, right: {}", i, lr, *exp_lr);
            // Process a step with dummy loss
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_midway() {
        let eta_0 = 1.0;
        let eta_1 = 0.0;
        let t_max = 2;
        let init_step = 1;
        let mut scheduler = CosineAnnealingLR::new(
            eta_0, eta_1, t_max, init_step
        );
        let expected_lrs = [0.5, 0.0, 0.5, 1.0, 0.5];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr();
            assert!(relative_eq!(lr, *exp_lr), "Step {}: left: {}, right: {}", i, lr, *exp_lr);
            // Process a step with dummy loss
            scheduler.step(0.0);
        }
    }

    #[test]
    fn long_repeat() {
        let repeat = 1000;
        let eta_0 = 1.0;
        let eta_1 = 0.0;
        let t_max = 2;
        let init_step = 0;
        let mut scheduler = CosineAnnealingLR::new(
            eta_0, eta_1, t_max, init_step
        );
        let expected_lrs = [1.0, 0.5, 0.0, 0.5].repeat(repeat);
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr();
            assert!(relative_eq!(lr, *exp_lr), "Step {}: left: {}, right: {}", i, lr, *exp_lr);
            // Process a step with dummy loss
            scheduler.step(0.0);
        }
    }
}