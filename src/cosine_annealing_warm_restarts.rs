use crate::Scheduler;

const PI: f64 = std::f64::consts::PI;

/// Changes the learning rate periodically with warmups.
/// 
/// # Examples
/// 
/// This scheduler generates periodically changing learning rates with warmups:
/// 
/// ```
/// # use lr_schedulers::cosine_annealing_warm_restarts::CosineAnnealingWarmRestarts;
/// # use lr_schedulers::Scheduler;
/// # use std::iter::zip;
/// let mut scheduler = CosineAnnealingWarmRestarts::new(1.0, 0.0, 2, 1, 0);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 5 {
///     // Note: loss value is not used in this scheduler.
///     learning_rates.push(scheduler.get_lr(0.01));
///     scheduler.step(0.01);
/// }
/// for (target, expected) in zip(learning_rates, [1.0, 0.5, 0.0, 1.0, 0.5]) {
///     assert!((target - expected).abs() < 1e-10);
/// }
/// ```
/// 
/// The length of period is multiplied by `t_mult` after every warm restarts:
/// 
/// ```
/// # use lr_schedulers::cosine_annealing_warm_restarts::CosineAnnealingWarmRestarts;
/// # use lr_schedulers::Scheduler;
/// # use std::iter::zip;
/// let t_mult = 2;
/// let mut scheduler = CosineAnnealingWarmRestarts::new(1.0, 0.0, 2, t_mult, 0);
/// let mut learning_rates = Vec::new();
/// for _ in 0 .. 8 {
///     // Note: loss value is not used in this scheduler.
///     learning_rates.push(scheduler.get_lr(0.01));
///     scheduler.step(0.01);
/// }
/// let expected_lrs = [
///     1.0,
///     0.5,
///     0.0,
///     1.0,
///     (1.0 + 1.0/2.0f64.sqrt())/2.0,
///     0.5,
///     (1.0 - 1.0/2.0f64.sqrt())/2.0,
///     0.0
/// ];
/// for (target, expected) in zip(learning_rates, expected_lrs) {
///     assert!((target - expected).abs() < 1e-10);
/// }
/// ```
/// 
/// The `get_lr` method returns the same value unless the `step` method is invoked.
/// 
/// ```no_run
/// # use lr_schedulers::cosine_annealing_warm_restarts::CosineAnnealingWarmRestarts;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = CosineAnnealingWarmRestarts::new(1.0, 0.0, 2, 1, 0);
/// // Note: loss value is not used in this scheduler.
/// let lr = scheduler.get_lr(0.01);
/// assert_eq!(lr, scheduler.get_lr(0.01));
/// scheduler.step(0.01);
/// let lr = scheduler.get_lr(0.01);
/// assert_ne!(lr, scheduler.get_lr(0.01));
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingWarmRestarts {
    lr: f64,
    eta_0: f64,
    eta_1: f64,
    step_cur: usize,
    t_max: usize,
    t_mult: usize,
}

impl CosineAnnealingWarmRestarts {
    /// Constructs a CosineAnnealingWarmRestarts instance.
    /// 
    /// This scheduler returns learning rates that changes from `eta_0` to `eta_1` using a cosine function.
    /// The learning rate is reset to `eta_0` at the beginning of period, hence the name warm restarts.
    /// The length of period is given by `t_0`, and the period is multiplied by `t_mult` after every warm restarts.
    /// The parameters `t_0` and `t_mult` must be larger than 0. When 0 is provided, their values are replaced with 1.
    /// Starting step can be specified by `init_step`. Use `init_step=0` to train a model from the beginning.
    pub fn new(
        eta_0: f64,
        eta_1: f64,
        t_0: usize,
        t_mult: usize,
        init_step: usize,
    ) -> Self {
        // When t_mult = 0 is given, replace it to 1 to prevent infinite loop.
        let t_mult = t_mult.max(1);
        // Aboid t_0 = 0 for the same reason as above.
        let t_0 = t_0.max(1);
        let (lr, step_cur, t_max) = if init_step == 0 {
            (eta_0, 0, t_0)
        } else {
            let mut step = init_step;
            let mut t_max = t_0;
            while step > t_max {
                step -= t_max + 1;
                t_max *= t_mult;
            }
            let periodic_factor = periodic_factor(step, t_max);
            let lr = (eta_0 - eta_1).mul_add(periodic_factor, eta_1);
            (lr, step, t_max)
        };
        CosineAnnealingWarmRestarts { lr, eta_0, eta_1, step_cur, t_max, t_mult }
    }
}

impl Scheduler for CosineAnnealingWarmRestarts {
    fn step(&mut self, _loss: f64) {
        self.step_cur += 1;
        while self.step_cur > self.t_max {
            self.step_cur -= self.t_max + 1;
            self.t_max *= self.t_mult;
        }
        let periodic_factor = periodic_factor(self.step_cur, self.t_max);
        self.lr = (self.eta_0 - self.eta_1).mul_add(periodic_factor, self.eta_1);
    }

    fn get_lr(&self, _loss: f64) -> f64 {
        self.lr
    }
}

fn periodic_factor(t: usize, t_max: usize) -> f64 {
    let phase = (t as f64) * PI / (t_max as f64);
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
        let t_0 = 2;
        let t_mult = 1;
        let init_step = 0;
        let mut scheduler = CosineAnnealingWarmRestarts::new(
            eta_0, eta_1, t_0, t_mult, init_step
        );
        let expected_lrs = [1.0, 0.5, 0.0, 1.0, 0.5, 0.0];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr(0.01);
            assert!(relative_eq!(lr, *exp_lr), "Step {}: left: {}, right: {}", i, lr, *exp_lr);
            // Process a step with dummy loss
            scheduler.step(0.0);
        }
    }

    #[test]
    fn scaled_period() {
        let eta_0 = 1.0;
        let eta_1 = 0.0;
        let t_0 = 2;
        let t_mult = 2;
        let init_step = 0;
        let mut scheduler = CosineAnnealingWarmRestarts::new(
            eta_0, eta_1, t_0, t_mult, init_step
        );
        let expected_lrs = [
            1.0,
            0.5,
            0.0,
            1.0,
            (1.0 + 1.0/2.0f64.sqrt())/2.0,
            0.5,
            (1.0 - 1.0/2.0f64.sqrt())/2.0,
            0.0
        ];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr(0.01);
            assert!(relative_eq!(lr, *exp_lr), "Step {}: left: {}, right: {}", i, lr, *exp_lr);
            // Process a step with dummy loss
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_midway() {
        let eta_0 = 1.0;
        let eta_1 = 0.0;
        let t_0 = 2;
        let t_mult = 1;
        let init_step = 2;
        let mut scheduler = CosineAnnealingWarmRestarts::new(
            eta_0, eta_1, t_0, t_mult, init_step
        );
        let expected_lrs = [0.0, 1.0, 0.5, 0.0, 1.0, 0.5];
        for (i, exp_lr) in expected_lrs.iter().enumerate() {
            let lr = scheduler.get_lr(0.01);
            assert!(relative_eq!(lr, *exp_lr), "Step {}: left: {}, right: {}", i, lr, *exp_lr);
            // Process a step with dummy loss
            scheduler.step(0.0);
        }
    }
}