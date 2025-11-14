use crate::Scheduler;

/// Decays the learning rate by gamma at specified milestone epochs.
///
/// # Examples
///
/// This scheduler generates step-wise decaying learning rates at specified milestones:
///
/// ```
/// # use lr_schedulers::multistep::MultiStepLR;
/// # use lr_schedulers::Scheduler;
/// # use approx::assert_relative_eq;
/// let mut scheduler = MultiStepLR::new(1.0, vec![3, 7], 0.1, 0);
/// let expected = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01];
/// for exp_lr in expected.iter() {
///     assert_relative_eq!(scheduler.get_lr(), exp_lr, epsilon = 1e-10);
///     scheduler.step(0.01);
/// }
/// ```
///
/// Multiple milestones with different gamma:
///
/// ```
/// # use lr_schedulers::multistep::MultiStepLR;
/// # use lr_schedulers::Scheduler;
/// # use approx::assert_relative_eq;
/// let mut scheduler = MultiStepLR::new(0.5, vec![2, 5, 8], 0.5, 0);
/// let expected = [0.5, 0.5, 0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.0625, 0.0625];
/// for exp_lr in expected.iter() {
///     assert_relative_eq!(scheduler.get_lr(), exp_lr, epsilon = 1e-10);
///     scheduler.step(0.01);
/// }
/// ```
///
/// Starting point can be changed with `init_step`:
///
/// ```
/// # use lr_schedulers::multistep::MultiStepLR;
/// # use lr_schedulers::Scheduler;
/// # use approx::assert_relative_eq;
/// let init_step = 4;
/// let mut scheduler = MultiStepLR::new(1.0, vec![3, 7], 0.1, init_step);
/// let expected = [0.1, 0.1, 0.1, 0.01, 0.01];
/// for exp_lr in expected.iter() {
///     assert_relative_eq!(scheduler.get_lr(), exp_lr, epsilon = 1e-10);
///     scheduler.step(0.01);
/// }
/// ```
///
/// The `get_lr` method returns the same value unless the `step` method is invoked.
///
/// ```no_run
/// # use lr_schedulers::multistep::MultiStepLR;
/// # use lr_schedulers::Scheduler;
/// let mut scheduler = MultiStepLR::new(1.0, vec![3, 7], 0.1, 0);
/// let lr = scheduler.get_lr();
/// assert_eq!(lr, scheduler.get_lr());
/// scheduler.step(0.01);
/// let new_lr = scheduler.get_lr();
/// assert_ne!(lr, new_lr);
/// ```
#[derive(Debug, Clone)]
pub struct MultiStepLR {
    lr: f64,
    base_lr: f64,
    step: usize,
    milestones: Vec<usize>,
    gamma: f64,
}

impl MultiStepLR {
    /// Constructs a MultiStepLR instance.
    ///
    /// This scheduler decays the learning rate by `gamma` when the current epoch
    /// reaches one of the milestones. The learning rate is calculated as:
    /// lr = base_lr * gamma^(number of milestones passed)
    ///
    /// Starting step can be specified by `init_step`. Use `init_step=0` to train a model from the beginning.
    pub fn new(base_lr: f64, mut milestones: Vec<usize>, gamma: f64, init_step: usize) -> Self {
        // Sort milestones to ensure they are in increasing order
        milestones.sort_unstable();

        // Count how many milestones have been passed at init_step
        let milestones_passed = milestones.iter().filter(|&&m| m <= init_step).count();
        let lr = base_lr * gamma.powi(milestones_passed as i32);

        MultiStepLR {
            lr,
            base_lr,
            step: init_step,
            milestones,
            gamma,
        }
    }
}

impl Scheduler for MultiStepLR {
    fn step(&mut self, _loss: f64) {
        self.step += 1;
        // Count how many milestones have been passed
        let milestones_passed = self.milestones.iter().filter(|&&m| m <= self.step).count();
        self.lr = self.base_lr * self.gamma.powi(milestones_passed as i32);
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
    fn basic_milestones() {
        let base_lr = 1.0;
        let milestones = vec![3, 7];
        let gamma = 0.1;
        let init_step = 0;
        let mut scheduler = MultiStepLR::new(base_lr, milestones, gamma, init_step);

        let expected_lrs = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn multiple_milestones_different_gamma() {
        let base_lr = 0.5;
        let milestones = vec![2, 5, 8];
        let gamma = 0.5;
        let init_step = 0;
        let mut scheduler = MultiStepLR::new(base_lr, milestones, gamma, init_step);

        let expected_lrs = [
            0.5, 0.5, 0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.0625, 0.0625,
        ];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn single_milestone() {
        let base_lr = 2.0;
        let milestones = vec![5];
        let gamma = 0.2;
        let init_step = 0;
        let mut scheduler = MultiStepLR::new(base_lr, milestones, gamma, init_step);

        // First 5 steps should have base_lr
        for _ in 0..5 {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, base_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }

        // After milestone, should be base_lr * gamma
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, base_lr * gamma, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn unsorted_milestones() {
        let base_lr = 1.0;
        let milestones = vec![7, 3, 10]; // Unsorted input
        let gamma = 0.1;
        let init_step = 0;
        let mut scheduler = MultiStepLR::new(base_lr, milestones, gamma, init_step);

        // Should work correctly even with unsorted input
        // Milestones should be treated as [3, 7, 10]
        let expected_lrs = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_after_milestone() {
        let base_lr = 1.0;
        let milestones = vec![3, 7];
        let gamma = 0.1;
        let init_step = 4;
        let mut scheduler = MultiStepLR::new(base_lr, milestones, gamma, init_step);

        // At init_step=4, one milestone (3) has been passed
        let expected_lrs = [0.1, 0.1, 0.1, 0.01, 0.01];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_at_milestone() {
        let base_lr = 1.0;
        let milestones = vec![3, 7];
        let gamma = 0.1;
        let init_step = 3;
        let mut scheduler = MultiStepLR::new(base_lr, milestones, gamma, init_step);

        // At init_step=3, one milestone (3) has been passed
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.1, epsilon = 1e-10, max_relative = 1e-10);

        scheduler.step(0.0);
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.1, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn gamma_greater_than_one() {
        let base_lr = 0.1;
        let milestones = vec![2, 5];
        let gamma = 2.0;
        let init_step = 0;
        let mut scheduler = MultiStepLR::new(base_lr, milestones, gamma, init_step);

        let expected_lrs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.4, 0.4];
        for exp_lr in expected_lrs.iter() {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, *exp_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn empty_milestones() {
        let base_lr = 1.0;
        let milestones = vec![];
        let gamma = 0.1;
        let init_step = 0;
        let mut scheduler = MultiStepLR::new(base_lr, milestones, gamma, init_step);

        // With no milestones, learning rate should remain constant
        for _ in 0..10 {
            let lr = scheduler.get_lr();
            assert_relative_eq!(lr, base_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_step_after_all_milestones() {
        let base_lr = 1.0;
        let milestones = vec![2, 5];
        let gamma = 0.1;
        let init_step = 10;
        let mut scheduler = MultiStepLR::new(base_lr, milestones, gamma, init_step);

        // At init_step=10, both milestones have been passed
        // lr = base_lr * gamma^2 = 1.0 * 0.1^2 = 0.01
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.01, epsilon = 1e-10, max_relative = 1e-10);

        // Should remain at this level
        scheduler.step(0.0);
        let lr = scheduler.get_lr();
        assert_relative_eq!(lr, 0.01, epsilon = 1e-10, max_relative = 1e-10);
    }
}
