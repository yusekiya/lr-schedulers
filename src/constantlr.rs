use crate::Scheduler;

/// Decays the learning rate by a constant factor until the number of epoch reaches a given number.
#[derive(Debug)]
pub struct ConstantLR {
    lr: f64,
    base_lr: f64,
    epoch: usize,
    total_iters: usize,
}

impl ConstantLR {
    /// Construct a ConstantLR instance.
    /// 
    /// This scheduler returns `factor * base_lr` before the number of epoch is less than `total_iters`, otherwise, returns `base_lr`.
    /// Starting epoch can be specified by `init_epoch`. Use `init_epoch=0` to train a model from the beginning.
    pub fn new(base_lr: f64, factor: f64, total_iters: usize, init_epoch: usize) -> Self {
        let lr = if init_epoch < total_iters {
            factor * base_lr
        } else {
            base_lr
        };
        ConstantLR {
            lr,
            base_lr,
            epoch: init_epoch,
            total_iters,
        }
    }
}

impl Scheduler for ConstantLR {
    fn step(&mut self, _loss: f64) {
        self.epoch += 1;
        if self.epoch == self.total_iters {
            self.lr = self.base_lr
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
    fn middle_step() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 2;
        let init_epoch = 0;
        let mut scheduler = ConstantLR::new(
            base_lr, factor, total_iters, init_epoch
        );
        for i in 0 .. total_steps {
            let lr = scheduler.get_lr();
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
        let init_epoch = 0;
        let mut scheduler = ConstantLR::new(
            base_lr, factor, total_iters, init_epoch
        );
        for i in 0 .. total_steps {
            let lr = scheduler.get_lr();
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
        let init_epoch = 0;
        let mut scheduler = ConstantLR::new(
            base_lr, factor, total_iters, init_epoch
        );
        for i in 0 .. total_steps {
            let lr = scheduler.get_lr();
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
        let init_epoch = 0;
        let mut scheduler = ConstantLR::new(
            base_lr, factor, total_iters, init_epoch
        );
        for i in 0 .. total_steps {
            let lr = scheduler.get_lr();
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
        let init_epoch = 0;
        let mut scheduler = ConstantLR::new(
            base_lr, factor, total_iters, init_epoch
        );
        for i in 0 .. total_steps {
            let lr = scheduler.get_lr();
            let expected = base_lr;
            assert_eq!(lr, expected, "Step {}", i);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_epoch_before_step() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 2;
        let init_epoch = 1;
        let mut scheduler = ConstantLR::new(
            base_lr, factor, total_iters, init_epoch
        );
        for i in init_epoch .. total_steps {
            let lr = scheduler.get_lr();
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
    fn start_epoch_after_step() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 2;
        let init_epoch = 3;
        let mut scheduler = ConstantLR::new(
            base_lr, factor, total_iters, init_epoch
        );
        for i in init_epoch .. total_steps {
            let lr = scheduler.get_lr();
            let expected = base_lr;
            assert_eq!(lr, expected, "Step {}", i);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }

    #[test]
    fn start_epoch_on_step() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 2;
        let init_epoch = 2;
        let mut scheduler = ConstantLR::new(
            base_lr, factor, total_iters, init_epoch
        );
        for i in init_epoch .. total_steps {
            let lr = scheduler.get_lr();
            let expected = base_lr;
            assert_eq!(lr, expected, "Step {}", i);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }
 
    #[test]
    fn fixed_lr_with_init_epoch() {
        let total_steps = 5;
        let base_lr = 0.5;
        let factor = 0.1;
        let total_iters = 0;
        let init_epoch = 2;
        let mut scheduler = ConstantLR::new(
            base_lr, factor, total_iters, init_epoch
        );
        for i in init_epoch .. total_steps {
            let lr = scheduler.get_lr();
            let expected = base_lr;
            assert_eq!(lr, expected, "Step {}", i);
            // Proceed a step with dummy loss.
            scheduler.step(0.0);
        }
    }
}