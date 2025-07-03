//! Reduces learning rate when a metric has stopped improving.
//!
//! # Examples
//!
//! This scheduler monitors loss and reduces learning rate when no improvement is seen:
//!
//! ```
//! # use lr_schedulers::reduce_lr_on_plateau::{ReduceLROnPlateau, Mode, ThresholdMode};
//! # use lr_schedulers::Scheduler;
//! let mut scheduler = ReduceLROnPlateau::new(0.1, Mode::Min, 0.5, 2, 0.001, ThresholdMode::Abs, 0, 0.0, 1e-8);
//!
//! // Simulate training with decreasing loss
//! scheduler.step(1.0); // loss=1.0, lr remains 0.1
//! scheduler.step(0.8); // loss=0.8, improvement, lr remains 0.1
//! scheduler.step(0.85); // loss=0.85, no improvement (1/2)
//! scheduler.step(0.82); // loss=0.82, no improvement (2/2), the first two epochs without improvement are ignored
//! scheduler.step(0.81); // loss=0.81, no improvement
//! // lr is now reduced to 0.05
//!
//! assert_eq!(scheduler.get_lr(0.0), 0.05);
//! ```
//!
//! Different modes for different metrics:
//!
//! ```
//! # use lr_schedulers::reduce_lr_on_plateau::{ReduceLROnPlateau, Mode, ThresholdMode};
//! # use lr_schedulers::Scheduler;
//! // For accuracy (higher is better)
//! let mut scheduler = ReduceLROnPlateau::new(0.1, Mode::Max, 0.1, 3, 0.01, ThresholdMode::Rel, 0, 0.0, 1e-8);
//!
//! scheduler.step(0.8); // accuracy=0.8
//! scheduler.step(0.85); // accuracy=0.85, improvement
//! scheduler.step(0.84); // accuracy=0.84, no improvement (1/3)
//! assert_eq!(scheduler.get_lr(0.0), 0.1); // lr not reduced yet
//! ```
//!
//! Cooldown prevents immediate consecutive reductions:
//!
//! ```
//! # use lr_schedulers::reduce_lr_on_plateau::{ReduceLROnPlateau, Mode, ThresholdMode};
//! # use lr_schedulers::Scheduler;
//! let mut scheduler = ReduceLROnPlateau::new(0.1, Mode::Min, 0.5, 1, 0.0, ThresholdMode::Abs, 2, 0.0, 1e-8);
//!
//! scheduler.step(1.0); // loss=1.0
//! scheduler.step(1.1); // loss=1.1, no improvement, lr reduced to 0.05
//! scheduler.step(1.2); // loss=1.2, in cooldown, no further reduction
//! scheduler.step(1.3); // loss=1.3, still in cooldown
//! scheduler.step(1.4); // loss=1.4, cooldown over, bad epochs start counting again
//! ```

use crate::Scheduler;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdMode {
    Rel,
    Abs,
}

#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    lr: f64,
    mode: Mode,
    factor: f64,
    patience: usize,
    threshold: f64,
    threshold_mode: ThresholdMode,
    cooldown: usize,
    min_lr: f64,
    eps: f64,
    best: f64,
    num_bad_epochs: usize,
    cooldown_counter: usize,
}

impl ReduceLROnPlateau {
    /// Constructs a ReduceLROnPlateau instance.
    /// 
    /// # Arguments
    /// 
    /// * `lr` - Initial learning rate
    /// * `mode` - Mode::Min for loss-like metrics, Mode::Max for accuracy-like metrics
    /// * `factor` - Factor by which the learning rate will be reduced (new_lr = lr * factor)
    /// * `patience` - Number of epochs with no improvement after which learning rate will be reduced
    /// * `threshold` - Threshold for measuring the new optimum
    /// * `threshold_mode` - ThresholdMode::Rel for relative threshold, ThresholdMode::Abs for absolute
    /// * `cooldown` - Number of epochs to wait before resuming normal operation after lr reduction
    /// * `min_lr` - Lower bound on the learning rate
    /// * `eps` - Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored
    pub fn new(
        lr: f64,
        mode: Mode,
        factor: f64,
        patience: usize,
        threshold: f64,
        threshold_mode: ThresholdMode,
        cooldown: usize,
        min_lr: f64,
        eps: f64,
    ) -> Self {
        let best = match mode {
            Mode::Min => f64::INFINITY,
            Mode::Max => f64::NEG_INFINITY,
        };
        
        ReduceLROnPlateau {
            lr,
            mode,
            factor,
            patience,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
            best,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        }
    }
    
    /// Check if current metric is better than the best seen so far
    fn is_better(&self, current: f64, best: f64) -> bool {
        let threshold = match self.threshold_mode {
            ThresholdMode::Rel => match self.mode {
                Mode::Min => best * (1.0 - self.threshold),
                Mode::Max => best * (1.0 + self.threshold),
            },
            ThresholdMode::Abs => match self.mode {
                Mode::Min => best - self.threshold,
                Mode::Max => best + self.threshold,
            },
        };
        
        match self.mode {
            Mode::Min => current < threshold,
            Mode::Max => current > threshold,
        }
    }
    
    /// Check if we are in cooldown period
    fn in_cooldown(&self) -> bool {
        self.cooldown_counter > 0
    }
    
    /// Reduce learning rate
    fn reduce_lr(&mut self) {
        let old_lr = self.lr;
        let new_lr = (old_lr * self.factor).max(self.min_lr);
        
        if (old_lr - new_lr).abs() > self.eps {
            self.lr = new_lr;
        }
        
        self.cooldown_counter = self.cooldown;
        self.num_bad_epochs = 0;
    }
}

impl Scheduler for ReduceLROnPlateau {
    fn step(&mut self, loss: f64) {
        let current = loss;
        
        if self.in_cooldown() {
            self.cooldown_counter -= 1;
            // Update best even during cooldown
            if self.is_better(current, self.best) {
                self.best = current;
            }
            return;
        }
        
        if self.is_better(current, self.best) {
            self.best = current;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }
        
        if self.patience == 0 {
            if self.num_bad_epochs > 0 {
                self.reduce_lr();
            }
        } else if self.num_bad_epochs >= self.patience {
            self.reduce_lr();
        }
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
    fn min_mode_basic() {
        let mut scheduler = ReduceLROnPlateau::new(
            0.1, Mode::Min, 0.5, 2, 0.001, ThresholdMode::Abs, 0, 0.0, 1e-8
        );
        
        // Initial loss, establishes best
        scheduler.step(1.0);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.1, epsilon = 1e-10);
        assert_relative_eq!(scheduler.best, 1.0, epsilon = 1e-10);
        
        // Improvement
        scheduler.step(0.8);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.1, epsilon = 1e-10);
        assert_relative_eq!(scheduler.best, 0.8, epsilon = 1e-10);
        
        // No improvement (1/2)
        scheduler.step(0.85);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.1, epsilon = 1e-10);
        assert_eq!(scheduler.num_bad_epochs, 1);
        
        // No improvement (2/2), should trigger reduction
        scheduler.step(0.82);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.05, epsilon = 1e-10);
        assert_eq!(scheduler.num_bad_epochs, 0);
    }

    #[test]
    fn max_mode_basic() {
        let mut scheduler = ReduceLROnPlateau::new(
            0.1, Mode::Max, 0.1, 3, 0.01, ThresholdMode::Rel, 0, 0.0, 1e-8
        );
        
        // Initial accuracy
        scheduler.step(0.8);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.1, epsilon = 1e-10);
        
        // Improvement
        scheduler.step(0.85);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.1, epsilon = 1e-10);
        assert_eq!(scheduler.num_bad_epochs, 0);
        
        // No improvement (1/3)
        scheduler.step(0.84);
        assert_eq!(scheduler.num_bad_epochs, 1);
        
        // No improvement (2/3)
        scheduler.step(0.83);
        assert_eq!(scheduler.num_bad_epochs, 2);
        
        // No improvement (3/3), should trigger reduction
        scheduler.step(0.82);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.01, epsilon = 1e-10);
    }

    #[test]
    fn cooldown_basic() {
        let mut scheduler = ReduceLROnPlateau::new(
            0.1, Mode::Min, 0.5, 1, 0.0, ThresholdMode::Abs, 0, 0.0, 1e-8
        );
        
        scheduler.step(1.0);
        scheduler.step(1.1); // Triggers reduction
        assert_relative_eq!(scheduler.get_lr(0.0), 0.05, epsilon = 1e-10);
        
        // With no cooldown, immediate reduction is possible
        scheduler.step(1.2);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.025, epsilon = 1e-10);
    }

    #[test]
    fn threshold_relative() {
        let mut scheduler = ReduceLROnPlateau::new(
            0.1, Mode::Min, 0.5, 1, 0.01, ThresholdMode::Rel, 0, 0.0, 1e-8
        );
        
        // Best = 1.0
        scheduler.step(1.0);
        
        // For relative threshold 0.01, improvement threshold = 1.0 * (1 - 0.01) = 0.99
        // 0.995 > 0.99, so this is NOT an improvement
        scheduler.step(0.995);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.05, epsilon = 1e-10);
        
        // Reset for next test
        let mut scheduler = ReduceLROnPlateau::new(
            0.1, Mode::Min, 0.5, 1, 0.01, ThresholdMode::Rel, 0, 0.0, 1e-8
        );
        scheduler.step(1.0);
        
        // 0.98 < 0.99, so this IS an improvement
        scheduler.step(0.98);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn threshold_absolute() {
        let mut scheduler = ReduceLROnPlateau::new(
            0.1, Mode::Min, 0.5, 1, 0.01, ThresholdMode::Abs, 0, 0.0, 1e-8
        );
        
        // Best = 1.0
        scheduler.step(1.0);
        
        // For absolute threshold 0.01, improvement threshold = 1.0 - 0.01 = 0.99
        // 0.995 > 0.99, so this is NOT an improvement
        scheduler.step(0.995);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.05, epsilon = 1e-10);
        
        // Reset for next test
        let mut scheduler = ReduceLROnPlateau::new(
            0.1, Mode::Min, 0.5, 1, 0.01, ThresholdMode::Abs, 0, 0.0, 1e-8
        );
        scheduler.step(1.0);
        
        // 0.98 < 0.99, so this IS an improvement
        scheduler.step(0.98);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn min_lr_enforcement() {
        let mut scheduler = ReduceLROnPlateau::new(
            0.1, Mode::Min, 0.5, 1, 0.0, ThresholdMode::Abs, 0, 0.03, 1e-8
        );
        
        scheduler.step(1.0);
        
        // First reduction: 0.1 * 0.5 = 0.05
        scheduler.step(1.1);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.05, epsilon = 1e-10);
        
        // Second reduction: 0.05 * 0.5 = 0.025, but min_lr = 0.03
        scheduler.step(1.2);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.03, epsilon = 1e-10);
        
        // Third reduction: should remain at min_lr
        scheduler.step(1.3);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.03, epsilon = 1e-10);
    }

    #[test]
    fn eps_threshold() {
        let mut scheduler = ReduceLROnPlateau::new(
            0.1, Mode::Min, 0.999, 1, 0.0, ThresholdMode::Abs, 0, 0.0, 1e-3
        );
        
        scheduler.step(1.0);
        
        // Reduction would be 0.1 * 0.999 = 0.0999
        // Difference = 0.1 - 0.0999 = 0.0001 < eps = 0.001
        // So LR should not change
        scheduler.step(1.1);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn patience_reset_on_improvement() {
        let mut scheduler = ReduceLROnPlateau::new(
            0.1, Mode::Min, 0.5, 3, 0.0, ThresholdMode::Abs, 0, 0.0, 1e-8
        );
        
        scheduler.step(1.0);
        
        // Two bad epochs
        scheduler.step(1.1);
        scheduler.step(1.2);
        assert_eq!(scheduler.num_bad_epochs, 2);
        
        // Improvement resets counter
        scheduler.step(0.9);
        assert_eq!(scheduler.num_bad_epochs, 0);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.1, epsilon = 1e-10);
        
        // Now need 3 more bad epochs to trigger reduction
        scheduler.step(1.0);
        scheduler.step(1.1);
        scheduler.step(1.2);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.05, epsilon = 1e-10);
    }

    #[test]
    fn zero_patience() {
        let mut scheduler = ReduceLROnPlateau::new(
            0.1, Mode::Min, 0.5, 0, 0.0, ThresholdMode::Abs, 0, 0.0, 1e-8
        );
        
        // First step establishes best = 1.0
        scheduler.step(1.0);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.1, epsilon = 1e-10);
        
        // With zero patience, any non-improvement should trigger reduction
        // 1.1 > 1.0, so bad_epochs = 1, and since patience = 0, we check > 0
        scheduler.step(1.1);
        assert_relative_eq!(scheduler.get_lr(0.0), 0.05, epsilon = 1e-10);
    }
}