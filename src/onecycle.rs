//! Implements the 1cycle learning rate policy.
//!
//! This scheduler anneals the learning rate from an initial learning rate to a maximum
//! learning rate and then from that maximum learning rate to a minimum learning rate
//! much lower than the initial learning rate, following the "Super-Convergence" approach.
//!
//! # Examples
//!
//! Basic usage with cosine annealing:
//!
//! ```
//! # use lr_schedulers::onecycle::{OneCycleLR, AnnealStrategy};
//! # use lr_schedulers::Scheduler;
//! let mut scheduler = OneCycleLR::new(0.1, 100, 0.3, AnnealStrategy::Cos, 25.0, 10000.0, false);
//! let mut learning_rates = Vec::new();
//! for _ in 0 .. 10 {
//!     learning_rates.push(scheduler.get_lr(0.0));
//!     scheduler.step(0.0);
//! }
//! // Forms 1cycle pattern: low -> high -> very low
//! ```
//!
//! Linear annealing strategy:
//!
//! ```
//! # use lr_schedulers::onecycle::{OneCycleLR, AnnealStrategy};
//! # use lr_schedulers::Scheduler;
//! let mut scheduler = OneCycleLR::new(0.1, 50, 0.2, AnnealStrategy::Linear, 10.0, 1000.0, false);
//! let mut learning_rates = Vec::new();
//! for _ in 0 .. 8 {
//!     learning_rates.push(scheduler.get_lr(0.0));
//!     scheduler.step(0.0);
//! }
//! // Uses linear interpolation for smoother transitions
//! ```
//!
//! Three-phase schedule:
//!
//! ```
//! # use lr_schedulers::onecycle::{OneCycleLR, AnnealStrategy};
//! # use lr_schedulers::Scheduler;
//! let mut scheduler = OneCycleLR::new(0.1, 60, 0.3, AnnealStrategy::Cos, 25.0, 10000.0, true);
//! let mut learning_rates = Vec::new();
//! for _ in 0 .. 12 {
//!     learning_rates.push(scheduler.get_lr(0.0));
//!     scheduler.step(0.0);
//! }
//! // Three phases: warmup -> annealing -> final annealing
//! ```

use crate::Scheduler;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnnealStrategy {
    Cos,
    Linear,
}

#[derive(Debug, Clone)]
pub struct OneCycleLR {
    max_lr: f64,
    total_steps: usize,
    #[allow(dead_code)]
    pct_start: f64,
    anneal_strategy: AnnealStrategy,
    #[allow(dead_code)]
    div_factor: f64,
    #[allow(dead_code)]
    final_div_factor: f64,
    three_phase: bool,
    step_count: usize,
    // Calculated values
    initial_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    annealing_steps: usize,
    final_annealing_steps: usize,
}

impl OneCycleLR {
    /// Constructs a OneCycleLR instance.
    /// 
    /// # Arguments
    /// 
    /// * `max_lr` - Maximum learning rate
    /// * `total_steps` - Total number of steps in the cycle
    /// * `pct_start` - Percentage of cycle spent increasing learning rate
    /// * `anneal_strategy` - Annealing strategy (Cos or Linear)
    /// * `div_factor` - Factor to determine initial learning rate (initial_lr = max_lr / div_factor)
    /// * `final_div_factor` - Factor to determine minimum learning rate (min_lr = initial_lr / final_div_factor)
    /// * `three_phase` - If true, use three-phase schedule; if false, use two-phase
    /// 
    pub fn new(
        max_lr: f64,
        total_steps: usize,
        pct_start: f64,
        anneal_strategy: AnnealStrategy,
        div_factor: f64,
        final_div_factor: f64,
        three_phase: bool,
    ) -> Self {
        let initial_lr = max_lr / div_factor;
        let min_lr = initial_lr / final_div_factor;
        
        let warmup_steps = (total_steps as f64 * pct_start).floor() as usize;
        
        let (annealing_steps, final_annealing_steps) = if three_phase {
            let annealing_steps = (total_steps as f64 * (1.0 - pct_start) / 2.0).floor() as usize;
            let final_annealing_steps = total_steps - warmup_steps - annealing_steps;
            (annealing_steps, final_annealing_steps)
        } else {
            let annealing_steps = total_steps - warmup_steps;
            (annealing_steps, 0)
        };
        
        OneCycleLR {
            max_lr,
            total_steps,
            pct_start,
            anneal_strategy,
            div_factor,
            final_div_factor,
            three_phase,
            step_count: 0,
            initial_lr,
            min_lr,
            warmup_steps,
            annealing_steps,
            final_annealing_steps,
        }
    }
    
    /// Get current phase and progress within that phase
    fn get_phase_and_progress(&self) -> (usize, f64) {
        if self.step_count <= self.warmup_steps {
            // Phase 1: Warmup
            let progress = if self.warmup_steps > 0 {
                self.step_count as f64 / self.warmup_steps as f64
            } else {
                1.0
            };
            (1, progress)
        } else if self.step_count <= self.warmup_steps + self.annealing_steps {
            // Phase 2: Annealing
            let steps_in_phase = self.step_count - self.warmup_steps;
            let progress = if self.annealing_steps > 0 {
                steps_in_phase as f64 / self.annealing_steps as f64
            } else {
                1.0
            };
            (2, progress)
        } else {
            // Phase 3: Final annealing (only in three-phase mode)
            let steps_in_phase = self.step_count - self.warmup_steps - self.annealing_steps;
            let progress = if self.final_annealing_steps > 0 {
                steps_in_phase as f64 / self.final_annealing_steps as f64
            } else {
                1.0
            };
            (3, progress.min(1.0))
        }
    }
    
    /// Calculate learning rate based on phase and progress
    fn calculate_lr(&self, phase: usize, progress: f64) -> f64 {
        match phase {
            1 => {
                // Phase 1: Warmup from initial_lr to max_lr
                self.interpolate(self.initial_lr, self.max_lr, progress)
            }
            2 => {
                // Phase 2: Annealing from max_lr to target
                let target_lr = if self.three_phase {
                    self.initial_lr
                } else {
                    self.min_lr
                };
                self.interpolate(self.max_lr, target_lr, progress)
            }
            3 => {
                // Phase 3: Final annealing from initial_lr to min_lr
                self.interpolate(self.initial_lr, self.min_lr, progress)
            }
            _ => self.min_lr,
        }
    }
    
    /// Interpolate between start and end values based on strategy
    fn interpolate(&self, start: f64, end: f64, progress: f64) -> f64 {
        match self.anneal_strategy {
            AnnealStrategy::Linear => {
                start + (end - start) * progress
            }
            AnnealStrategy::Cos => {
                let cosine_factor = 0.5 * (1.0 + (PI * progress).cos());
                end + (start - end) * cosine_factor
            }
        }
    }
}

impl Scheduler for OneCycleLR {
    fn step(&mut self, _loss: f64) {
        self.step_count += 1;
    }

    fn get_lr(&self, _loss: f64) -> f64 {
        if self.step_count >= self.total_steps {
            return self.min_lr;
        }
        
        let (phase, progress) = self.get_phase_and_progress();
        self.calculate_lr(phase, progress)
    }
}

#[cfg(test)]
mod tests {
    use crate::Scheduler;
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn two_phase_cosine() {
        let mut scheduler = OneCycleLR::new(0.1, 10, 0.3, AnnealStrategy::Cos, 10.0, 100.0, false);
        
        // Expected: 3 warmup steps, 7 annealing steps
        // initial_lr = 0.1 / 10.0 = 0.01
        // min_lr = 0.01 / 100.0 = 0.0001
        
        let lr_start = scheduler.get_lr(0.0);
        assert_relative_eq!(lr_start, 0.01, epsilon = 1e-10, max_relative = 1e-10);
        
        // Move to warmup steps
        for _ in 0..3 {
            scheduler.step(0.0);
        }
        
        // Should be at or near max_lr
        let lr_max = scheduler.get_lr(0.0);
        assert!(lr_max > 0.09); // Should be close to max_lr = 0.1
        
        // Move through annealing
        for _ in 0..7 {
            scheduler.step(0.0);
        }
        
        // Should be at min_lr
        let lr_final = scheduler.get_lr(0.0);
        assert_relative_eq!(lr_final, 0.0001, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn three_phase_linear() {
        let mut scheduler = OneCycleLR::new(0.2, 12, 0.25, AnnealStrategy::Linear, 20.0, 1000.0, true);
        
        // Expected: 3 warmup steps, 4-5 annealing steps, 4-5 final annealing steps
        // initial_lr = 0.2 / 20.0 = 0.01
        // min_lr = 0.01 / 1000.0 = 0.00001
        
        let lr_start = scheduler.get_lr(0.0);
        assert_relative_eq!(lr_start, 0.01, epsilon = 1e-10, max_relative = 1e-10);
        
        // Test progression through phases
        let mut max_lr_seen = 0.0f64;
        for _ in 0..12 {
            let lr = scheduler.get_lr(0.0);
            max_lr_seen = max_lr_seen.max(lr);
            scheduler.step(0.0);
        }
        
        // Should have reached close to max_lr
        assert!(max_lr_seen > 0.15);
        
        // Final LR should be min_lr
        let lr_final = scheduler.get_lr(0.0);
        assert_relative_eq!(lr_final, 0.00001, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn phase_boundaries() {
        let mut scheduler = OneCycleLR::new(1.0, 10, 0.4, AnnealStrategy::Linear, 10.0, 100.0, false);
        
        // 4 warmup steps, 6 annealing steps
        let mut phase_lrs = Vec::new();
        
        for step in 0..=10 {
            let lr = scheduler.get_lr(0.0);
            phase_lrs.push((step, lr));
            if step < 10 {
                scheduler.step(0.0);
            }
        }
        
        // Check that we reach max_lr during warmup
        let max_warmup_lr = phase_lrs[0..=4].iter().map(|(_, lr)| *lr).fold(0.0f64, f64::max);
        assert!(max_warmup_lr > 0.9); // Should be close to 1.0
        
        // Check that final LR is minimal
        let final_lr = phase_lrs[10].1;
        assert_relative_eq!(final_lr, 0.001, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn cosine_vs_linear_annealing() {
        let mut cos_scheduler = OneCycleLR::new(1.0, 8, 0.5, AnnealStrategy::Cos, 10.0, 100.0, false);
        let mut lin_scheduler = OneCycleLR::new(1.0, 8, 0.5, AnnealStrategy::Linear, 10.0, 100.0, false);
        
        let mut cos_lrs = Vec::new();
        let mut lin_lrs = Vec::new();
        
        for _ in 0..8 {
            cos_lrs.push(cos_scheduler.get_lr(0.0));
            lin_lrs.push(lin_scheduler.get_lr(0.0));
            cos_scheduler.step(0.0);
            lin_scheduler.step(0.0);
        }
        
        // Both should start at similar points
        assert_relative_eq!(cos_lrs[0], lin_lrs[0], epsilon = 1e-10, max_relative = 1e-10);
        
        // But should differ in the middle due to different interpolation
        assert!((cos_lrs[3] - lin_lrs[3]).abs() > 1e-6);
        
        // The final values might differ due to the different interpolation methods
        // but both should be approaching the min_lr
        assert!(cos_lrs[7] < 0.5 && lin_lrs[7] < 0.5);
    }

    #[test]
    fn edge_cases() {
        // Very short cycle
        let mut scheduler = OneCycleLR::new(1.0, 2, 0.5, AnnealStrategy::Linear, 10.0, 100.0, false);
        
        let lr1 = scheduler.get_lr(0.0);
        scheduler.step(0.0);
        let lr2 = scheduler.get_lr(0.0);
        scheduler.step(0.0);
        let lr3 = scheduler.get_lr(0.0);
        
        assert_relative_eq!(lr1, 0.1, epsilon = 1e-10, max_relative = 1e-10);
        assert!(lr2 > lr1); // Should increase during warmup
        assert_relative_eq!(lr3, 0.001, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn zero_warmup() {
        let mut scheduler = OneCycleLR::new(1.0, 10, 0.0, AnnealStrategy::Linear, 10.0, 100.0, false);
        
        // With pct_start = 0.0, should start at max_lr
        let lr_start = scheduler.get_lr(0.0);
        assert_relative_eq!(lr_start, 1.0, epsilon = 1e-10, max_relative = 1e-10);
        
        scheduler.step(0.0);
        let lr_next = scheduler.get_lr(0.0);
        assert!(lr_next < lr_start); // Should immediately start decreasing
    }

    #[test]
    fn beyond_total_steps() {
        let mut scheduler = OneCycleLR::new(1.0, 5, 0.4, AnnealStrategy::Linear, 10.0, 100.0, false);
        
        // Move beyond total_steps
        for _ in 0..10 {
            scheduler.step(0.0);
        }
        
        // Should return min_lr
        let lr = scheduler.get_lr(0.0);
        assert_relative_eq!(lr, 0.001, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn different_factors() {
        let scheduler1 = OneCycleLR::new(1.0, 10, 0.3, AnnealStrategy::Linear, 5.0, 50.0, false);
        let scheduler2 = OneCycleLR::new(1.0, 10, 0.3, AnnealStrategy::Linear, 20.0, 2000.0, false);
        
        // scheduler1: initial_lr = 1.0/5.0 = 0.2, min_lr = 0.2/50.0 = 0.004
        // scheduler2: initial_lr = 1.0/20.0 = 0.05, min_lr = 0.05/2000.0 = 0.000025
        
        let lr1_start = scheduler1.get_lr(0.0);
        let lr2_start = scheduler2.get_lr(0.0);
        
        assert_relative_eq!(lr1_start, 0.2, epsilon = 1e-10, max_relative = 1e-10);
        assert_relative_eq!(lr2_start, 0.05, epsilon = 1e-10, max_relative = 1e-10);
        
        assert!(lr1_start > lr2_start); // Different div_factors produce different initial LRs
    }
}