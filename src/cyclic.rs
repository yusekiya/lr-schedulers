//! Cyclically varies learning rate between two boundary values.
//!
//! This scheduler implements the cyclical learning rate policy where learning rate
//! cycles between base_lr and max_lr according to different policies.
//!
//! # Examples
//!
//! Basic triangular mode:
//!
//! ```
//! # use lr_schedulers::cyclic::{CyclicLR, Mode};
//! # use lr_schedulers::Scheduler;
//! let mut scheduler = CyclicLR::new(0.01, 0.1, 4, None, Mode::Triangular, 1.0, None);
//! let mut learning_rates = Vec::new();
//! for _ in 0 .. 12 {
//!     learning_rates.push(scheduler.get_lr(0.0));
//!     scheduler.step(0.0);
//! }
//! // Forms triangular pattern: base -> max -> base -> max -> base
//! ```
//!
//! Triangular2 mode with decreasing amplitude:
//!
//! ```
//! # use lr_schedulers::cyclic::{CyclicLR, Mode};
//! # use lr_schedulers::Scheduler;
//! let mut scheduler = CyclicLR::new(0.01, 0.1, 2, None, Mode::Triangular2, 1.0, None);
//! let mut learning_rates = Vec::new();
//! for _ in 0 .. 8 {
//!     learning_rates.push(scheduler.get_lr(0.0));
//!     scheduler.step(0.0);
//! }
//! // Each cycle has half the amplitude of the previous cycle
//! ```
//!
//! Exponential range mode:
//!
//! ```
//! # use lr_schedulers::cyclic::{CyclicLR, Mode};
//! # use lr_schedulers::Scheduler;
//! let mut scheduler = CyclicLR::new(0.01, 0.1, 3, None, Mode::ExpRange, 0.99, None);
//! let mut learning_rates = Vec::new();
//! for _ in 0 .. 9 {
//!     learning_rates.push(scheduler.get_lr(0.0));
//!     scheduler.step(0.0);
//! }
//! // Amplitude decreases exponentially with gamma
//! ```
//!
//! Custom scale function:
//!
//! ```
//! # use lr_schedulers::cyclic::{CyclicLR, Mode, ScaleMode};
//! # use lr_schedulers::Scheduler;
//! let custom_fn = |x: f64| 0.5 * (1.0 + (PI * x).cos());
//! let mut scheduler = CyclicLR::new(0.01, 0.1, 4, None, Mode::Triangular, 1.0, Some((custom_fn, ScaleMode::Cycle)));
//! let mut learning_rates = Vec::new();
//! for _ in 0 .. 8 {
//!     learning_rates.push(scheduler.get_lr(0.0));
//!     scheduler.step(0.0);
//! }
//! // Uses cosine-based custom scaling
//! ```

use crate::Scheduler;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Triangular,
    Triangular2,
    ExpRange,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleMode {
    Cycle,
    Iterations,
}

#[derive(Debug, Clone)]
pub struct CyclicLR<F>
where
    F: Fn(f64) -> f64,
{
    base_lr: f64,
    max_lr: f64,
    step_size_up: usize,
    step_size_down: usize,
    mode: Mode,
    gamma: f64,
    scale_fn: Option<(F, ScaleMode)>,
    step_count: usize,
}

impl CyclicLR<fn(f64) -> f64> {
    /// Constructs a CyclicLR instance without custom scale function.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Lower boundary of the cycle (minimum learning rate)
    /// * `max_lr` - Upper boundary of the cycle (maximum learning rate)
    /// * `step_size_up` - Number of training iterations in the increasing half of a cycle
    /// * `step_size_down` - Number of training iterations in the decreasing half of a cycle (defaults to step_size_up)
    /// * `mode` - Cycling mode: Triangular, Triangular2, or ExpRange
    /// * `gamma` - Constant for exponential range scaling (used only in ExpRange mode)
    ///
    pub fn new(
        base_lr: f64,
        max_lr: f64,
        step_size_up: usize,
        step_size_down: Option<usize>,
        mode: Mode,
        gamma: f64,
        _scale_fn: Option<()>,
    ) -> Self {
        let step_size_down = step_size_down.unwrap_or(step_size_up);

        CyclicLR {
            base_lr,
            max_lr,
            step_size_up,
            step_size_down,
            mode,
            gamma,
            scale_fn: None,
            step_count: 0,
        }
    }
}

impl<F> CyclicLR<F>
where
    F: Fn(f64) -> f64,
{
    /// Constructs a CyclicLR instance with custom scale function.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Lower boundary of the cycle (minimum learning rate)
    /// * `max_lr` - Upper boundary of the cycle (maximum learning rate)
    /// * `step_size_up` - Number of training iterations in the increasing half of a cycle
    /// * `step_size_down` - Number of training iterations in the decreasing half of a cycle (defaults to step_size_up)
    /// * `mode` - Cycling mode (ignored when scale_fn is provided)
    /// * `gamma` - Constant for exponential range scaling (ignored when scale_fn is provided)
    /// * `scale_fn` - Custom scaling function and mode
    ///
    pub fn with_scale_fn(
        base_lr: f64,
        max_lr: f64,
        step_size_up: usize,
        step_size_down: Option<usize>,
        mode: Mode,
        gamma: f64,
        scale_fn: Option<(F, ScaleMode)>,
    ) -> Self {
        let step_size_down = step_size_down.unwrap_or(step_size_up);

        CyclicLR {
            base_lr,
            max_lr,
            step_size_up,
            step_size_down,
            mode,
            gamma,
            scale_fn,
            step_count: 0,
        }
    }

    /// Calculate the current cycle number and position within cycle
    fn get_cycle_and_position(&self) -> (usize, f64) {
        let cycle_length = self.step_size_up + self.step_size_down;
        let cycle = 1 + self.step_count / cycle_length;

        let x = if self.step_count % cycle_length <= self.step_size_up {
            // Increasing phase
            (self.step_count % cycle_length) as f64 / self.step_size_up as f64
        } else {
            // Decreasing phase
            let steps_in_down = (self.step_count % cycle_length) - self.step_size_up;
            1.0 - (steps_in_down as f64 / self.step_size_down as f64)
        };

        (cycle, x)
    }

    /// Calculate the scale factor based on mode and cycle position
    fn get_scale_factor(&self, cycle: usize, _x: f64) -> f64 {
        if let Some((ref scale_fn, scale_mode)) = self.scale_fn {
            match scale_mode {
                ScaleMode::Cycle => scale_fn(cycle as f64),
                ScaleMode::Iterations => {
                    let cycle_iterations =
                        self.step_count % (self.step_size_up + self.step_size_down);
                    scale_fn(cycle_iterations as f64)
                }
            }
        } else {
            match self.mode {
                Mode::Triangular => 1.0,
                Mode::Triangular2 => 1.0 / (2.0_f64.powi((cycle - 1) as i32)),
                Mode::ExpRange => self.gamma.powi(self.step_count as i32),
            }
        }
    }
}

impl<F> Scheduler for CyclicLR<F>
where
    F: Fn(f64) -> f64,
{
    fn step(&mut self, _loss: f64) {
        self.step_count += 1;
    }

    fn get_lr(&self, _loss: f64) -> f64 {
        let (cycle, x) = self.get_cycle_and_position();
        let scale_factor = self.get_scale_factor(cycle, x);
        let amplitude = (self.max_lr - self.base_lr) * scale_factor;

        self.base_lr + amplitude * x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scheduler;
    use approx::assert_relative_eq;

    #[test]
    fn triangular_mode_basic() {
        let mut scheduler = CyclicLR::new(0.1, 0.9, 4, None, Mode::Triangular, 1.0, None);

        // First cycle: 0 -> 1 -> 0 over 8 steps
        let expected_positions = [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0];

        for (_i, &expected_x) in expected_positions.iter().enumerate() {
            let lr = scheduler.get_lr(0.0);
            let expected_lr = 0.1 + (0.9 - 0.1) * expected_x;
            assert_relative_eq!(lr, expected_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn triangular2_mode() {
        let mut scheduler = CyclicLR::new(0.1, 0.5, 2, None, Mode::Triangular2, 1.0, None);

        // First cycle: amplitude = 0.4, second cycle: amplitude = 0.2
        let test_cases = [
            (0, 0.1), // Start of first cycle
            (1, 0.3), // Mid of first cycle (0.1 + 0.4 * 0.5)
            (2, 0.5), // Peak of first cycle
            (3, 0.3), // End of first cycle
            (4, 0.1), // Start of second cycle
            (5, 0.2), // Mid of second cycle (0.1 + 0.2 * 0.5)
            (6, 0.3), // Peak of second cycle (0.1 + 0.2 * 1.0)
            (7, 0.2), // End of second cycle
        ];

        for &(step, expected_lr) in test_cases.iter() {
            while scheduler.step_count < step {
                scheduler.step(0.0);
            }
            let lr = scheduler.get_lr(0.0);
            assert_relative_eq!(lr, expected_lr, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn exp_range_mode() {
        let mut scheduler = CyclicLR::new(0.1, 0.5, 2, None, Mode::ExpRange, 0.9, None);

        // Test first few steps
        for step in 0..4 {
            let lr = scheduler.get_lr(0.0);
            let (_, x) = scheduler.get_cycle_and_position();
            let expected_scale = 0.9_f64.powi(step);
            let expected_lr = 0.1 + (0.5 - 0.1) * expected_scale * x;
            assert_relative_eq!(lr, expected_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }

    #[test]
    fn custom_scale_function() {
        let cosine_fn = |x: f64| 0.5 * (1.0 + (PI * x).cos());
        let scheduler = CyclicLR::with_scale_fn(
            0.1,
            0.5,
            2,
            None,
            Mode::Triangular,
            1.0,
            Some((cosine_fn, ScaleMode::Cycle)),
        );

        // First cycle should use cosine_fn(1.0)
        let expected_scale = 0.5 * (1.0 + (PI * 1.0).cos()); // ≈ 0.0

        let lr = scheduler.get_lr(0.0);
        let (_, x) = scheduler.get_cycle_and_position();
        let expected_lr = 0.1 + (0.5 - 0.1) * expected_scale * x;
        assert_relative_eq!(lr, expected_lr, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn asymmetric_step_sizes() {
        let mut scheduler = CyclicLR::new(0.1, 0.9, 2, Some(6), Mode::Triangular, 1.0, None);

        // Cycle length = 2 + 6 = 8
        // Steps 0-1: increasing (2 steps)
        // Steps 2-7: decreasing (6 steps)

        let test_cases = [
            (0, 0.0),       // Start
            (1, 0.5),       // Mid of increasing phase
            (2, 1.0),       // Peak
            (3, 5.0 / 6.0), // Start of decreasing phase
            (4, 4.0 / 6.0), // Mid of decreasing phase
            (7, 1.0 / 6.0), // Near end of decreasing phase
        ];

        for &(step, expected_x) in test_cases.iter() {
            while scheduler.step_count < step {
                scheduler.step(0.0);
            }
            let lr = scheduler.get_lr(0.0);
            let expected_lr = 0.1 + (0.9 - 0.1) * expected_x;
            assert_relative_eq!(lr, expected_lr, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn cycle_boundaries() {
        let mut scheduler = CyclicLR::new(0.2, 0.8, 3, None, Mode::Triangular, 1.0, None);

        // Test that we correctly handle cycle boundaries
        for _cycle in 0..2 {
            // Start of cycle should be at base_lr
            let lr = scheduler.get_lr(0.0);
            assert_relative_eq!(lr, 0.2, epsilon = 1e-10, max_relative = 1e-10);

            // Move through the cycle
            for _ in 0..6 {
                scheduler.step(0.0);
            }
        }
    }

    #[test]
    fn minimum_maximum_values() {
        let mut scheduler = CyclicLR::new(0.01, 0.1, 5, None, Mode::Triangular, 1.0, None);

        let mut min_lr = f64::INFINITY;
        let mut max_lr = f64::NEG_INFINITY;

        // Run for multiple cycles
        for _ in 0..20 {
            let lr = scheduler.get_lr(0.0);
            min_lr = min_lr.min(lr);
            max_lr = max_lr.max(lr);
            scheduler.step(0.0);
        }

        // Should reach base_lr and max_lr
        assert_relative_eq!(min_lr, 0.01, epsilon = 1e-10, max_relative = 1e-10);
        assert_relative_eq!(max_lr, 0.1, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn scale_mode_iterations() {
        let linear_fn = |x: f64| x * 0.1; // Linear scaling
        let mut scheduler = CyclicLR::with_scale_fn(
            0.1,
            0.5,
            2,
            None,
            Mode::Triangular,
            1.0,
            Some((linear_fn, ScaleMode::Iterations)),
        );

        // Test that scale function is called with iteration count
        for step in 0..4 {
            let lr = scheduler.get_lr(0.0);
            let (_, x) = scheduler.get_cycle_and_position();
            let cycle_iterations = step % 4; // cycle length = 4
            let expected_scale = (cycle_iterations as f64) * 0.1;
            let expected_lr = 0.1 + (0.5 - 0.1) * expected_scale * x;
            assert_relative_eq!(lr, expected_lr, epsilon = 1e-10, max_relative = 1e-10);
            scheduler.step(0.0);
        }
    }
}

