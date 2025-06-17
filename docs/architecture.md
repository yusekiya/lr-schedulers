# Project Architecture

## Library Structure (`src/`)

- **lib.rs:** Library root file, defines public API and Scheduler trait
- **constant.rs:** Fixed learning rate scheduler implementation
- **linear.rs:** Linear decay scheduler implementation
- **exponential.rs:** Exponential decay scheduler implementation
- **cosine_annealing.rs:** Cosine annealing scheduler implementation
- **cosine_annealing_warm_restarts.rs:** Cosine annealing with warm restarts scheduler implementation

## Core Interface

- `Scheduler` trait: Common interface implemented by all schedulers
  - `get_lr(&self, loss: f64) -> f64`: Get current learning rate
  - `step(&mut self, loss: f64)`: Update scheduler state
