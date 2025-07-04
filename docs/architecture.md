# Architecture

**Document Contents:** Description of the software architecture for this project
**Document Purpose:** To ensure consistency in development methods and enable sustainable development

## Key Implementation Patterns

- All schedulers implement the `Scheduler` trait
- Each scheduler maintains internal state (step counter, current lr)
- Most schedulers ignore the `loss` parameter (only `ReduceLROnPlateau` uses it)
- `get_lr()` is idempotent until `step()` is called
- All schedulers support `init_step` for training resumption

## Core Interface

```rust
pub trait Scheduler {
    fn step(&mut self, loss: f64);
    fn get_lr(&self, loss: f64) -> f64;
}
```

## Directory Structure

### Library Structure (`src/`)

- **lib.rs:** Library root file, defines public API and Scheduler trait
- **constant.rs:** Constant learning rate scheduler implementation
- **linear.rs:** Linear decay scheduler implementation
- **exponential.rs:** Exponential decay scheduler implementation
- **polynomial.rs:** Polynomial decay scheduler implementation
- **step.rs:** Step decay scheduler implementation
- **multistep.rs:** Multi-step scheduler implementation
- **cosine_annealing.rs:** Cosine annealing scheduler implementation
- **cosine_annealing_warm_restarts.rs:** Cosine annealing with warm restarts scheduler implementation
- **cyclic.rs:** Cyclic scheduler implementation
- **multiplicative.rs:** Multiplicative scheduler implementation
- **reduce_lr_on_plateau.rs:** Reduce learning rate on plateau scheduler implementation
- **onecycle.rs:** One cycle scheduler implementation

### Documentation (`docs/`)

- **architecture.md:** Library structure and core interface
- **common-commands.md:** Extended list of development commands
- **code-style.md:** Detailed style guidelines
- **testing-instructions.md:** Comprehensive testing guide
- **building-instructions.md:** Build and dependency descriptions
- **git-guidelines.md:** Git workflow and commit conventions
- **project-requirements.md:** Core project requirements
- **common-implementation-approach.md:** Common implementation procedures
- **developer-environment-setup.md:** Development environment setup instructions
- **project-knowledge.md:** Technical insights
- **project-improvements.md:** Improvement history
