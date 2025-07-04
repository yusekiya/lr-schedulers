# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: Update relevant documentation files in `docs/` whenever new implementations or important decisions are made.

## Project Overview

lr-schedulers is a Rust library that provides learning rate schedulers for training machine learning models. This is a pure Rust implementation with no external runtime dependencies.

### Implemented Schedulers (13 total)

- **ConstantLR**: Maintains constant learning rate for specified iterations
- **LinearLR**: Linear decay/growth between start and end values
- **ExponentialLR**: Exponential decay with configurable gamma
- **PolynomialLR**: Polynomial decay of learning rate
- **StepLR**: Step-wise reduction at regular intervals
- **MultiStepLR**: Reduction at specified milestone steps
- **CosineAnnealingLR**: Cosine annealing between min and max values
- **CosineAnnealingWarmRestarts**: Cosine annealing with periodic restarts
- **CyclicLR**: Cycles between minimum and maximum learning rates
- **MultiplicativeLR**: Multiplies learning rate by a custom factor function
- **ReduceLROnPlateau**: Reduces learning rate when metric stops improving
- **OneCycleLR**: One cycle learning rate policy
- **[Any new schedulers should be added here]**

### Core Architecture

All schedulers implement the `Scheduler` trait:

```rust
pub trait Scheduler {
    fn step(&mut self, loss: f64);
    fn get_lr(&self, loss: f64) -> f64;
}
```

Key implementation patterns:

- Each scheduler maintains internal state (step counter, current lr)
- Most schedulers ignore the `loss` parameter (only `ReduceLROnPlateau` uses it)
- `get_lr()` is idempotent until `step()` is called

## Quick Development Commands

```bash
# Build & Test
cargo build              # Debug build
cargo build --release    # Release build
cargo test              # Run all tests
cargo test [module]     # Test specific module (e.g., cargo test linear)

# Code Quality
cargo fmt               # Format code (4-space indentation)
cargo clippy            # Lint checks (must resolve all warnings)
cargo doc --open        # Generate and view documentation

# Development
cargo test -- --nocapture  # Show test output
cargo test --release      # Test in release mode
cargo bench              # Run benchmarks
```

## When Adding a New Scheduler

1. Create a new module file in `src/` (e.g., `src/my_scheduler.rs`)
2. Implement the `Scheduler` trait with proper state management
3. Add comprehensive unit tests in the same file
4. Use `approx::assert_relative_eq!` for floating-point comparisons in tests
5. Add documentation with usage examples
6. Export the scheduler in `src/lib.rs`
7. Update this file's scheduler list
8. Update README.md if it's still outdated

## Testing Best Practices

- Test edge cases (zero steps, negative values, boundary conditions)
- Test initialization with different `init_step` values
- Use dummy loss value (0.0) for schedulers that don't use loss
- Include examples in doc comments that can be tested

## Project Documentation

The `docs/` directory contains detailed documentation:

- `architecture.md` - Library structure and core interface
- `common-commands.md` - Extended list of development commands
- `code-style.md` - Detailed style guidelines
- `testing-instructions.md` - Comprehensive testing guide
- `building-instructions.md` - Build and dependency instructions
- `git-guidelines.md` - Git workflow and commit conventions
- `project-requirements.md` - Core project requirements
- Other documentation files for environment setup and project knowledge

