# CLAUDE.md

This file provides guidance for Claude Code (claude.ai/code) when working on this repository.

**IMPORTANT**: Update relevant files whenever new implementations or important decisions are made.

## Project Overview

lr-schedulers is a Rust library that provides learning rate schedulers for training machine learning models. This is a pure Rust implementation with no external runtime dependencies.

### Implemented Schedulers

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

### Core Architecture

All schedulers implement the `Scheduler` trait:
```rust
pub trait Scheduler {
    fn step(&mut self, loss: f64);
    fn get_lr(&self, loss: f64) -> f64;
}
```

Note: For more detailed architecture information including library structure and implementation patterns, see @docs/architecture.md

## Quick Development Commands

```bash
# Build & Test
cargo build              # Debug build
cargo build --release    # Release build
cargo test              # Run all tests
cargo test [module]     # Test specific module

# Code Quality
cargo fmt               # Format code (4-space indentation)
cargo clippy            # Lint checks (must resolve all warnings)
cargo doc --open        # Generate and view documentation

# Development
cargo test -- --nocapture  # Show test output
```

Note: Additional commonly used commands (including `cargo test --release` and `cargo bench`) are documented in @docs/common-commands.md

## Common Development Commands

- @docs/common-commands.md

## Project Architecture

- @docs/architecture.md

## Project Requirements

- @docs/project-requirements.md

## Code Style Guidelines

- @docs/code-style.md

## Testing Instructions

- @docs/testing-instructions.md

## Build Instructions

- @docs/building-instructions.md

## Git Guidelines

- @docs/git-guidelines.md

## Developer Environment Setup

- @docs/developer-environment-setup.md

## Technical Knowledge

- @docs/project-knowledge.md

## Improvement History

- @docs/project-improvements.md

