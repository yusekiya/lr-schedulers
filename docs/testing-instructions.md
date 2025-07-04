# Testing Instructions

**Document Contents:** Testing policies and implementation procedures

**Document Purpose:** To share methods for maintaining software quality

## Basic Principles

- Implement tests for all functions provided externally
- Ensure all tests pass before creating Git commits

## Test Commands

- `cargo test`: Run all tests
- `cargo test --lib`: Run library tests only
- `cargo test --doc`: Run documentation tests only
- `cargo test [module_name]`: Run tests for a specific module
- `cargo test -- --nocapture`: Display standard output during test execution
- `cargo test --release`: Run tests in release mode

## Testing Best Practices

- Test edge cases (zero steps, negative values, boundary conditions)
- Test initialization with different `init_step` values
- Use dummy loss values (0.0) for schedulers that don't use loss
- Include testable examples in doc comments
- Use `approx::assert_relative_eq!` for floating-point comparisons in tests
