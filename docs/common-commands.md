# Common Development Commands

**Document Contents**: Reference for Rust development commands frequently used in the lr-schedulers project. Includes basic commands for building, testing, formatting, and documentation generation

**Document Purpose**: To enable developers to quickly reference commands and work efficiently

## Rust Development Commands

### Build

- `cargo build`: Debug build
- `cargo build --release`: Release build (with optimizations)

### Test

- `cargo test`: Run all tests
- `cargo test --release`: Run tests in release mode
- `cargo test [module_name]`: Run tests for a specific module
- `cargo test -- --nocapture`: Display standard output during test execution
- `cargo bench`: Run benchmarks

### Code Quality

- `cargo fmt`: Format code
- `cargo clippy`: Run lint checks

### Documentation

- `cargo doc --open`: Generate and display documentation
