# Code Style Guidelines

**Document Contents**: Coding conventions for the lr-schedulers project. Defines rules for indentation, formatting, error handling, documentation creation, etc.
**Document Purpose**: To maintain a consistent codebase and improve readability and maintainability

## Formatting Rules

- **Rust code**: 4-space indentation
- **Markdown**: 2-space indentation
- Apply `cargo fmt` on save
- Resolve all `cargo clippy` warnings

## Coding Conventions

- Always include documentation comments for public APIs
- Include test code (unit tests, integration tests)
- Handle errors appropriately (avoid panics)
- Use `std::f64::consts` for mathematical constants and functions
