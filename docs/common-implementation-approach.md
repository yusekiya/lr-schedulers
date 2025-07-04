# Common Implementation Procedures

**Document Contents:** Frequently used implementation procedures

**Document Purpose:** To maintain code consistency

## Adding a New Scheduler

1. Create a new module file in `src/` (e.g., `src/my_scheduler.rs`)
2. Implement the `Scheduler` trait with appropriate state management
3. Add comprehensive unit tests in the same file
4. Use `approx::assert_relative_eq!` for floating-point comparisons in tests
5. Add documentation including usage examples
6. Export the scheduler in `src/lib.rs`
7. Update the scheduler list in this file
8. Update README.md if it is still outdated
