# Build Instructions

**Document Contents**: How to build the lr-schedulers library and use it in other projects. Describes build commands using Cargo and dependency configuration methods.

**Document Purpose**: To enable developers to properly build the library and use it in other projects.

## Build Procedure

1. `cargo build`: Debug build
2. `cargo build --release`: Release build (with optimizations)

## Usage as a Library

Usage in other projects:

### Direct Usage from GitHub

```toml
[dependencies]
lr-schedulers = { git = "https://github.com/yusekiya/lr-schedulers" }
```

### Specifying a Version

```toml
[dependencies]
lr-schedulers = { git = "https://github.com/yusekiya/lr-schedulers", version = "0.2.0" }
```

