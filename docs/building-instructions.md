# Build Instructions

## Steps

1. cargo build: Debug build
2. cargo build --release: Release build (with optimizations)

## Using as a Library

Usage in other projects:

### Direct from GitHub

```toml
[dependencies]
lr-schedulers = { git = "https://github.com/yusekiya/lr-schedulers" }
```

### Specify a Version

```toml
[dependencies]
lr-schedulers = { git = "https://github.com/yusekiya/lr-schedulers", version = "0.2.0" }
```
