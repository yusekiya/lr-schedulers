# LR Schedulers

This crate provides learning rate schedulers for training of machine learning models.

Implemented schedulers:

* ConstantLR
* LineareLR
* ExponentialLR
* CosineAnnealingLR
* CosineAnnealingWarmRestarts

## Examples

```rust
use lr_schedulers::cosine_annealing_warm_restarts::CosineAnnealingWarmRestarts;
use lr_schedulers::Scheduler;

let eta_max = 1.0;
let eta_min = 0.1;
let t_0 = 2;
let t_mult = 2;
let init_step = 0;
let mut scheduler = CosineAnnealingWarmRestarts::new(
    eta_max, eta_min, t_0, t_mult, init_step
);
for epoch in 0 .. 100 {
    let lr = scheduler.get_lr();
    // Train a model with a larning rate `lr` and calculate loss.
    let loss = ...
    scheduler.step(loss);
}
```

## Usage

```bash
cargo add --git https://github.com/yusekiya/lr-schedulers lr-schedulers
```