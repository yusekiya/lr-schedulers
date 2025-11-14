# LR Schedulers

This crate provides learning rate schedulers for training of machine learning models.

Implemented schedulers:

* ConstantLR
* CosineAnnealingLR
* CosineAnnealingWarmRestarts
* CyclicLR
* ExponentialLR
* LinearLR
* MultiplicativeLR
* MultiStepLR
* OneCycleLR
* PolynomialLR
* ReduceLROnPlateau
* StepLR

## Examples

The following pseudo code shows the usage of this scheduler.

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
    // Get the learning rate for this epoch.
    let lr = scheduler.get_lr();
    // Calculate loss tensor.
    let loss = (y - labels).sqr().mean_all();
    let loss_scalar = loss.to_f64();
    // Train a model with the learning rate `lr`.
    let optimizer = Optimizer::new(lr);
    optimizer.backward_step(&loss);
    // Then update the scheduler for the next iteration.
    scheduler.step(loss_scalar);
}
```

## Usage

```bash
cargo add --git https://github.com/yusekiya/lr-schedulers lr-schedulers
```