# Focused MEow regression tests

From the repository root, in a separate test environment with Python 3.12:

```bash
python -m pip install numpy==2.3.5
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The tests exercise the actual toy training loop, CleanRL MEow policy and training
function, and SKRL MEow initialization and model parameter-copying methods.
Environment, replay-buffer, plotting and logging integrations are replaced with
small test doubles; policy math, gradient updates and optimizers use real PyTorch.
No GPU, MuJoCo, Isaac Sim, Ray cluster, network service or full training run is needed.

Covered regressions:

- Plotting cannot overwrite an improved evaluated `best.pt` with a worse policy,
  or create a best checkpoint before evaluation; the last checkpoint still saves.
- Value plotting respects the policy device instead of requiring CUDA.
- Automatic entropy tuning reaches both the online and target policies at startup
  and after optimizer updates; fixed-temperature behavior is unchanged.
- Q and V use the current temperature.
- The SKRL target receives online weights at initialization, stays frozen, and
  does not share the online model's parameter storage.

These are focused CPU checks, not a reproduction of the paper's experiments or
an end-to-end check of simulator, Docker, plotting or legacy dependency support.
The test environment is separate from the frozen experiment environments.
