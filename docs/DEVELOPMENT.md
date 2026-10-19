# Developing the ExpertParkour extension

## Ownership boundary

The upstream baseline is commit `d6766d8883cffe427f2d4c087717d4eca1f23d8c`
from [Isaaclab_Parkour](https://github.com/CAI23sbP/Isaaclab_Parkour).
ExpertParkour's added modules and launchers can be identified with:

```bash
git diff --diff-filter=A --name-only d6766d8883cffe427f2d4c087717d4eca1f23d8c HEAD -- '*.py' '*.sh'
```

The cleanup targets those added files. Existing upstream files, including our
earlier integration patches inside them, are excluded from broad formatting.
The explicit include list in `ruff.toml` enforces this boundary when running
the formatter or linter from the repository root. License notices remain intact.

## Contracts worth preserving

- `student_observations.py` names the inherited heading, terrain-class, and
  contact slices, plus the five-control-step perception interval. Changing an
  offset requires checking the upstream observation builder.
- The default context and router refresh at 10 Hz; experts consume fresh
  proprioception at 50 Hz. The control-rate variant updates only the live token
  between new visual frames. Reset and graph detachment have different purposes.
- Router and experts share an input layout. PPO replay must use the observation
  that originally refreshed the cached gates, rather than the latest expert
  input. Router noise is disabled during refinement.
- GAE receives rewards that already contain timeout bootstrap credit. Done flags
  still stop recursion across episode boundaries. Logged reward excludes that
  synthetic credit.
- Keep module/parameter names and checkpoint dictionary keys stable. Released
  checkpoints contain state dictionaries; moving the critic helpers does not
  change the `net.*` state layout.
- Start `AppLauncher` before simulator-dependent imports in CLI entry points.
  Do not sort imports across that startup boundary. Offline tools should remain
  usable without launching Isaac Sim.

Document public APIs and explain shapes, state ownership, and non-obvious
decisions in comments. Experimental measurements and historical diagnoses belong
in `CONCLUSIONS.md`; comments should state the contract needed to maintain code.

## Checks

Activate the recorded Python environment with NumPy and PyTorch installed. The
tests use standard-library `unittest`; they do not require pytest, Isaac Sim,
an available GPU, or a training run.

```bash
python -m unittest discover -s tests -v

# Install these development tools separately from the simulation dependencies.
ruff check .
ruff format --check .
shellcheck scripts/*.sh scripts/lib/*.sh
for script in scripts/*.sh scripts/lib/*.sh; do
    bash -n "$script"
done
git diff --check
```

The suite covers fused expert equivalence, routing caches and replay, causal
attention, temporal buffers, trajectory returns, critic transfer, spike recovery,
checkpoint selection, censored exposure, damaged archives, log parsing, and
launcher overrides. Shell tests source only the pure helpers or environment
defaults; they never launch or terminate experiments.

The inherited `scripts/__init__.py` and module initializers import simulator
dependencies eagerly. The CPU test loader bypasses those initializers and loads
the actual source modules through a separate namespace. It does not mock the
model or change the upstream import behavior.

## Released-policy regression fixture

`tests/fixtures/released_policy_outputs.npz` was captured from the **unmodified
code at commit `248275e`**, before cleanup, using the committed top-2 checkpoint:

```text
logs/rsl_rl/unitree_go2_parkour_moe/2026-08-12_12-16-19_moe_transformer/slim/model_14999.pt
SHA-256: 90f35eb09225070c2ae3fe9113cea1552028c065607bf1e572a37c0b33df0ae5
```

`tests/_support.py::released_rollout` defines deterministic NumPy inputs for
eight control steps over two environments. It includes perception refreshes and
separate episode resets between frames. The fixture stores actions, context,
and gate weights; the test loads released parameters strictly and compares all
three with `rtol=atol=1e-5` to allow CPU numerical variation. That test is skipped
when the released checkpoint is absent. Do not regenerate the expected values
from a changed implementation merely to make a failing test pass.

## Refinement scope

The PPO entry point supports Transformer/MoE students with frozen perception.
`--unfreeze_encoder` is rejected because vector-only replay cannot propagate
gradients through depth/token history. GRU refinement is also rejected before
starting the simulator; its registered task and historical launcher commands do
not constitute an implemented replay path.

This cleanup preserves the default losses and published inference behavior.
It does not establish that the training loop, optional variants, GPU rendering,
or multi-GPU execution have been revalidated by a fresh full training run.
