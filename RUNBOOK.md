# Runbook

Every command in this project, what it does, and what it leaves behind.

All paths are relative to the repo root. Activate the Isaac Lab environment
first. The wrappers source `scripts/launch_env.sh`, which selects the active
Python interpreter, supplies repository-relative defaults, and accepts overrides:

```bash
export CUDA_VISIBLE_DEVICES=0      # an empty value makes Isaac Sim fall back to CPU and abort
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHON_BIN="$(command -v python)"
export TEACHER_CKPT=/absolute/path/to/model_49999.pt
source scripts/launch_env.sh
```

The wrapper scripts (`run_experiments.sh`, `make_video.sh`, `run_analysis.sh`)
source it themselves. If you call a Python script directly, `source` it first.
The launcher preserves an existing `LD_LIBRARY_PATH`, prepends optional WSL/CUDA
directories only when present, and honors a nonempty `CUDA_VISIBLE_DEVICES`.
`CUDA_VISIBLE_DEVICES_OVERRIDE` takes precedence when set. Checkpoint selection
and iteration counting are shared in `scripts/lib/experiment_helpers.sh`.

For source checks and simulator-free regression tests, see
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

---

## 1. Training

### Both students, back to back

```bash
./scripts/run_experiments.sh
```

Trains the MoE student, then the GRU baseline, 15 000 iterations each at 192
environments. They are chained deliberately: running them concurrently was
measured at 31.9 / 32.6 GB peak VRAM and ~6x slower per iteration.

Overridable by environment variable:

```bash
ITERS=5000 NUM_ENVS=128 SEED=2 ./scripts/run_experiments.sh
```

Writes `logs/rsl_rl/unitree_go2_parkour_{moe,baseline}/<timestamp>_<name>/` and
console logs to `logs/run_console/{moe,baseline}.log`. Metrics stream to wandb
(project `parkour-moe-transformer`).

Run it detached so it survives the terminal:

```bash
setsid nohup ./scripts/run_experiments.sh > logs/run_experiments.out 2>&1 < /dev/null &
```

### One student on its own

```bash
source scripts/launch_env.sh
$PYTHON_BIN scripts/rsl_rl/train.py \
    --task Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-v0 \
    --seed 1 --headless --num_envs 192 --max_iterations 15000 \
    --teacher_checkpoint $TEACHER_CKPT
```

Swap `StudentMoE` for `Student` to train the baseline. `--teacher_checkpoint` is
required for both: distillation needs a frozen phase-1 teacher, and passing it
explicitly avoids the upstream lookup that searches the log directory.

### Watching a run

```bash
grep -c "Learning iteration" logs/run_console/moe.log      # iterations done
tail -30 logs/run_console/moe.log                          # latest block
```

The ETA the log prints wraps every 24 hours — it formats with `gmtime`, so
`03:16:18` after a day of training means 27 h 16 m.

---

## 2. Video

### From the newest checkpoint, one command

```bash
./scripts/make_video.sh
```

Records a rollout and renders a third-person video per environment plus the
egocentric depth view with the expert gates overlaid.

```bash
./scripts/make_video.sh -s 900 -n 8        # longer, more environments
./scripts/make_video.sh -e 2               # bypass the router, run expert 2 alone
./scripts/make_video.sh -c logs/.../model_15000.pt
./scripts/make_video.sh -h
```

Output lands in `<run>/playback_it<N>/`, so successive checkpoints accumulate
instead of overwriting. Safe against a live training job: it skips any checkpoint
written in the last 20 seconds, which would otherwise be a half-flushed
`torch.save`.

**There is no RTX rendering on this machine** — no NVIDIA Vulkan ICD under
`/usr/lib/wsl/lib`, so `gym.wrappers.RecordVideo` hangs. The third-person view is
reconstructed offline from link poses and the height scanner, which needs no
renderer. `play.py --video` still works on a host with a display.

### Re-render without re-simulating

```bash
python scripts/analysis/render_rollout.py                       # newest rollout
python scripts/analysis/render_rollout.py <run_dir> --env 1
python scripts/analysis/render_rollout.py <rollout_router.npz> --env 0 --elev 25 --azim -50
```

The `.npz` holds every link pose, so camera angle and styling are free to change.
`--env 0..3` picks the environment; the four sit on different terrains.

Reading the picture: the robot is flat white and carries no colour code — colour
belongs to the gate panel alone. A filled foot marker means that foot is in
contact. Terrain colour is height relative to the base on a fixed scale, so
yellow is an obstacle above the robot, blue is the ground, magenta is the floor
of a gap.

---

## 3. Analysis

### Everything, one command

```bash
./scripts/run_analysis.sh
./scripts/run_analysis.sh -s 1500              # longer rollouts, tighter error bars
./scripts/run_analysis.sh --skip-ablation      # skip the slowest stage
./scripts/run_analysis.sh -c logs/.../model_15000.pt
```

Five stages, all under `<checkpoint_dir>/analysis/`:

| stage | question | output |
|---|---|---|
| 1 | how well does it traverse each terrain? | `metrics_main.json` |
| 2 | what did the experts specialise in? | `routing.json` |
| 3 | does the Transformer use its 0.8 s window? | `attention.json` |
| 4 | is the hidden dynamics decodable from the context? | `probe_directions.npz` |
| 5 | does the actor actually *use* it? | `metrics_ablated.json` |

Nice'd and read-only, so it can run against a live training job — expect training
to slow while the simulator is up.

### The pieces, separately

**Performance per terrain.** MXD is waypoints reached at episode end / 7; MEV is
feet contacting within 5 cm of an edge, per step. Lower MEV is better, but only
comparable between policies that travel similar distances.

```bash
source scripts/launch_env.sh
$PYTHON_BIN scripts/rsl_rl/evaluate_students.py \
    --task Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Eval-v0 \
    --checkpoint <model_*.pt> --num_envs 192 --steps 900 --tag main --headless
```

**The ceiling.** No student can regress its way past the policy it imitates, so
the teacher's own score on the same environment is what makes the student's
number readable.

```bash
$PYTHON_BIN scripts/rsl_rl/evaluate_students.py \
    --task Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Eval-v0 \
    --checkpoint $TEACHER_CKPT --policy teacher \
    --num_envs 192 --steps 900 --tag teacher --out_dir logs/teacher_eval --headless
```

Use the same `--steps` as the student run: the metric is measured at episode end,
so a shorter rollout completes fewer, and differently biased, episodes.

**Expert specialisation and effective expert count.**

```bash
python scripts/analysis/analyze_student.py <trace_*.npz>
```

**Attention over the temporal window**, with plots:

```bash
python scripts/analysis/attention_probe.py                  # newest suitable trace
python scripts/analysis/plot_attention.py                   # light
python scripts/analysis/plot_attention.py --dark
```

The trace must have been recorded with `--attention`.

**Dynamics probe.** Must be run on the **training** task: the EVAL configs set the
mass and COM randomisation events to `None`, so those parameters are constant and
the probe has nothing to predict. The script warns when that happens.

```bash
$PYTHON_BIN scripts/rsl_rl/evaluate_students.py \
    --task Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-v0 \
    --checkpoint <model_*.pt> --num_envs 192 --steps 600 --context_every 3 \
    --tag probe --headless

python scripts/analysis/fit_context_probe.py <trace_probe.npz>
```

Splits by environment, not by timestep: randomisation runs at startup, so each
environment holds one constant parameter vector for its whole life and a
timestep split would put the same target in train and test. Every R² is reported
next to a shuffled-context control.

**Causal test — does the actor use the representation?** Projects the decodable
directions out of the context before the actor sees it.

```bash
$PYTHON_BIN scripts/rsl_rl/evaluate_students.py \
    --task Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Eval-v0 \
    --checkpoint <model_*.pt> \
    --ablate_context <probe_directions.npz> --ablate_dims mass,com_x,com_y \
    --num_envs 192 --steps 900 --tag ablated --headless
```

Compare its MXD against the unablated run. Storing the information and using it
are different claims; only this separates them.

**What can be read off the weights alone**, no simulator:

```bash
python scripts/analysis/inspect_moe_weights.py <model_*.pt>
```

Reports which input block drives the router, how distinct the experts' functions
are, and the per-joint output profile.

---

## 4. Checkpoints

A training checkpoint is 103 MB, of which 66.6 MB is Adam state needed only to
resume. GitHub rejects files over 100 MB, so publish the export:

```bash
python scripts/export_slim_checkpoint.py <run_dir> --every 5000
python scripts/export_slim_checkpoint.py <model_15000.pt>
```

Writes 36.5 MB copies to `<run>/slim/`, always keeping the final iteration even
when it is not a multiple of `--every`. `.gitignore` tracks `**/slim/*.pt` and
ignores the raw ones.

---

## 5. Housekeeping

**Orphaned simulators.** Isaac Sim can hang at shutdown and survive `timeout`,
holding GPU memory. Check and clear:

```bash
ps -eo pid,etime,cmd | grep -E "scripts/rsl_rl/(train|play|evaluate)" | grep -v grep
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
kill -9 <pid>
```

The wrapper scripts use `timeout -k`, which follows SIGTERM with SIGKILL; a plain
`timeout` sends only SIGTERM, which Isaac Sim ignores.

**Pulling metric series out of wandb** for offline comparison:

```python
import wandb
api = wandb.Api()
run = api.run("claudiocamolese-politecnico-di-torino/parkour-moe-transformer/<run_id>")
df = run.history(keys=["Loss_depth/depth_actor_loss",
                       "Metrics/base_parkour/terrain_levels"],
                 samples=40000, pandas=True)
```

---

## Reading the metrics

`Episode_Reward/*` is identically zero for the students and means nothing: they
are trained by supervised distillation, the reward is discarded in the rollout
loop, and the single remaining reward term is configured at weight `-0.`.

`current_goal_idx` and `how_far_from_start_point` are measured on terrain whose
difficulty the curriculum keeps raising, so a flat curve there can mean "holding
the same performance on harder terrain" rather than "not improving". Read them
against `terrain_levels`, or deconfound with a regression on both. The
uncontaminated measure is MXD at a fixed configuration.

`terrain_levels` never saturates at its maximum: an environment that clears the
last level is reassigned to a uniformly random one, so the mean settles at an
equilibrium that measures the policy's competence in units of terrain difficulty.
