# The MoE student at 15 000 iterations: what it does, where it fails, what to change

Measured on `model_14999`, 2 000-step rollouts, 128 environments, the fixed EVAL
config. Student and teacher were scored in the same run of the same script, so
the comparison carries no protocol difference.

## Headline

| | student (MoE) | teacher | gap |
|---|---|---|---|
| goals per env-minute | 25.38 ± 0.64 | 28.54 ± 0.61 | −11.1% (3.6σ) |
| fall hazard per 1k steps | 0.309 ± 0.033 | 0.098 ± 0.018 | **3.2×** (5.6σ) |
| courses completed per env-min | 2.58 | 3.25 | −21% |

Per terrain, the student is behind everywhere:

| terrain | student pace | teacher pace | student hazard | teacher hazard |
|---|---|---|---|---|
| parkour_step | 32.13 | 37.85 | 0.231 | 0.000 |
| parkour | 28.08 | 32.70 | 0.300 | 0.160 |
| parkour_gap | 23.42 | 25.10 | 0.308 | 0.115 |
| parkour_hurdle | 19.74 | 21.84 | **0.400** | 0.100 |

Hurdles are now the worst case at 4× the teacher's fall rate. The mid-training
observation that the student beat the teacher on steps did not survive to
convergence.

## A correction to an earlier claim

I previously reported that the run *degraded* after iteration 11 168, reading a
falling `terrain_levels`. That was wrong. `terrain_levels` is a curriculum
quantity; on the fixed evaluation, at a matched horizon of 500 steps:

| | iter 12 000 | iter 14 999 | difference |
|---|---|---|---|
| goals per env-min | 31.22 ± 0.58 | 30.89 ± 0.71 | 0.4σ — none |
| fall hazard per 1k | 0.188 ± 0.042 | 0.172 ± 0.049 | 0.2σ — none |
| courses per env-min | 0.28 | 0.47 | 1.2σ — none |

The two checkpoints are statistically indistinguishable. The narrower claim
still holds and is still decisive: the last 3 000 iterations bought nothing, so
a longer run will not close the gap.

## What the architecture is actually doing

Eleven conditions, one pathway deleted at a time, weights untouched, paired over
the same 128 terrains. Reference: pace 25.00, hazard 0.302.

| condition | Δ goals/env-min | falls vs reference |
|---|---|---|
| depth frame zeroed | −16.72 ± 0.62 | 5.4× |
| context zeroed | −9.14 ± 0.80 | 4.9× |
| attention → uniform | −1.99 ± 0.81 | 2.0× |
| routing → uniform | **−0.05 ± 0.84** | 2.8× |
| expert 0 alone | −7.96 ± 0.92 | 4.1× |
| expert 1 alone | −14.55 ± 0.68 | 4.4× |
| expert 2 alone | −11.39 ± 0.63 | 6.0× |
| expert 3 alone | −8.53 ± 0.94 | 21.9× |
| all 4 experts, soft weights | +5.89 ± 0.80 | 1.3× |
| top-1 routing | +5.43 ± 0.79 | 1.3× |

**Nothing is decoration.** Every deletion costs between 2× and 22× the falls.

**The self-attention is not a mean pool.** Replacing it with a causal uniform
average — same weights, same window — doubles the falls. The averaged attention
curve looks flat only because individual steps peak at different positions;
per-step divergence from uniform is 3.5× what the mean suggests.

**The router's job is safety, not speed.** This is the sharpest change since
mid-training. Flattening the gates now costs *nothing* in pace (−0.05 ± 0.84,
indistinguishable from zero) while tripling the falls. At iteration 12 500 it
still cost 1.7 goals per minute. The mixture has converged into a fall-avoidance
mechanism, and no expert can run the course alone.

**Top-1 at inference** goes 21% faster for 1.3× the falls, on weights trained at
top-2. An operating point, not a free win.

## What the encoder knows

| dynamics target | probe R² | shuffled control |
|---|---|---|
| com_y | 0.863 | −0.007 |
| mass | 0.754 | −0.003 |
| com_x | 0.419 | −0.009 |
| com_z | −0.225 | −0.010 |
| friction | −0.610 | −0.006 |

Two of the five targets carry no recoverable signal, confirmed with a
same-capacity MLP held out by environment. Friction is the clean case: nothing
in a depth image, or in a gait that has not yet slipped, contains it. The
dynamics term is 35% of the training objective.

Routing specialises by **gait phase, not terrain**: across terrains the gate
distribution barely moves (spread 0.076), across phases it does — flight routes
0.487 to expert 3, push-and-land 0.471 to expert 0, gait and stance 0.41–0.42 to
expert 2. Two experts carry 77% of the blended weight; effective expert count is
1.62 of 4. Attention uses 300 ms of effective lookback out of 700 ms available,
70% of its mass on tokens older than 200 ms.

## What to change

Perception works, the context is load-bearing, the mixture is real, and more
iterations do nothing. What is left is the objective: the action loss averages
over every timestep, and the timesteps that end episodes are well under 1% of
them. A linear readout of the context predicts a fall one second ahead at AUC
0.73 against a shuffled control of 0.54 — the warning is already in the vector
the actor consumes, and the actor is not acting on it.

1. **Weight the imitation loss by predicted risk.** One logit on `c_t` predicting
   "a fall follows in 0.5–1.5 s", supervised from the rollout's own terminations;
   each timestep's action loss scaled by `1 + 4·risk`, renormalised to mean 1,
   capped at 5×, ramped in over 2 000 iterations. 8 321 parameters, +0.1%.
   Implemented and registered as `StudentMoERisk`; defaults off, so this run and
   the baseline are untouched.

2. **Stop regressing what cannot be inferred.** Drop `com_z` and `friction`, keep
   mass and COM x/y — the actor demonstrably uses those.

3. **Give the router a reason to key on terrain.** New, and now the strongest of
   the three: uniform gates cost zero pace while tripling falls, so the mixture
   is doing safety work only, and it selects on gait phase, which proprioception
   already encodes. Feeding the terrain head's output to the router is what would
   let experts differentiate gap from hurdle — which is exactly where the falls
   are.

4. **Score both arms the same way.** Pace, fall hazard and courses per minute at a
   matched horizon with bootstrap CIs, never termination-only MXD, at ≥2 000
   steps — shorter rollouts cannot resolve a 30% change in hazard at 2σ.

```bash
$PYTHON_BIN scripts/rsl_rl/train.py \
    --task Isaac-Extreme-Parkour-StudentMoERisk-Unitree-Go2-v0 \
    --seed 1 --headless --num_envs 192 --max_iterations 15000 \
    --teacher_checkpoint $TEACHER_CKPT
```

## Ruled out

- **Removing or shrinking any component.** Every deletion costs 2–22× the falls.
- **Training longer.** Iterations 12 000 and 14 999 are statistically identical.
- **Chasing the dynamics representation.** Mass and COM are already recovered at
  R² 0.75–0.86; the two missing ones are not inferable in principle.
- **A longer temporal window.** Untested, now lower priority: the model uses
  300 ms of the 700 ms it already has.

## Caveats

- The recipe is a hypothesis backed by diagnosis, **not a measured win**.
- **One seed** throughout. With hazard SEs near ±0.03, a small final margin could
  be seed noise.
- The GRU baseline **warm-starts its actor from the teacher**; the MoE cannot and
  trains its actor from scratch. Inherent to the architectures, and it favours
  the baseline.
- Zeroing the depth frame is a *corruption*, not a clean deletion: an
  out-of-distribution image yields a confidently wrong context, which is why it
  costs more pace than removing the context outright.
- The baseline has not finished, so no student-vs-student number exists yet.
