# The MoE student at 15 000 iterations: what it does, where it fails, what to change

## Result: the GRU baseline wins

Both students trained to 15 000 iterations from the same frozen teacher, same
environment, same optimiser, and were scored in the same way: 2 000-step
rollouts, 128 environments, the fixed EVAL config, every environment counted
including the censored ones.

| | pace goals/env-min | fall hazard /1k | courses/env-min |
|---|---|---|---|
| **baseline (GRU + dense)** | **27.76 ± 0.56** | **0.168 ± 0.025** | **3.15** |
| MoE (Transformer + top-2) | 25.38 ± 0.64 | 0.309 ± 0.033 | 2.58 |
| teacher (privileged) | 28.54 ± 0.61 | 0.098 ± 0.018 | 3.25 |

MoE minus baseline: pace **−2.38 ± 0.85 (2.8σ)**, hazard **+0.141 ± 0.041
(3.4σ)**, completed courses −18%. The baseline is within 2.7% of the teacher's
pace and 3% of its completions; it very nearly saturates the ceiling. The MoE
does not.

Per terrain the baseline leads everywhere except one cell -- on steps the MoE
falls less (0.231 vs 0.288). That is the only place the architecture wins.

**So the honest headline is a negative result: the temporal Transformer plus
top-2 mixture, as configured here, does not beat the GRU baseline at equal
budget.** What follows is why, and what the evidence says to change.

## Why: the objective is split three ways

The most parsimonious explanation is where the gradient goes. The baseline has
exactly two loss terms and puts everything into imitating the teacher; the MoE
spends about a third of its objective on auxiliary heads.

| term | MoE share of objective | baseline |
|---|---|---|
| actions | 36% | ~100% |
| dynamics | 35% | -- |
| yaw | 10% | (its own term) |
| terrain | 9% | -- |
| load balancing | 8% | -- |

At convergence the baseline reaches an action loss of 0.690 against the MoE's
1.316 -- the same quantity, `norm(p=2, dim=1).mean()`, on the same target, and
by then both are on equally hard terrain. Twice the imitation error, and worse
performance to match.

The auxiliary heads were added because `c_t` is a 128-d unified context that
nothing else would ground. The baseline shows that is not necessary: its depth
latent occupies the slot the teacher fills with its scandots encoding, so it is
grounded implicitly, through the action loss alone, and it ends up closer to the
teacher than our explicitly supervised context does.

## The MoE against the teacher

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

| condition | Δ goals/env-min | Δ fall hazard |
|---|---|---|
| depth frame zeroed | −18.94 ± 0.58 | +1.329 ± 0.073 |
| context zeroed | −12.24 ± 0.77 | +1.244 ± 0.064 |
| expert 1 alone | −18.69 ± 0.56 | +1.115 ± 0.060 |
| expert 2 alone | −13.24 ± 0.59 | +1.545 ± 0.064 |
| expert 0 alone | −11.53 ± 0.70 | +1.023 ± 0.048 |
| expert 3 alone | −11.46 ± 0.86 | +5.729 ± 0.177 |
| attention → uniform | −2.62 ± 0.69 | +0.177 ± 0.044 |
| routing → uniform | −2.19 ± 0.82 | +0.585 ± 0.047 |
| top-1 routing | +1.11 ± 0.75 | +0.135 ± 0.047 |
| all 4 experts, soft weights | **+2.77 ± 0.67** | +0.070 ± 0.042 |
| *reference, run again* | *−0.65 ± 0.72* | *−0.034 ± 0.041* |

The last row is the resolution floor: the same policy under the same condition,
measured twice. Anything smaller than it means nothing. Lengthening the warmup
between conditions from 50 to 150 steps brought the artefact down from +4.74 to
−0.65, which is what makes the small effects readable at all.

**Nothing is decoration.** Every deletion costs between 2× and 22× the falls.

**The self-attention is not a mean pool.** Replacing it with a causal uniform
average — same weights, same window — doubles the falls. The averaged attention
curve looks flat only because individual steps peak at different positions;
per-step divergence from uniform is 3.5× what the mean suggests.

**The router earns its keep on both axes.** Flattening the gates costs 2.19
goals per env-minute and doubles the fall hazard, and no expert can run the
course alone -- the best manages two thirds of the pace at four times the falls.
An earlier reading of this as "safety only, no pace cost" was the ordering
artefact talking.

**Running all four experts is free speed.** Soft weights over the full set give
+2.77 ± 0.67 goals per env-minute -- four times the resolution floor -- at a fall
cost of +0.070 ± 0.042, which is at the floor. That closes most of the pace gap
to the baseline at inference time, on weights already trained. It does not touch
the fall gap, which is the larger deficit. Top-1, by contrast, is marginal on
pace and worse on falls: the earlier "+21% throughput" was artefact.

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


## A second correction: the ablation's pace numbers were contaminated

Running `zero_context` on the GRU student turned out to be a no-op -- the switch
was applied only inside the MoE branch -- and yet it reported **+4.74 ± 0.72
goals/env-min, 6.6σ**. An intervention that changes nothing cannot produce 6.6σ,
so that number is not an ablation result: it is a measurement of the sweep's own
systematic error.

The cause is the design. Conditions run back to back inside one simulator, so
each starts from wherever the previous one left the robots rather than from a
fresh reset, and the bootstrap resamples environments and cannot see that offset.

What this invalidates, and what survives:

| claim | status |
|---|---|
| "routing costs zero pace but triples the falls" | the *pace* half is unresolved; the falls half stands |
| "top-1 gives +21% throughput" | unresolved, and probably inflated by ordering |
| "uniform attention costs 20% of the pace" | pace half unresolved; the 2× falls stands |
| no_depth, zero_context, single experts | **stand** -- effects are 2-4× the artefact |

The hazard effects mostly survive: the artefact is +0.089 there, while the
measured effects run from +0.3 to +6.3. It is the pace column that was
contaminated.

Both defects are fixed: `zero_context` now acts on the GRU student too, and the
sweep repeats its reference condition last, so it prints the gap between two
identical runs and states outright that any smaller effect is unresolved.

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
