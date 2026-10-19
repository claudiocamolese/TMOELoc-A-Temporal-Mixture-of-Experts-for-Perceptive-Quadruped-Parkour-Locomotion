"""Unbiased episode statistics from an evaluation trace.

MXD as reported by ``evaluate_students.py`` is conditional on the episode having
*ended*. On the EVAL config an episode lasts 1000 control steps and the rollouts
are shorter than that, so a policy that neither falls nor finishes contributes
nothing to the average. In the 700-step MoE rollout 108 of 192 environments never
terminated, i.e. 56% of the evidence is discarded, and the discarded part is not
missing at random -- it is exactly the population of steady, mid-pace runs.

That is enough to invert a comparison: the teacher terminated 27 times and scored
0.727, the student 85 times and scored 0.741, which reads as "the student beat
its own ceiling" and is an artefact.

This script recomputes the numbers using every environment. A termination is
either a *success* (``cur_goal_idx`` reached ``num_goals``) or a *fall* (roll,
pitch or height cutoff); the two are competing risks, so they are reported
separately and per unit of time rather than per episode. Environments still
running at the horizon are censored, not dropped.

  python scripts/analysis/episode_metrics.py <trace_*.npz> [<trace_*.npz> ...]
  python scripts/analysis/episode_metrics.py --glob 'logs/**/trace_main.npz'
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

NUM_GOALS = 8  # terrain_generator_cfg.num_goals
CONTROL_DT = 0.02  # 50 Hz control step
MAX_EPISODE_STEPS = 1000  # episode_length_s 20.0 / CONTROL_DT, EVAL config


def episode_table(dones: np.ndarray, goal: np.ndarray):
    """Split every recorded termination into success / fall, and censor the rest.

    ``goal[t, n]`` is ``cur_goal_idx`` *before* step ``t``, so on the step where
    the environment finishes it still reads ``NUM_GOALS - 1``: the increment that
    triggers the cutoff happens inside the step. A termination at that value is a
    success, anything lower is a fall.

    Returns one record per episode-segment, censored ones included.
    """
    T, N = dones.shape
    rows = []
    for n in range(N):
        start = 0
        for t in np.flatnonzero(dones[:, n]):
            reached = int(goal[t, n])
            rows.append(
                dict(
                    env=n,
                    t0=start,
                    t1=int(t),
                    steps=int(t) - start + 1,
                    goal=reached,
                    censored=False,
                    success=reached >= NUM_GOALS - 1,
                )
            )
            start = int(t) + 1
        if start < T:  # still running at the horizon
            rows.append(
                dict(
                    env=n,
                    t0=start,
                    t1=T - 1,
                    steps=T - start,
                    goal=int(goal[-1, n]),
                    censored=True,
                    success=False,
                )
            )
    return rows


def bootstrap_over_envs(rows, stat, n_boot=2000, seed=0):
    """Resample *environments*, not segments.

    Segments from one environment share a terrain assignment and a body-mass
    draw, so they are not independent; resampling them individually would give
    error bars that are too tight. Environments are independent by construction.
    """
    rng = np.random.default_rng(seed)
    by_env: dict[int, list] = {}
    for r in rows:
        by_env.setdefault(r["env"], []).append(r)
    envs = list(by_env)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(envs), len(envs))
        sample = [r for i in pick for r in by_env[envs[i]]]
        vals[b] = stat(sample)
    return float(vals.std()), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _goals_per_min(rows):
    """Compute progress per minute using all observed trajectory exposure."""
    g = sum(r["goal"] for r in rows)
    s = sum(r["steps"] for r in rows)
    return g / (s * CONTROL_DT) * 60.0


def _fall_hazard(rows):
    """Compute falls per 1,000 observed control steps, including censored exposure."""
    f = sum(1 for r in rows if not r["censored"] and not r["success"])
    s = sum(r["steps"] for r in rows)
    return f / s * 1000.0


def summarise(rows, label="", boot=0):
    """Aggregate episode segments and optionally bootstrap uncertainty over environments."""
    steps = np.array([r["steps"] for r in rows], dtype=float)
    goals = np.array([r["goal"] for r in rows], dtype=float)
    cens = np.array([r["censored"] for r in rows])
    succ = np.array([r["success"] for r in rows])
    fall = ~cens & ~succ

    exposure_s = steps.sum() * CONTROL_DT
    # goals cleared per unit of exposure: the one quantity censoring does not bias,
    # because a censored segment contributes both its progress and its time
    progress_rate = goals.sum() / exposure_s

    out = {
        "segments": int(len(rows)),
        "terminated": int((~cens).sum()),
        "censored": int(cens.sum()),
        "censored_frac": float(cens.mean()),
        "exposure_min": exposure_s / 60.0,
        "success_per_min": float(succ.sum() / exposure_s * 60.0),
        "fall_per_min": float(fall.sum() / exposure_s * 60.0),
        "goals_per_min": float(progress_rate * 60.0),
        # what the old metric reported: mean over terminated segments only
        "mxd_terminated_only": float((goals[~cens] / NUM_GOALS).mean())
        if (~cens).any()
        else float("nan"),
        # every segment, censored ones at the progress they had reached
        "mxd_all_segments": float((goals / NUM_GOALS).mean()),
        "mxd_all_sem": float((goals / NUM_GOALS).std(ddof=1) / np.sqrt(len(rows))),
    }
    # hazard: per-step probability of falling while a segment is alive. Unlike a
    # per-episode fall fraction this is comparable between policies that travel
    # at different speeds.
    out["fall_hazard_per_1k_steps"] = float(fall.sum() / steps.sum() * 1000.0)
    out["success_share_of_ends"] = float(succ.sum() / max((~cens).sum(), 1))
    if boot:
        out["goals_per_min_se"], out["goals_per_min_ci"] = bootstrap_over_envs(
            rows, _goals_per_min, boot
        )
        out["fall_hazard_se"], out["fall_hazard_ci"] = bootstrap_over_envs(rows, _fall_hazard, boot)
    if label:
        out["trace"] = label
    return out


def per_terrain(rows, terrain, vocab):
    """Same statistics, split by the terrain the segment started on."""
    out = {}
    for code, name in enumerate(vocab):
        sel = [r for r in rows if terrain[r["t0"], r["env"]] == code]
        if len(sel) >= 5:
            out[str(name)] = summarise(sel)
    return out


def main():
    """Load evaluation traces and print or save matched-horizon traversal statistics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--out", default=None, help="write the summary as json")
    ap.add_argument("--per-terrain", action="store_true")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples; 0 disables")
    ap.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="truncate every trace to this many steps, so the censoring "
        "fraction is identical across policies",
    )
    args = ap.parse_args()

    paths = list(args.traces)
    if args.glob:
        paths += sorted(glob.glob(args.glob, recursive=True))
    if not paths:
        raise SystemExit("give at least one trace_*.npz")

    report = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        if "dones" not in d.files:
            print(f"skipping {p}: no 'dones'")
            continue
        dones = d["dones"].astype(bool)
        goal = d["goal_idx"].astype(int)
        if args.horizon:
            dones, goal = dones[: args.horizon], goal[: args.horizon]
        rows = episode_table(dones, goal)
        name = os.path.basename(p)
        s = summarise(rows, name, boot=args.boot)
        report[p] = s
        if args.per_terrain and "terrain" in d.files:
            report[p]["per_terrain"] = per_terrain(
                rows, d["terrain"].astype(int), d["terrain_vocab"]
            )
        T, N = dones.shape
        print(
            f"\n=== {p}  ({T} steps x {N} envs, {T * CONTROL_DT:.0f}s per env, "
            f"episode cap {MAX_EPISODE_STEPS} steps) ==="
        )

    if not report:
        raise SystemExit("nothing to report")

    keys = [
        ("censored_frac", "never ended (censored)", "{:.0%}"),
        ("success_share_of_ends", "of the ends, successes", "{:.0%}"),
        ("success_per_min", "successes / env-min", "{:.2f}"),
        ("fall_per_min", "falls / env-min", "{:.2f}"),
        ("fall_hazard_per_1k_steps", "fall hazard / 1k steps", "{:.2f}"),
        ("goals_per_min", "goals cleared / env-min", "{:.2f}"),
        ("mxd_all_segments", "MXD, every segment", "{:.3f}"),
        ("mxd_terminated_only", "MXD, terminated only (old)", "{:.3f}"),
    ]
    names = [os.path.basename(p).replace("trace_", "").replace(".npz", "") for p in report]
    w = max(26, max(len(n) for n in names) + 2)
    print(f"\n{'':<28}" + "".join(f"{n:>{w}}" for n in names))
    print("-" * (28 + w * len(names)))
    for k, lbl, fmt in keys:
        print(f"{lbl:<28}" + "".join(fmt.format(r[k]).rjust(w) for r in report.values()))
    if args.boot:
        print()
        for k, se, lbl, fmt in [
            ("goals_per_min", "goals_per_min_se", "goals/env-min +- boot SE", "{:.2f}"),
            ("fall_hazard_per_1k_steps", "fall_hazard_se", "fall hazard +- boot SE", "{:.3f}"),
        ]:
            print(
                f"{lbl:<28}"
                + "".join(
                    (fmt.format(r[k]) + " +- " + fmt.format(r[se])).rjust(w)
                    for r in report.values()
                )
            )
    print("\nMXD ceiling is 7/8 = 0.875: the goal counter reads 7 on the step the")
    print("cutoff fires, so a flawless run scores 0.875, not 1.0.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
