"""Fit linear probes from the context to the hidden dynamics, and save the directions.

Two uses:

  1. report how much of each physical parameter is linearly decodable from c_t
  2. save the direction in c_t space that carries it, so that
     `evaluate_students.py --ablate_context` can project it out and test whether
     the actor actually *uses* the information, rather than merely storing it

The trace must come from the TRAINING task, not an EVAL one: the EVAL configs
switch the mass and COM randomisation events off, leaving nothing to predict.

  python scripts/analysis/fit_context_probe.py <trace_*.npz>
  python scripts/analysis/fit_context_probe.py            # newest suitable trace
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

# informative entries of the 29-d privileged latent; the other 24 are joint
# stiffness and damping ratios, identically zero because the actuator-gain
# randomisation event is disabled
PRIV_NAMES = ["mass", "com_x", "com_y", "com_z", "friction"]


def find_trace(path: str | None) -> str:
    if path and os.path.isfile(path):
        return path
    cands = glob.glob(os.path.join(path or "logs", "**", "trace_*.npz"), recursive=True)
    good = []
    for c in cands:
        try:
            d = np.load(c, allow_pickle=True)
            if "context" in d.files and "priv_latent" in d.files:
                p = d["priv_latent"].astype(np.float32)
                if p.shape[-1] >= 5 and p[:, :, 0].std() > 1e-6:   # mass actually varies
                    good.append(c)
        except Exception:
            pass
    if not good:
        raise SystemExit(
            "no usable trace found.\nRecord one on the TRAINING task (mass and COM must vary):\n"
            "  python scripts/rsl_rl/evaluate_students.py \\\n"
            "      --task Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-v0 \\\n"
            "      --checkpoint <model_*.pt> --num_envs 192 --steps 600 --context_every 3 --headless"
        )
    return max(good, key=os.path.getmtime)


def ridge(x, y, train, test, alpha=1.0):
    """Returns (r2, direction in raw x space)."""
    xt, yt = x[train], y[train]
    mu, sigma = xt.mean(0), xt.std(0) + 1e-6
    xs = np.hstack([(xt - mu) / sigma, np.ones((len(xt), 1))])
    reg = alpha * np.eye(xs.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xs.T @ xs + reg, xs.T @ yt)
    pred = np.hstack([(x[test] - mu) / sigma, np.ones((test.sum(), 1))]) @ w
    resid = ((y[test] - pred) ** 2).sum()
    total = ((y[test] - yt.mean()) ** 2).sum()
    r2 = 1.0 - resid / max(total, 1e-12)
    # the functional is w[:-1] . (x - mu)/sigma, so in raw x space the direction
    # along which the target varies is w[:-1] / sigma
    direction = w[:-1] / sigma
    n = np.linalg.norm(direction)
    return float(r2), (direction / n if n > 0 else direction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default=None)
    ap.add_argument("--out", default=None, help="default: <trace_dir>/probe_directions.npz")
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge regularisation")
    args = ap.parse_args()

    path = find_trace(args.trace)
    print(f"reading {path}")
    d = np.load(path, allow_pickle=True)
    ctx = d["context"].astype(np.float32)
    priv = d["priv_latent"].astype(np.float32)
    n_env = ctx.shape[1]
    print(f"context {ctx.shape[2]}-d, {ctx.shape[0]} snapshots x {n_env} environments\n")

    # Domain randomisation runs at startup, so each environment holds one constant
    # parameter vector for its whole life. The split has to be BY ENVIRONMENT:
    # splitting by timestep puts the same target value in train and test.
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_env)
    tr, te = perm[: n_env // 2], perm[n_env // 2:]
    x = ctx.reshape(-1, ctx.shape[2])
    env_idx = np.tile(np.arange(n_env), ctx.shape[0])
    train, test = np.isin(env_idx, tr), np.isin(env_idx, te)

    dirs, report = {}, {}
    print(f"{'target':<10} | {'R2':>7} | {'control':>8} | verdict")
    print("-" * 48)
    for i, name in enumerate(PRIV_NAMES):
        y = priv[:, :, i].reshape(-1)
        if y.std() < 1e-6:
            print(f"{name:<10} |       - |        - | constant in this trace")
            continue
        r2, v = ridge(x, y, train, test, args.alpha)
        r2c, _ = ridge(x[rng.permutation(len(x))], y, train, test, args.alpha)
        good = r2 > 0.2
        print(f"{name:<10} | {r2:7.3f} | {r2c:8.3f} | {'recoverable' if good else 'not recoverable'}")
        report[name] = {"r2": r2, "r2_shuffled_control": r2c}
        if good:
            dirs[name] = v.astype(np.float32)

    if not dirs:
        print("\nnothing recoverable; no directions to save")
        return
    dst = args.out or os.path.join(os.path.dirname(path), "probe_directions.npz")
    np.savez(dst, **dirs)
    with open(os.path.splitext(dst)[0] + ".json", "w") as fh:
        json.dump({"trace": path, "probes": report, "saved_directions": list(dirs)}, fh, indent=2)
    print(f"\nsaved {len(dirs)} direction(s) [{', '.join(dirs)}] to {dst}")
    print("\nnext: test whether the actor uses them\n"
          "  python scripts/rsl_rl/evaluate_students.py --task <...-Eval-v0> \\\n"
          f"      --checkpoint <model_*.pt> --ablate_context {dst} --ablate_dims {','.join(dirs)} \\\n"
          "      --num_envs 192 --steps 800 --tag ablated --headless")


if __name__ == "__main__":
    main()
