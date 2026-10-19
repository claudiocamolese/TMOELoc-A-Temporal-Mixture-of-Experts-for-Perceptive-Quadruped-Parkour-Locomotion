"""Is an impending fall already visible in the context the actor is handed?

The gap to the teacher is falls, not pace. That leaves two very different
diagnoses, and they call for opposite fixes:

  the warning is in c_t  ->  perception is fine, the actor is not acting on it.
                            Fix the objective: the imitation loss averages over
                            every timestep and the dangerous ones are rare.
  the warning is absent  ->  the student cannot see the failure coming.
                            Fix perception: window, resolution, or grounding.

So: fit a linear classifier from the context at time t to "this environment
falls within the next W control steps", and compare it against the same
classifier on shuffled contexts. Split by environment, never by timestep --
consecutive samples from one environment are near-duplicates and a random split
would leak the answer across it.

  python scripts/analysis/fall_predictability.py <trace_probe.npz> [--horizon 50]
"""

from __future__ import annotations

import argparse

import numpy as np

NUM_GOALS = 8


def label_falls(dones: np.ndarray, goal: np.ndarray, ctx_stride: int, horizon: int, lead: int = 0):
    """Does a *fall* follow in the window ``[t + lead, t + horizon)``?

    A termination at goal ``NUM_GOALS - 1`` is a success, not a fall, and must
    not be counted: successes are what we want more of.

    ``lead`` is what makes the result mean anything. Roll and pitch are part of
    the proprioception that feeds the token, so a context sampled a tenth of a
    second before the cutoff contains the robot already tipping past 1.5 rad;
    predicting that is bookkeeping, not foresight. Requiring the fall to be at
    least ``lead`` steps away excludes the already-doomed states and asks
    whether the danger was visible while it was still avoidable.
    """
    T, N = dones.shape
    fall = dones & (goal < NUM_GOALS - 1)
    n_ctx = (T + ctx_stride - 1) // ctx_stride
    y = np.zeros((n_ctx, N), dtype=bool)
    drop = np.zeros((n_ctx, N), dtype=bool)
    for i in range(n_ctx):
        t = i * ctx_stride
        y[i] = fall[t + lead : t + horizon].any(axis=0)
        # samples inside the excluded lead window are neither positive nor a
        # clean negative -- the fall is already under way -- so drop them
        drop[i] = fall[t : t + lead].any(axis=0) if lead else False
    return y, drop


def fit(X, y, groups, seed=0):
    """Ridge-regularised logistic fit, grouped split, reported as AUC."""
    rng = np.random.default_rng(seed)
    envs = np.unique(groups)
    rng.shuffle(envs)
    cut = int(0.7 * len(envs))
    tr = np.isin(groups, envs[:cut])
    te = ~tr
    if y[tr].sum() < 10 or y[te].sum() < 5:
        return float("nan")

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])

    w = np.zeros(Xtr.shape[1])
    ytr = y[tr].astype(float)
    lam = 1.0
    for _ in range(60):  # Newton steps on the ridge objective
        p = 1.0 / (1.0 + np.exp(-Xtr @ w))
        g = Xtr.T @ (p - ytr) + lam * w
        s = np.clip(p * (1 - p), 1e-6, None)
        H = (Xtr * s[:, None]).T @ Xtr + lam * np.eye(Xtr.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break

    score = Xte @ w
    pos, neg = score[y[te]], score[~y[te]]
    # AUC as the probability a random positive outranks a random negative
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def main():
    """Compare grouped fall-prediction AUC against shuffled-context controls."""
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--horizon", type=int, default=50, help="control steps ahead (50 = 1.0 s)")
    args = ap.parse_args()

    d = np.load(args.trace, allow_pickle=True)
    ctx = d["context"].astype(np.float32)  # (n_ctx, N, C)
    dones = d["dones"].astype(bool)  # (T, N)
    goal = d["goal_idx"].astype(int)
    stride = int(round(dones.shape[0] / ctx.shape[0]))

    print(
        f"context every {stride} control steps, horizon {args.horizon} steps "
        f"({args.horizon * 0.02:.1f} s)"
    )

    print(f"{'lead':>6}{'horizon':>9}{'positives':>11}{'AUC':>8}{'shuffled':>10}")
    for lead, h in [(0, 25), (0, 50), (25, 75), (50, 100), (75, 125), (100, 150)]:
        y, drop = label_falls(dones, goal, stride, h, lead)
        n = min(len(y), len(ctx))
        y, drop, X = y[:n], drop[:n], ctx[:n]
        groups = np.broadcast_to(np.arange(X.shape[1]), y.shape).reshape(-1)
        keep = ~drop.reshape(-1)
        Xf, yf = X.reshape(-1, X.shape[-1])[keep], y.reshape(-1)[keep]
        gf = groups[keep]
        auc = fit(Xf, yf, gf)
        # control: shuffle the contexts across samples, keeping the label rate
        rng = np.random.default_rng(0)
        auc_ctrl = fit(Xf[rng.permutation(len(Xf))], yf, gf)
        print(f"{lead * 0.02:5.1f}s{h * 0.02:8.1f}s{yf.mean():10.2%}{auc:8.3f}{auc_ctrl:10.3f}")

    print("\nAUC 0.5 = the context says nothing about the coming fall.")
    print("Rows with a lead exclude states where the fall is already under way, so")
    print("they measure foresight rather than a tipping robot reading its own tilt.")


if __name__ == "__main__":
    main()
