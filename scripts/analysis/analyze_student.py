"""Offline analysis of an evaluation trace. No Isaac Sim, no GPU.

Answers the three questions the project was set up to ask:

  1. Is the number of experts right?      routing perplexity and usage
  2. Do the experts specialise?           expert x terrain, expert x gait phase
  3. Does the context encode dynamics?    linear probe c_t -> friction / mass / COM

  python scripts/analysis/analyze_student.py logs/.../eval/trace_router.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Layout of the 29-dim privileged latent, from ExtremeParkourObservations.
# Only mass, COM and friction ever vary: joint stiffness and damping are the
# ratio to their defaults and the actuator-gain randomisation event is disabled,
# so 24 of the 29 dimensions are identically zero.
PRIV_NAMES = ["mass", "com_x", "com_y", "com_z", "friction"]


def ridge_probe(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, alpha: float = 1.0):
    """Ridge regression R^2 on held-out data. Returns (r2, baseline_r2)."""
    xt, yt = x[train], y[train]
    mu, sigma = xt.mean(0), xt.std(0) + 1e-6
    xt = (xt - mu) / sigma
    xs = (x[test] - mu) / sigma
    xt = np.hstack([xt, np.ones((len(xt), 1))])
    xs = np.hstack([xs, np.ones((len(xs), 1))])
    reg = alpha * np.eye(xt.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xt.T @ xt + reg, xt.T @ yt)
    pred = xs @ w
    resid = ((y[test] - pred) ** 2).sum()
    total = ((y[test] - yt.mean()) ** 2).sum()
    return 1.0 - resid / max(total, 1e-12), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="trace_*.npz produced by evaluate_students.py")
    ap.add_argument("--out", default=None, help="where to write analysis.json")
    args = ap.parse_args()

    d = np.load(args.trace, allow_pickle=True)
    vocab = [str(s) for s in d["terrain_vocab"]]
    gates = d["gates"].astype(np.float32)          # (T, N, K)
    contacts = d["contacts"].astype(np.float32)    # (T, N, 4)
    terrain = d["terrain"]                         # (T, N)
    out: dict = {"trace": os.path.basename(args.trace)}

    if gates.shape[-1] < 2:
        print("This trace has no routing information (dense actor). Only the probe applies.\n")
        gates = None

    # ---------------------------------------------------- 1. how many experts
    if gates is not None:
        k = gates.shape[-1]
        flat = gates.reshape(-1, k)
        usage = (flat > 0).mean(0)
        weight = flat.mean(0)
        top_k = float((flat > 0).sum(-1).mean())

        # Two different quantities, easy to confuse:
        #  - blend perplexity is computed per step over the *sparse* gates, so
        #    top-k routing bounds it at k, not at num_experts. It says how much
        #    two experts are blended, not how many experts the task uses.
        #  - usage perplexity is computed over the batch-average gate weight and
        #    does range over num_experts. This is the one that answers "how many
        #    experts does this task actually need".
        p = np.clip(flat, 1e-8, None)
        p = p / p.sum(-1, keepdims=True)
        blend_ppl = float(np.exp(-(p * np.log(p)).sum(-1)).mean())
        w = np.clip(weight / max(weight.sum(), 1e-8), 1e-8, None)
        usage_ppl = float(np.exp(-(w * np.log(w)).sum()))

        print("=== 1. expert count ===")
        print(f"experts: {k}   experts active per step: {top_k:.2f} (top-k)")
        print(f"usage perplexity (experts the task actually uses): {usage_ppl:.3f} / {k}")
        print(f"blend perplexity (mixing within the top-k):        {blend_ppl:.3f} / {top_k:.0f}")
        for i in range(k):
            print(f"  expert {i}: selected {usage[i] * 100:5.1f}% of steps, mean gate weight {weight[i]:.3f}")
        verdict = (
            "collapsed -- fewer experts would lose nothing" if usage_ppl < 0.5 * k
            else "saturated -- worth testing more experts" if usage_ppl > 0.95 * k
            else "well matched to the task"
        )
        print(f"verdict: {verdict}\n")
        out["routing"] = {
            "num_experts": int(k), "experts_per_step": top_k,
            "usage_perplexity": usage_ppl, "blend_perplexity": blend_ppl,
            "usage": usage.tolist(), "mean_weight": weight.tolist(), "verdict": verdict,
        }

        # ------------------------------------------- 2a. expert x terrain
        print("=== 2a. expert use by terrain ===")
        print(f"{'terrain':<18} | " + " | ".join(f"{'E' + str(i):>7}" for i in range(k)) + " |  samples")
        table = {}
        for code, name in enumerate(vocab):
            sel = terrain == code
            if not sel.any():
                continue
            share = gates[sel].reshape(-1, k).mean(0)
            share = share / max(share.sum(), 1e-8)
            table[name] = share.tolist()
            print(f"{name:<18} | " + " | ".join(f"{100 * s:6.1f}%" for s in share) + f" | {int(sel.sum()):8d}")
        out["expert_by_terrain"] = table

        # ------------------------------------------- 2b. expert x gait phase
        n_contact = (contacts > 0).sum(-1)
        phases = {"flight (0 feet)": n_contact == 0, "push/land (1)": n_contact == 1,
                  "gait (2-3)": (n_contact >= 2) & (n_contact <= 3), "stance (4)": n_contact == 4}
        print("\n=== 2b. expert use by gait phase ===")
        print(f"{'phase':<18} | " + " | ".join(f"{'E' + str(i):>7}" for i in range(k)) + " |  samples")
        pt = {}
        for name, sel in phases.items():
            if not sel.any():
                continue
            share = gates[sel].reshape(-1, k).mean(0)
            share = share / max(share.sum(), 1e-8)
            pt[name] = share.tolist()
            print(f"{name:<18} | " + " | ".join(f"{100 * s:6.1f}%" for s in share) + f" | {int(sel.sum()):8d}")
        out["expert_by_phase"] = pt

        # a specialised router routes differently in different situations; if
        # every row of those tables is the same, the router ignores the state
        rows = np.array(list(table.values()) + list(pt.values()))
        spread = float(rows.std(0).mean()) if len(rows) > 1 else 0.0
        print(f"\nrouting spread across conditions: {spread:.4f}  "
              f"({'state-dependent' if spread > 0.02 else 'nearly constant -- no specialisation'})")
        out["routing_spread"] = spread

    # ------------------------------------------------- 3. dynamics probe
    ctx = d["context"].astype(np.float32)          # (T', N, D)
    priv = d["priv_latent"].astype(np.float32)     # (T', N, 29)
    n_env = ctx.shape[1]
    print("\n=== 3. linear probe: context -> hidden dynamics ===")
    print(f"context {ctx.shape[2]}-d, {ctx.shape[0]} snapshots x {n_env} environments")

    # Domain randomisation is applied at startup, so every environment holds one
    # constant parameter vector for the whole run. The split must therefore be
    # BY ENVIRONMENT: splitting by timestep would put the same target value in
    # both train and test and report a meaningless R^2.
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_env)
    tr_env, te_env = perm[: n_env // 2], perm[n_env // 2:]
    x = ctx.reshape(-1, ctx.shape[2])
    env_idx = np.tile(np.arange(n_env), ctx.shape[0])
    train = np.isin(env_idx, tr_env)
    test = np.isin(env_idx, te_env)

    probes = {}
    for i, name in enumerate(PRIV_NAMES):
        y = priv[:, :, i].reshape(-1)
        if y.std() < 1e-6:
            print(f"  {name:<10} constant in this environment config -- not probeable")
            continue
        r2, _ = ridge_probe(x, y, train, test)
        shuffled = x[rng.permutation(len(x))]
        r2_ctrl, _ = ridge_probe(shuffled, y, train, test)
        probes[name] = {"r2": float(r2), "r2_shuffled_control": float(r2_ctrl)}
        flag = "recoverable" if r2 > 0.2 else "not recoverable"
        print(f"  {name:<10} R2 = {r2:6.3f}   (shuffled-context control {r2_ctrl:6.3f})   {flag}")
    out["dynamics_probe"] = probes
    if not probes:
        print("  nothing varies: run the probe on the TRAINING task config, not the EVAL one,")
        print("  which disables the mass and COM randomisation events.")

    dst = args.out or os.path.join(os.path.dirname(args.trace), "analysis.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
