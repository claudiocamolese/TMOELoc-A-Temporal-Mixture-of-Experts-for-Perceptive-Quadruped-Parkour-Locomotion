"""Is the RL stage actually improving the policy?

Reward alone cannot answer that here. The fine-tuning environment inherits the
training curriculum, so terrain difficulty climbs during the run and a falling
reward may only mean the same policy is being asked for more. The trend that
matters is reward at constant difficulty, which is the terrain-level coefficient
regressed out.

  python scripts/rl_verdict.py                 # both arms
  python scripts/rl_verdict.py --min-iter 2000 # only judge once there is enough
"""
from __future__ import annotations
import argparse, re, sys
import numpy as np

PAT = re.compile(
    r"it\s+(\d+) \| reward\s+([-\d.]+) \| ep_len\s+([-\d.]+).*?kl ([\d.]+) \| lr ([\d.e+-]+).*?lvl ([\d.]+)"
)


def load(path):
    rows = []
    try:
        f = open(path, errors="ignore")
    except FileNotFoundError:
        return None
    for line in f:
        for sub in line.replace("\r", "\n").split("\n"):
            m = PAT.match(sub)
            if m:
                rows.append([float(x) for x in m.groups()])
    return np.array(rows) if rows else None


def judge(a, min_iter, warmup_frac=0.15):
    """Slope of reward per 1000 iterations, holding terrain level fixed.

    The first fraction of the run is dropped: the policy is still absorbing the
    switch from a deterministic distilled actor to a stochastic one, and that
    transient is not what we are trying to measure.
    """
    it, rw, ep, kl, lr, lv = a.T
    if it.max() < min_iter:
        return None, f"only {int(it.max())} iterations, need {min_iter}"
    keep = it >= it.max() * warmup_frac
    it, rw, lv, lr = it[keep], rw[keep], lv[keep], lr[keep]
    raw = np.polyfit(it, rw, 1)[0] * 1000
    if lv.std() > 0.05:
        A = np.vstack([it / 1000.0, lv, np.ones_like(lv)]).T
        coef, *_ = np.linalg.lstsq(A, rw, rcond=None)
        adj = coef[0]
    else:
        adj = raw
    # bootstrap the deconfounded slope so "negative" means more than noise
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(400):
        idx = rng.integers(0, len(it), len(it))
        A = np.vstack([it[idx] / 1000.0, lv[idx], np.ones(len(idx))]).T
        try:
            c, *_ = np.linalg.lstsq(A, rw[idx], rcond=None)
            boots.append(c[0])
        except Exception:
            pass
    se = float(np.std(boots)) if boots else float("nan")
    at_floor = 100 * float((lr <= 1.01e-5).mean())
    msg = (f"iter {int(a[-1,0])}  reward/1k raw {raw:+.2f}  at constant difficulty "
           f"{adj:+.2f} +- {se:.2f}  terrain {lv.mean():.2f}  lr at floor {at_floor:.0f}%")
    if np.isnan(se) or se == 0:
        return "UNRESOLVED", msg
    if adj > 2 * se:
        return "IMPROVING", msg
    if adj < -2 * se:
        return "DEGRADING", msg
    return "FLAT", msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-iter", type=int, default=2000)
    args = ap.parse_args()
    out = {}
    for tag, path in (("dense", "logs/run_console/ft_moe.log"),
                      ("top2", "logs/run_console/ft_top2.log"),
                      ("baseline", "logs/run_console/ft_base.log")):
        a = load(path)
        if a is None or len(a) < 20:
            print(f"{tag:9s} no data yet")
            continue
        verdict, msg = judge(a, args.min_iter)
        out[tag] = verdict
        print(f"{tag:9s} {str(verdict):11s} {msg}")
    if out and all(v == "DEGRADING" for v in out.values()):
        print("\nEvery arm is degrading at constant difficulty. That is not a tuning "
              "problem; stop and look at the advantage computation before spending "
              "the remaining budget.")
    elif any(v == "IMPROVING" for v in out.values()):
        print("\nAt least one arm is improving; let them run.")


if __name__ == "__main__":
    main()
