"""Does the Transformer use its temporal window, or only the current frame?

This is the test that decides whether replacing the GRU bought anything. The
context is built from 8 tokens spanning 0.8 s; if the newest token attends
almost entirely to itself, the window is decoration and the model is close to a
per-frame MLP.

Reports the attention the newest token pays to each position, overall and split
by gait phase, plus an effective lookback in milliseconds.

  python scripts/analysis/attention_probe.py logs/.../eval/trace_attn.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

STEP_MS = 100.0  # one token per perception step, 10 Hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.trace
    if path is None or not os.path.isfile(path):
        found = glob.glob(os.path.join(path or "logs", "**", "trace_*.npz"), recursive=True)
        found = [f for f in found if "attention" in np.load(f, allow_pickle=True).files]
        if not found:
            raise SystemExit("no trace with recorded attention; rerun evaluate_students.py --attention")
        path = max(found, key=os.path.getmtime)
    print(f"reading {path}\n")

    d = np.load(path, allow_pickle=True)
    if "attention" not in d.files:
        raise SystemExit("this trace has no attention; rerun evaluate_students.py --attention")
    a = d["attention"].astype(np.float32)        # (T, N, layers, H)
    contacts = d["attn_contacts"].astype(np.float32)  # (T, N, 4)
    T, N, L, H = a.shape
    print(f"{T} perception steps x {N} environments, {L} layers, window {H} tokens "
          f"({H * STEP_MS / 1000:.1f} s)\n")

    uniform = 1.0 / H
    pos_labels = [f"t-{(H - 1 - i) * int(STEP_MS)}ms" for i in range(H)]
    pos_labels[-1] = "now"

    out: dict = {"window": H, "layers": L}
    print("=== attention of the newest token, averaged over the rollout ===")
    print(f"{'layer':>7} | " + " | ".join(f"{p:>9}" for p in pos_labels))
    print("-" * (9 + 12 * H))
    for l in range(L):
        m = a[:, :, l, :].reshape(-1, H).mean(0)
        print(f"{l:>7} | " + " | ".join(f"{v:9.4f}" for v in m))
    m_all = a.reshape(-1, H).mean(0)
    print(f"{'mean':>7} | " + " | ".join(f"{v:9.4f}" for v in m_all))
    print(f"{'uniform':>7} | " + " | ".join(f"{uniform:9.4f}" for _ in range(H)))
    out["mean_attention"] = m_all.tolist()

    # how much of the mass sits outside the two most recent tokens
    older = m_all[: H - 2].sum()
    print(f"\nmass on tokens older than 200 ms : {older:.3f}   (uniform would be {(H - 2) / H:.3f})")
    idx = np.arange(H)[::-1] * STEP_MS       # 700, 600, ... 0 ms into the past
    lookback = float((m_all * idx).sum())
    print(f"effective lookback                : {lookback:.0f} ms   "
          f"(uniform {(idx.mean()):.0f} ms, newest-token-only 0 ms)")
    out["mass_older_than_200ms"] = float(older)
    out["effective_lookback_ms"] = lookback

    verdict = ("the window is decoration: attention sits on the current frame" if lookback < 100
               else "history is used, but shallowly" if lookback < 250
               else "the full window is in use")
    print(f"verdict: {verdict}")
    out["verdict"] = verdict

    # ------------------------------------------------ conditioned on gait phase
    n_contact = (contacts > 0).sum(-1)
    phases = {"flight (0 feet)": n_contact == 0, "push/land (1)": n_contact == 1,
              "gait (2-3)": (n_contact >= 2) & (n_contact <= 3), "stance (4)": n_contact == 4}
    print("\n=== does the lookback change with the gait phase? ===")
    print(f"{'phase':<18} | {'lookback':>9} | " + " | ".join(f"{p:>8}" for p in pos_labels) + " | samples")
    table = {}
    for name, sel in phases.items():
        if sel.sum() < 50:
            continue
        m = a[sel].reshape(-1, H).mean(0)
        lb = float((m * idx).sum())
        table[name] = {"lookback_ms": lb, "attention": m.tolist(), "samples": int(sel.sum())}
        print(f"{name:<18} | {lb:8.0f}ms | " + " | ".join(f"{v:8.4f}" for v in m) + f" | {int(sel.sum()):7d}")
    out["by_phase"] = table
    if len(table) > 1:
        lbs = {k: v["lookback_ms"] for k, v in table.items()}
        hi, lo = max(lbs, key=lbs.get), min(lbs, key=lbs.get)
        print(f"\nlongest lookback in '{hi}' ({lbs[hi]:.0f} ms), shortest in '{lo}' ({lbs[lo]:.0f} ms), "
              f"spread {lbs[hi] - lbs[lo]:.0f} ms")

    dst = args.out or os.path.join(os.path.dirname(path), "attention.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
