"""Strip optimiser state from a checkpoint so it can be committed.

A training checkpoint here is 103 MB, of which 66.6 MB is the Adam state for the
student (two moment tensors per parameter). That is needed only to *resume*
training, never to run or share a policy, and it puts the file over GitHub's
100 MB hard limit. Dropping it leaves 36.5 MB.

  python scripts/export_slim_checkpoint.py logs/.../model_15000.pt
  python scripts/export_slim_checkpoint.py logs/.../  --every 5000
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import torch

# everything needed to evaluate, play or export a policy
KEEP = (
    "model_state_dict",            # teacher / actor-critic
    "estimator_state_dict",
    "depth_encoder_state_dict",    # student perception + temporal context
    "depth_actor_state_dict",      # student actor (MoE or dense)
    "obs_norm_state_dict",
    "privileged_obs_norm_state_dict",
    "iter",
    "infos",
)


def slim(src: str, dst: str) -> tuple[float, float]:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    out = {k: v for k, v in ckpt.items() if k in KEEP}
    dropped = sorted(set(ckpt) - set(out))
    torch.save(out, dst)
    before = os.path.getsize(src) / 1048576
    after = os.path.getsize(dst) / 1048576
    print(f"  {os.path.basename(src):<20} {before:6.1f} MB -> {after:6.1f} MB   dropped: {', '.join(dropped) or 'nothing'}")
    return before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a model_*.pt, or a run directory")
    ap.add_argument("--out_dir", default=None, help="default: <run>/slim")
    ap.add_argument("--every", type=int, default=None,
                    help="when given a directory, keep only iterations that are multiples of this")
    args = ap.parse_args()

    if os.path.isfile(args.path):
        srcs, run = [args.path], os.path.dirname(args.path)
    else:
        run = args.path.rstrip("/")
        srcs = sorted(glob.glob(os.path.join(run, "model_*.pt")),
                      key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
        if args.every:
            last = srcs[-1]
            srcs = [p for p in srcs
                    if int(re.search(r"model_(\d+)", p).group(1)) % args.every == 0]
            if last not in srcs:
                srcs.append(last)  # always keep the final one
    if not srcs:
        raise SystemExit(f"no model_*.pt under {args.path}")

    out_dir = args.out_dir or os.path.join(run, "slim")
    os.makedirs(out_dir, exist_ok=True)
    tot_b = tot_a = 0.0
    for s in srcs:
        b, a = slim(s, os.path.join(out_dir, os.path.basename(s)))
        tot_b += b
        tot_a += a
    print(f"\n{len(srcs)} checkpoint(s): {tot_b:.0f} MB -> {tot_a:.0f} MB   written to {out_dir}")
    over = [f for f in os.listdir(out_dir) if os.path.getsize(os.path.join(out_dir, f)) > 100 * 1048576]
    if over:
        print(f"WARNING: still over GitHub's 100 MB limit: {', '.join(over)}")


if __name__ == "__main__":
    main()
