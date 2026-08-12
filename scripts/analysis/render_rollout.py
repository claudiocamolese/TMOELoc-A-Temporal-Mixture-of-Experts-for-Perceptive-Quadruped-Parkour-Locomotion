"""Third-person video of a rollout, reconstructed from state. No renderer.

Isaac Sim needs an RTX device for RecordVideo, which is unavailable on hosts
without an NVIDIA Vulkan ICD (WSL2, in particular). None of that is needed to
*see* the robot: the simulator already reports where every link is, and the
height scanner reports the terrain around it. This draws both.

  python scripts/analysis/render_rollout.py logs/.../playback/rollout_router.npz --env 0
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

LEGS = ["FL", "FR", "RL", "RR"]

# The robot and the gate panel must not share a palette. An earlier version gave
# the legs the expert colours, which reads as "the blue expert drives the blue
# leg". No such mapping exists: every expert outputs all twelve joints, and the
# router picks a pair of experts for the whole body. The robot is drawn in a
# single flat white so no part of it can be read as carrying a colour code;
# which foot is which stays legible from geometry, and contact from whether the
# marker is filled.
ROBOT_COLOR = "#ffffff"
LEG_COLORS = {leg: ROBOT_COLOR for leg in LEGS}
BASE_COLOR = ROBOT_COLOR
EXPERT_COLORS = ["#4da3ff", "#5fdc78", "#ff9646", "#dc6edc"]


def build_skeleton(body_names: list[str]):
    """Chains of link indices to draw as connected segments."""
    idx = {n: i for i, n in enumerate(body_names)}
    chains = []
    for leg in LEGS:
        chain = [idx.get("base")]
        for part in ("hip", "thigh", "calf", "foot"):
            j = idx.get(f"{leg}_{part}")
            if j is not None:
                chain.append(j)
        chains.append((leg, [c for c in chain if c is not None]))
    return chains


def resolve_rollout(path: str | None) -> str:
    """Accept a file, a directory anywhere above the rollout, or nothing."""
    if path and os.path.isfile(path):
        return path
    roots = [path] if path else ["logs"]
    found: list[str] = []
    for root in roots:
        if not os.path.exists(root):
            continue
        found += glob.glob(os.path.join(root, "**", "rollout_*.npz"), recursive=True)
    if not found:
        where = path or "logs/"
        raise SystemExit(
            f"no rollout_*.npz under {where}\n"
            "record one first:\n"
            "  python scripts/rsl_rl/play_student.py --task <...-Play-v0> "
            "--checkpoint <model_*.pt> --num_envs 4 --steps 400 --headless"
        )
    newest = max(found, key=os.path.getmtime)
    if len(found) > 1:
        print(f"{len(found)} rollouts found, using the newest")
    return newest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout", nargs="?", default=None,
                    help="rollout_*.npz, or a run directory, or omit for the newest")
    ap.add_argument("--env", type=int, default=0, help="which environment to follow")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=2, help="use every Nth control step")
    ap.add_argument("--out", default=None)
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=-70.0)
    args = ap.parse_args()

    args.rollout = resolve_rollout(args.rollout)
    print(f"reading {args.rollout}")
    d = np.load(args.rollout, allow_pickle=True)
    if "body_pos" not in d:
        raise SystemExit("This rollout predates the state dump. Re-record with play_student.py.")

    e = args.env
    body_pos = d["body_pos"][:, e]           # (T, num_bodies, 3)
    scan = d["scan_hits"][:, e]              # (T, num_rays, 3)
    contacts = d["contacts"][:, e]           # (T, 4)
    gates = d["gates"][:, e] if d["gates"].shape[-1] > 1 else None
    terrain = str(d["terrain_names"][e]) if "terrain_names" in d else "?"
    body_names = [str(s) for s in d["body_names"]]
    chains = build_skeleton(body_names)
    steps = range(0, len(body_pos), args.stride)

    out = args.out or os.path.join(
        os.path.dirname(args.rollout), f"thirdperson_env{e}.mp4"
    )

    fig = plt.figure(figsize=(9, 5.5), facecolor="#111318")
    gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1], wspace=0.02)
    ax = fig.add_subplot(gs[0], projection="3d", facecolor="#111318")
    axg = fig.add_subplot(gs[1], facecolor="#15181f")

    writer = FFMpegWriter(fps=args.fps, bitrate=2400)
    print(f"env {e} on {terrain}: {len(list(steps))} frames -> {out}")

    with writer.saving(fig, out, dpi=110):
        for t in steps:
            ax.clear()
            axg.clear()
            base = body_pos[t, 0]

            # local terrain patch, coloured by height
            pts = scan[t]
            finite = np.isfinite(pts).all(1)
            pts = pts[finite]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis",
                           s=26, alpha=0.9, linewidths=0, depthshade=False)

            # where the base has been, so motion is visible in a still frame
            trail = body_pos[max(0, t - 120):t + 1, 0]
            if len(trail) > 1:
                ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], "-",
                        color="#7f8ea3", lw=1.2, alpha=0.75, zorder=3)

            # robot skeleton, one colour per leg, foot marker filled on contact
            for li, (leg, chain) in enumerate(chains):
                p = body_pos[t, chain]
                ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", color=LEG_COLORS[leg],
                        lw=3.0, ms=4.5, zorder=5)
                touching = contacts[t, li] > 0
                ax.scatter(*p[-1], s=110 if touching else 40,
                           c=LEG_COLORS[leg] if touching else "none",
                           edgecolors=LEG_COLORS[leg], linewidths=1.8, zorder=6)
            ax.scatter(*base, s=150, c=BASE_COLOR, marker="s", zorder=7)

            r = 0.52
            ax.set_xlim(base[0] - r, base[0] + r)
            ax.set_ylim(base[1] - r, base[1] + r)
            ax.set_zlim(base[2] - 0.42, base[2] + 0.26)
            ax.set_box_aspect((1, 1, 0.62))
            ax.view_init(elev=args.elev, azim=args.azim)
            ax.set_axis_off()
            ax.set_title(f"{terrain}   step {t}   x={base[0]:.2f} m",
                         color="#e8e8e8", fontsize=10, pad=2)

            # live expert gates
            if gates is not None:
                k = gates.shape[-1]
                axg.barh(np.arange(k), gates[t], color=EXPERT_COLORS[:k], height=0.6)
                axg.set_yticks(np.arange(k))
                axg.set_yticklabels([f"E{i}" for i in range(k)], color="#cfcfcf", fontsize=9)
                axg.set_xlim(0, 1)
                axg.invert_yaxis()
                axg.set_title("expert gates", color="#cfcfcf", fontsize=9)
                # spelled out, because the obvious wrong reading is that each
                # expert drives one leg
                axg.set_xlabel("whole-body mix\n(not per-leg)", color="#6f7684", fontsize=7.5)
            axg.tick_params(colors="#8a8a8a", labelsize=8)
            for s in axg.spines.values():
                s.set_color("#2a2f3a")

            writer.grab_frame()

    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
