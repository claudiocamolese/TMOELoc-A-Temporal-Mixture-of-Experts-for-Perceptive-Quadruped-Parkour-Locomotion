"""Plot what the Transformer attends to across its temporal window.

Three panels, each answering one question:

  1  Does the newest token use the 0.8 s window, or only the current frame?
     Attention per position, one line per layer, against the uniform reference.
  2  Does the pattern depend on what the robot is doing?
     Attention as a phase x position grid.
  3  How far back does each phase look?
     Effective lookback in milliseconds, per gait phase.

  python scripts/analysis/plot_attention.py                    # newest trace
  python scripts/analysis/plot_attention.py <trace_*.npz> --dark
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import matplotlib.patches  # noqa: E402

STEP_MS = 100.0  # one token per perception step, 10 Hz

# Validated palette (see the data-viz reference). Two categorical slots for the
# two layers; a single sequential blue ramp for magnitude. Both modes are
# selected, not flipped: each was validated against its own surface.
THEME = {
    "light": dict(
        surface="#fcfcfb", panel="#fcfcfb",
        ink="#0b0b0b", ink2="#52514e", ink3="#8a8880",
        grid="#e6e5e1", series=("#2a78d6", "#eb6834"), ref="#9c9a92",
        seq=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        bar="#2a78d6",
    ),
    "dark": dict(
        surface="#1a1a19", panel="#1a1a19",
        ink="#ffffff", ink2="#c3c2b7", ink3="#8a8880",
        grid="#33332f", series=("#3987e5", "#d95926"), ref="#6f6e68",
        seq=["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        bar="#3987e5",
    ),
}


def find_trace(path: str | None) -> str:
    if path and os.path.isfile(path):
        return path
    cands = glob.glob(os.path.join(path or "logs", "**", "trace_*.npz"), recursive=True)
    good = [c for c in cands if "attention" in np.load(c, allow_pickle=True).files]
    if not good:
        raise SystemExit(
            "no trace with recorded attention.\nRecord one with:\n"
            "  python scripts/rsl_rl/evaluate_students.py --task <...-Eval-v0> \\\n"
            "      --checkpoint <model_*.pt> --attention --num_envs 192 --steps 800 --headless"
        )
    return max(good, key=os.path.getmtime)


def rounded_barh(ax, y, width, color, height=0.5):
    """Horizontal bar anchored at zero, rounded at the data end only.

    Drawn as a rectangle plus a half-disc so the baseline stays square: a
    symmetric rounded box would round the zero end too and detach the bar from
    its axis.
    """
    r = height / 2
    body = max(width - r, 1e-9)
    ax.add_patch(plt.Rectangle((0, y - r), body, height, linewidth=0,
                               facecolor=color, zorder=3))
    ax.add_patch(matplotlib.patches.Wedge((body, y), r, -90, 90, linewidth=0,
                                          facecolor=color, zorder=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dark", action="store_true", help="render for a dark surface")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    t = THEME["dark" if args.dark else "light"]
    path = find_trace(args.trace)
    print(f"reading {path}")
    d = np.load(path, allow_pickle=True)
    a = d["attention"].astype(np.float32)                 # (T, N, layers, H)
    contacts = d["attn_contacts"].astype(np.float32)      # (T, N, 4)
    T, N, L, H = a.shape
    idx_ms = np.arange(H)[::-1] * STEP_MS                 # 700 … 0 ms into the past
    labels = [f"−{int(v)}" if v else "now" for v in idx_ms]
    uniform = 1.0 / H

    n_contact = (contacts > 0).sum(-1)
    phases = [("flight\n0 feet", n_contact == 0),
              ("push / land\n1 foot", n_contact == 1),
              ("gait\n2–3 feet", (n_contact >= 2) & (n_contact <= 3)),
              ("stance\n4 feet", n_contact == 4)]
    phases = [(nm, s) for nm, s in phases if s.sum() >= 50]

    fig = plt.figure(figsize=(14.6, 4.6), facecolor=t["surface"])
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 0.9], wspace=0.62,
                          left=0.052, right=0.985, top=0.79, bottom=0.16)

    def style(ax):
        ax.set_facecolor(t["panel"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["grid"])
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(colors=t["ink2"], labelsize=9, length=3, width=1.0)

    # ---------------------------------------------------------------- panel 1
    ax1 = fig.add_subplot(gs[0]); style(ax1)
    ax1.grid(axis="y", color=t["grid"], linewidth=1.0, zorder=0)
    ax1.set_axisbelow(True)
    x = np.arange(H)
    ax1.axhline(uniform, color=t["ref"], linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    # anchored left, away from the direct labels that sit at the right edge
    ax1.annotate("uniform", (0, uniform), textcoords="offset points", xytext=(0, 6),
                 color=t["ink3"], fontsize=8.5, ha="left")
    ends = [a[:, :, l, :].reshape(-1, H).mean(0)[-1] for l in range(L)]
    # the axis top has to be known before the labels are placed, so it is
    # computed once here and reused by set_ylim below
    span = max(0.28, float(np.percentile(a.reshape(-1, L, H)[:, 0, :], 75, axis=0).max() * 1.25))
    # the two layers can finish within a hair of each other (and of the uniform
    # line); nudge the direct labels apart rather than letting them overprint
    label_y = list(ends)
    if L == 2 and abs(ends[0] - ends[1]) < 0.05 * span:
        mid = sum(ends) / 2
        label_y = [mid + 0.035 * span, mid - 0.035 * span]
        if ends[0] < ends[1]:
            label_y = label_y[::-1]

    for l in range(L):
        flat = a[:, :, l, :].reshape(-1, H)
        m = flat.mean(0)
        # The mean alone is misleading: individual steps peak at different
        # positions, so averaging them flattens the curve towards uniform. The
        # band is the interquartile range across steps, i.e. the structure that
        # the mean hides. Replacing this attention with a uniform average is
        # what the ablation measures, and it costs 3.8x the falls.
        lo, hi = np.percentile(flat, [25, 75], axis=0)
        ax1.fill_between(x, lo, hi, color=t["series"][l % 2], alpha=0.16,
                         linewidth=0, zorder=3)
        ax1.plot(x, m, color=t["series"][l % 2], linewidth=2.0, marker="o",
                 markersize=5.5, markeredgecolor=t["panel"], markeredgewidth=1.4, zorder=4)
        ax1.annotate(f"layer {l}", (x[-1], label_y[l]), textcoords="offset points",
                     xytext=(7, 0), color=t["series"][l % 2], fontsize=9.5,
                     fontweight="medium", va="center")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_xlim(-0.45, H - 0.45)
    ax1.set_ylim(0, span)
    ax1.set_xlabel("position in the window (ms into the past)", color=t["ink2"], fontsize=9.5)
    ax1.set_ylabel("attention weight", color=t["ink2"], fontsize=9.5)
    ax1.set_title("Attention of the newest token   (line = mean, band = IQR)",
                  color=t["ink"], fontsize=11.5, fontweight="semibold", loc="left", pad=10)

    # ---------------------------------------------------------------- panel 2
    ax2 = fig.add_subplot(gs[1]); style(ax2)
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    grid = np.stack([a[s].reshape(-1, H).mean(0) for _, s in phases])
    im = ax2.imshow(grid, aspect="auto", cmap=cmap, origin="upper",
                    vmin=0, vmax=grid.max())
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_yticks(range(len(phases)))
    ax2.set_yticklabels([nm.replace("\n", " ") for nm, _ in phases], fontsize=9)
    # a 2 px surface gap between cells, per the mark spec
    ax2.set_xticks(np.arange(-0.5, H, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(phases), 1), minor=True)
    ax2.grid(which="minor", color=t["panel"], linewidth=2.0)
    ax2.tick_params(which="minor", length=0)
    for i in range(grid.shape[0]):
        j = int(np.argmax(grid[i]))
        ax2.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                 fontsize=8.5, color="#ffffff" if grid[i, j] > grid.max() * 0.55 else t["ink"],
                 fontweight="semibold")
    cb = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.025)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["ink2"], labelsize=8, length=2)
    # the label goes above the bar: as a rotated ylabel it collided with the
    # tick labels of the next panel
    cb.ax.set_title("attn", color=t["ink2"], fontsize=8, pad=6)
    ax2.set_xlabel("position in the window (ms into the past)", color=t["ink2"], fontsize=9.5)
    ax2.set_title("…by gait phase   (peak cell labelled)", color=t["ink"], fontsize=11.5,
                  fontweight="semibold", loc="left", pad=10)

    # ---------------------------------------------------------------- panel 3
    ax3 = fig.add_subplot(gs[2]); style(ax3)
    ax3.grid(axis="x", color=t["grid"], linewidth=1.0, zorder=0)
    ax3.set_axisbelow(True)
    lookback = [float((a[s].reshape(-1, H).mean(0) * idx_ms).sum()) for _, s in phases]
    order = np.argsort(lookback)
    for k, i in enumerate(order):
        rounded_barh(ax3, k, lookback[i], t["bar"])
        # inside the bar: outside, the labels crossed the uniform reference line
        ax3.text(lookback[i] - 12, k, f"{lookback[i]:.0f} ms", va="center", ha="right",
                 fontsize=9.5, color="#ffffff", fontweight="medium", zorder=4)
    ax3.axvline(idx_ms.mean(), color=t["ref"], linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax3.set_yticks(range(len(order)))
    ax3.set_yticklabels([phases[i][0].replace("\n", " ") for i in order], fontsize=9)
    ax3.set_ylim(-0.62, len(phases) - 0.15)
    ax3.set_xlim(0, max(max(lookback), idx_ms.mean()) * 1.16)
    ax3.annotate("uniform", (idx_ms.mean(), len(phases) - 0.32), xytext=(4, 0),
                 textcoords="offset points", color=t["ink3"], fontsize=8.5, va="center")
    ax3.set_xlabel("effective lookback (ms)", color=t["ink2"], fontsize=9.5)
    ax3.set_title("How far back each phase looks", color=t["ink"], fontsize=11.5,
                  fontweight="semibold", loc="left", pad=10)

    older = a.reshape(-1, H).mean(0)[: H - 2].sum()
    u = 1.0 / H
    fl = a.reshape(-1, H)
    kl_sample = float((np.clip(fl, 1e-9, 1) * np.log(np.clip(fl, 1e-9, 1) / u)).sum(-1).mean())
    mdist = fl.mean(0)
    kl_mean = float((mdist * np.log(mdist / u)).sum())
    sub = (f"window {H} tokens = {H * STEP_MS / 1000:.1f} s at 10 Hz   ·   "
           f"{100 * older:.0f}% of attention older than 200 ms   ·   "
           f"per-step divergence from uniform is {kl_sample / max(kl_mean, 1e-9):.1f}× "
           f"the mean curve's   ·   {T} × {N} samples")
    fig.suptitle(args.title or "Temporal attention of the context encoder",
                 color=t["ink"], fontsize=13.5, fontweight="semibold", x=0.055, ha="left", y=0.965)
    fig.text(0.055, 0.895, sub, color=t["ink2"], fontsize=9.5, ha="left")

    dst = args.out or os.path.join(os.path.dirname(path),
                                   "attention" + ("_dark" if args.dark else "") + ".png")
    fig.savefig(dst, dpi=170, facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote {dst}")

    # the table view: identity is never colour-alone
    print(f"\n{'phase':<16} | {'lookback':>9} | " + " | ".join(f"{l:>6}" for l in labels))
    for (nm, s), lb in zip(phases, lookback):
        m = a[s].reshape(-1, H).mean(0)
        print(f"{nm.replace(chr(10), ' '):<16} | {lb:8.0f}ms | " + " | ".join(f"{v:6.3f}" for v in m))


if __name__ == "__main__":
    main()
