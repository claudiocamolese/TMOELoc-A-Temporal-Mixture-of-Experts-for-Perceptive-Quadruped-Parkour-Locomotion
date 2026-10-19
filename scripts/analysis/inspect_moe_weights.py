"""What can be read off a MoE checkpoint without running the simulator.

Two questions answerable from weights alone:

  1. What does the router look at?  The router input is [proprioception(53);
     context(128); estimated velocity(9)], so the first-layer weight magnitude
     per input block says which modality drives the routing decision.
  2. Are the experts functionally distinct yet, or still near-copies?

What weights *cannot* tell you is which terrain or gait phase each expert
handles. That needs a rollout: see evaluate_students.py + analyze_student.py.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

# proprioception layout, from ExtremeParkourObservations
PROP_BLOCKS = [
    ("base ang vel", 0, 3),
    ("imu roll/pitch", 3, 5),
    ("delta yaw", 5, 8),
    ("command", 8, 11),
    ("env class (masked)", 11, 13),
    ("joint pos", 13, 25),
    ("joint vel", 25, 37),
    ("prev action", 37, 49),
    ("foot contacts", 49, 53),
]


def main():
    """Inspect router input weights and expert diversity without launching the simulator."""
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--num_prop", type=int, default=53)
    ap.add_argument("--context_dim", type=int, default=128)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["depth_actor_state_dict"]
    print(f"checkpoint iteration: {ckpt.get('iter')}")

    w_in = sd["router.0.weight"]  # (hidden, in_features)
    in_features = w_in.shape[1]
    n_prop, n_ctx = args.num_prop, args.context_dim

    # Mean |w| per input unit, so blocks of different width stay comparable.
    per_unit = w_in.abs().mean(0)
    groups = [
        ("proprioception", 0, n_prop),
        ("context c_t", n_prop, n_prop + n_ctx),
        ("base velocity", n_prop + n_ctx, in_features),
    ]
    print("\n=== what the router reads (mean |weight| per input unit) ===")
    total = sum(float(per_unit[a:b].sum()) for _, a, b in groups)
    for name, a, b in groups:
        share = 100 * float(per_unit[a:b].sum()) / total
        print(
            f"  {name:<16} {float(per_unit[a:b].mean()):.4f} per unit   {share:5.1f}% of total influence"
        )

    print("\n=== within proprioception ===")
    for name, a, b in PROP_BLOCKS:
        print(f"  {name:<20} {float(per_unit[a:b].mean()):.4f}")

    # ------------------------------------------------ expert distinctness
    ws = [
        sd[f"experts.weights.{i}"]
        for i in range(len([k for k in sd if k.startswith("experts.weights.")]))
    ]
    bs = [sd[f"experts.biases.{i}"] for i in range(len(ws))]
    k = ws[0].shape[0]
    print(f"\n=== expert distinctness ({k} experts) ===")

    # Random probe inputs: this measures whether the *functions* differ, not
    # whether the weight tensors differ, which permutation symmetry would make
    # meaningless. The input distribution is generic, so read the numbers as a
    # lower bound on distinctness.
    torch.manual_seed(0)
    u = torch.randn(4096, in_features)
    h = u.unsqueeze(0).expand(k, -1, -1)
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = torch.baddbmm(b, h, w)
        if i < len(ws) - 1:
            h = torch.nn.functional.elu(h)
    out = h  # (k, batch, num_actions)

    flat = out.reshape(k, -1)
    flat = flat - flat.mean(1, keepdim=True)
    norm = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
    corr = (norm @ norm.T).numpy()
    print("pairwise output correlation (1.0 = redundant experts):")
    print("      " + "".join(f"{'E' + str(j):>8}" for j in range(k)))
    for i in range(k):
        print(f"  E{i}  " + "".join(f"{corr[i, j]:8.3f}" for j in range(k)))
    off = corr[~np.eye(k, dtype=bool)]
    print(
        f"mean off-diagonal correlation: {off.mean():.3f}  "
        f"({'experts are near-copies' if off.mean() > 0.9 else 'experts compute different functions'})"
    )

    scale = out.reshape(k, -1).std(1).numpy()
    print("\nper-expert output scale (std over probe inputs):")
    for i in range(k):
        print(f"  E{i}: {scale[i]:.4f}")

    # Which joints does each expert move most? Rows are experts, columns joints.
    per_joint = out.std(1).numpy()  # (k, 12)
    names = [
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
    ]
    print("\nper-expert output magnitude by joint (relative within expert):")
    print(f"  {'joint':<10}" + "".join(f"{'E' + str(i):>8}" for i in range(k)))
    rel = per_joint / (per_joint.sum(1, keepdims=True) + 1e-8)
    for j, nm in enumerate(names[: per_joint.shape[1]]):
        print(f"  {nm:<10}" + "".join(f"{100 * rel[i, j]:7.1f}%" for i in range(k)))


if __name__ == "__main__":
    main()
