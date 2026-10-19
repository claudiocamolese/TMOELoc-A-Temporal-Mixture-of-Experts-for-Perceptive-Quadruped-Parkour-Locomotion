"""Evaluate a trained policy per terrain type and dump the routing trace.

Reports the two metrics of the Extreme Parkour paper, split by terrain instead
of pooled:

  MXD  mean x-displacement, i.e. waypoints reached at episode end / num_goals
  MEV  mean edge violation, i.e. feet in contact within 5 cm of an edge, per step

Handles the teacher and both students. Students go through StudentPolicy, which
raises rather than silently falling back to the privileged teacher.

  python scripts/rsl_rl/evaluate_students.py \
      --task Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Eval-v0 \
      --checkpoint logs/.../model_15000.pt --num_envs 256 --steps 1500 --headless
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a parkour policy per terrain.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--steps", type=int, default=1500, help="Number of control steps.")
parser.add_argument(
    "--force_expert", type=int, default=None, help="Pin routing to one expert (MoE only)."
)
parser.add_argument("--vel_source", type=str, default=None, choices=["true", "predicted"])
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--tag", type=str, default=None, help="Name for the output files.")
parser.add_argument(
    "--policy",
    type=str,
    default="student",
    choices=["student", "teacher"],
    help="Evaluate the trained student, or the phase-1 teacher as the distillation ceiling.",
)
parser.add_argument(
    "--context_every", type=int, default=10, help="Store the context every N steps."
)
parser.add_argument(
    "--ablate_context",
    type=str,
    default=None,
    help="probe_directions.npz from fit_context_probe.py: project those directions "
    "out of the context, to test whether the actor uses that information.",
)
parser.add_argument(
    "--ablate_dims",
    type=str,
    default=None,
    help="comma-separated names to ablate (default: all in the file).",
)
parser.add_argument(
    "--attention",
    action="store_true",
    default=False,
    help="Record how much attention the newest token pays to each position of the "
    "temporal window. Answers whether the 0.8 s history is used at all.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
if args_cli.steps < 1:
    parser.error("--steps must be positive")
if args_cli.context_every < 1:
    parser.error("--context_every must be positive")
if args_cli.policy == "teacher" and (args_cli.ablate_context or args_cli.attention):
    parser.error("--ablate_context and --attention require a student policy")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# registers the gym tasks; `import parkour_tasks` alone hits an empty namespace
# package formed by the repo-root directory of the same name
import parkour_tasks.extreme_parkour_task.config.go2  # noqa: F401, E402
from scripts.rsl_rl.modules.on_policy_runner_moe import OnPolicyRunnerMoE  # noqa: E402
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import (  # noqa: E402
    OnPolicyRunnerWithExtractor,
)
from scripts.rsl_rl.modules.student_inference import StudentPolicy, TeacherPolicy  # noqa: E402
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: E402

from scripts.rsl_rl.modules.student_observations import CONTACT_SLICE, PERCEPTION_INTERVAL  # noqa: E402

RUNNER_CLASSES = {
    "OnPolicyRunnerWithExtractor": OnPolicyRunnerWithExtractor,
    "OnPolicyRunnerMoE": OnPolicyRunnerMoE,
}


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):  # noqa: C901
    """Evaluate an explicit teacher or student and save per-terrain metrics and traces."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.depth_camera.depth_cam.params["debug_vis"] = False

    tag = args_cli.tag or (
        "router" if args_cli.force_expert is None else f"expert{args_cli.force_expert}"
    )
    out_dir = args_cli.out_dir or os.path.join(os.path.dirname(args_cli.checkpoint), "eval")
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        cfg_dict = agent_cfg.to_dict()
        runner = RUNNER_CLASSES[agent_cfg.runner_class_name](
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        print(f"[INFO]: Loading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint, load_optimizer=False)
        policy_class = TeacherPolicy if args_cli.policy == "teacher" else StudentPolicy
        policy = policy_class(runner, cfg_dict, agent_cfg.device, vel_source=args_cli.vel_source)
        policy.force_expert(args_cli.force_expert)
        if args_cli.ablate_context:
            z = np.load(args_cli.ablate_context)
            names = args_cli.ablate_dims.split(",") if args_cli.ablate_dims else list(z.files)
            missing = [n for n in names if n not in z.files]
            if missing:
                raise SystemExit(
                    f"{args_cli.ablate_context} has no direction for {missing}; it holds {list(z.files)}"
                )
            policy.ablate_context(np.stack([z[n] for n in names]))
            print(f"[INFO]: ablated directions: {', '.join(names)}")
        print(
            f"[INFO]: variant={policy.variant} vel_source={policy.vel_source} forced_expert={args_cli.force_expert}"
        )

        parkour = env.unwrapped.parkour_manager.get_term("base_parkour")
        num_goals = float(parkour.num_goals)

        # The EVAL configs switch the mass and COM randomisation events off, so the
        # trace they produce has nothing for the dynamics probe to predict and its
        # R^2 is meaningless. Easy to miss, and easy to then misread as "the context
        # failed to learn the dynamics", so say it up front.
        ev = env.unwrapped.cfg.events
        disabled = [
            n
            for n in ("randomize_rigid_body_mass", "randomize_rigid_body_com")
            if getattr(ev, n, None) is None
        ]
        if disabled:
            print(
                f"[WARNING]: {', '.join(disabled)} disabled in this config, so those parameters are\n"
                "           constant across environments and the dynamics probe cannot be run on this\n"
                "           trace. Use the training task (no -Eval-/-Play- suffix) for the probe."
            )
        # The EVAL config carries the full reward set, but the *training* config
        # keeps only reward_collision. The training config is the one to use for the
        # dynamics probe, because EVAL disables the mass and COM randomisation
        # events and leaves nothing to predict, so MEV is reported only when the
        # edge term exists.
        try:
            feet_at_edge = env.unwrapped.reward_manager.get_term_cfg("reward_feet_edge").func
        except (ValueError, KeyError):
            feet_at_edge = None
            print("[INFO]: no reward_feet_edge term in this config; MEV will not be reported")

        obs, extras = env.get_observations()
        obs = obs.to(agent_cfg.device)
        depth = extras["observations"]["depth_camera"].to(agent_cfg.device)

        # per-episode outcomes, keyed by the terrain the episode was run on
        per_terrain: dict[str, dict[str, list]] = {}
        trace = {
            k: [] for k in ["gates", "contacts", "terrain", "goal_idx", "base_lin_vel", "dones"]
        }
        attention = []
        if args_cli.attention:
            enc = getattr(runner.alg, "depth_encoder", None)
            if not hasattr(enc, "capture_attention"):
                raise SystemExit("--attention only applies to the Transformer student")
            enc.capture_attention = True
        contexts, priv_latents, probe_terrain = [], [], []
        terrain_vocab: list[str] = []

        def terrain_codes():
            """Assign stable archive-local integer codes to the current terrain names."""
            names = np.asarray(parkour.env_per_terrain_name).reshape(-1)
            codes = np.empty(len(names), dtype=np.int16)
            for i, nm in enumerate(names):
                if nm not in terrain_vocab:
                    terrain_vocab.append(nm)
                codes[i] = terrain_vocab.index(nm)
            return names, codes

        for step in tqdm(range(args_cli.steps)):
            is_perception_step = env.unwrapped.common_step_counter % PERCEPTION_INTERVAL == 0
            actions = policy.act(obs, depth.clone(), is_perception_step)

            # capture before stepping: reset() reassigns terrain and zeroes the goal
            names_before, codes_before = terrain_codes()
            goal_before = parkour.cur_goal_idx.clone()

            obs, _, dones, extras = env.step(actions)
            obs = obs.to(agent_cfg.device)
            dones = dones.to(agent_cfg.device)
            depth = extras["observations"]["depth_camera"].to(agent_cfg.device)
            policy.reset(dones)

            edge = (
                feet_at_edge.feet_at_edge.sum(dim=1).float().cpu().numpy()
                if feet_at_edge is not None
                else np.full(env.num_envs, np.nan)
            )
            done_np = dones.cpu().numpy().astype(bool)
            goal_np = goal_before.cpu().numpy()

            for name in np.unique(names_before):
                sel = names_before == name
                bucket = per_terrain.setdefault(name, {"mxd": [], "mev": []})
                bucket["mev"].extend(edge[sel].tolist())
                finished = sel & done_np
                if finished.any():
                    bucket["mxd"].extend((goal_np[finished] / num_goals).tolist())

            gates = (
                policy.last_gates.cpu().numpy()
                if policy.last_gates is not None
                else np.zeros((env.num_envs, 1))
            )
            trace["gates"].append(gates.astype(np.float16))
            trace["contacts"].append(obs[:, CONTACT_SLICE].cpu().numpy().astype(np.float16))
            trace["terrain"].append(codes_before)
            trace["goal_idx"].append(goal_np.astype(np.int16))
            trace["base_lin_vel"].append(
                obs[:, policy.priv_explicit_slice][:, :3].cpu().numpy().astype(np.float16)
            )
            trace["dones"].append(done_np)

            # attention is only refreshed on a perception step; recording it on the
            # others would just duplicate the cached value
            if args_cli.attention and is_perception_step and enc.last_attention is not None:
                attention.append(enc.last_attention.cpu().numpy().astype(np.float16))
                trace_attn_phase = obs[:, CONTACT_SLICE].cpu().numpy().astype(np.float16)
                trace.setdefault("attn_contacts", []).append(trace_attn_phase)
                trace.setdefault("attn_terrain", []).append(codes_before)

            if step % args_cli.context_every == 0:
                contexts.append(policy.last_context.cpu().numpy().astype(np.float16))
                priv_latents.append(obs[:, 194:223].cpu().numpy().astype(np.float16))
                probe_terrain.append(codes_before)

        # ---------------------------------------------------------------- report
        summary = {
            "variant": policy.variant,
            "checkpoint": args_cli.checkpoint,
            "tag": tag,
            "terrains": {},
        }
        print(f"\n{'terrain':<18} | {'MXD':>16} | {'MEV':>16} | {'episodes':>8}")
        print("-" * 70)
        all_mxd, all_mev = [], []
        for name in sorted(per_terrain):
            mxd = np.asarray(per_terrain[name]["mxd"], dtype=float)
            mev = np.asarray(per_terrain[name]["mev"], dtype=float)
            all_mxd.extend(mxd.tolist())
            all_mev.extend(mev.tolist())
            row = {
                "mxd_mean": float(mxd.mean()) if mxd.size else float("nan"),
                "mxd_std": float(mxd.std()) if mxd.size else float("nan"),
                "mev_mean": float(mev.mean()) if mev.size else float("nan"),
                "mev_std": float(mev.std()) if mev.size else float("nan"),
                "episodes": int(mxd.size),
            }
            summary["terrains"][name] = row
            print(
                f"{name:<18} | {row['mxd_mean']:7.3f}+-{row['mxd_std']:<7.3f} | "
                f"{row['mev_mean']:7.3f}+-{row['mev_std']:<7.3f} | {row['episodes']:8d}"
            )
        a_mxd, a_mev = np.asarray(all_mxd), np.asarray(all_mev)
        summary["total"] = {
            "mxd_mean": float(a_mxd.mean()),
            "mxd_std": float(a_mxd.std()),
            "mev_mean": float(a_mev.mean()),
            "mev_std": float(a_mev.std()),
            "episodes": int(a_mxd.size),
        }
        print("-" * 70)
        print(
            f"{'TOTAL':<18} | {a_mxd.mean():7.3f}+-{a_mxd.std():<7.3f} | "
            f"{a_mev.mean():7.3f}+-{a_mev.std():<7.3f} | {a_mxd.size:8d}"
        )

        with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        extra = {"attention": np.asarray(attention)} if attention else {}
        np.savez_compressed(
            os.path.join(out_dir, f"trace_{tag}.npz"),
            terrain_vocab=np.asarray(terrain_vocab),
            context=np.asarray(contexts),
            priv_latent=np.asarray(priv_latents),
            probe_terrain=np.asarray(probe_terrain),
            **extra,
            **{k: np.asarray(v) for k, v in trace.items()},
        )
        print(f"\n[INFO]: wrote {out_dir}/metrics_{tag}.json and trace_{tag}.npz")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
