"""Run a trained distillation student and record what it sees and how it routes.

Works for both variants (GRU+dense baseline and Transformer+MoE) through the
shared StudentPolicy wrapper, which refuses to silently fall back to the teacher.

Third-person video needs the RTX renderer, which is unavailable on machines
without a Vulkan device (notably WSL2 without the NVIDIA ICD). This script does
not depend on it: it records the egocentric depth stream the policy actually
consumes, overlaid with the live expert gates, and dumps a per-step archive for
the offline analyses.

Examples
--------
  python scripts/rsl_rl/play_student.py \
      --task Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Play-v0 \
      --checkpoint logs/.../model_15000.pt --num_envs 4 --steps 600 --headless

  # what does expert 2 do on its own?
  python scripts/rsl_rl/play_student.py ... --force_expert 2
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a trained parkour student.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--steps", type=int, default=600, help="Number of control steps to run.")
parser.add_argument("--force_expert", type=int, default=None, help="Pin routing to this expert (MoE only).")
parser.add_argument("--vel_source", type=str, default=None, choices=["true", "predicted"])
parser.add_argument("--out_dir", type=str, default=None, help="Where to write the video and the archive.")
parser.add_argument("--no_video", action="store_true", default=False, help="Skip the depth video.")
parser.add_argument("--fps", type=int, default=50, help="Playback frame rate of the depth video.")
cli_args.add_rsl_rl_args(parser)  # provides --checkpoint, --logger, ...
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required: point it at a trained student model_*.pt")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# NB: `import parkour_tasks` alone is not enough. The repo root holds an outer
# `parkour_tasks/` directory with no __init__.py, so the name resolves to an
# empty namespace package and no task ever gets registered. The gym.register
# calls live in the go2 config package, so import that directly.
import parkour_tasks.extreme_parkour_task.config.go2  # noqa: F401, E402
from scripts.rsl_rl.modules.on_policy_runner_moe import OnPolicyRunnerMoE  # noqa: E402
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import (  # noqa: E402
    OnPolicyRunnerWithExtractor,
)
from scripts.rsl_rl.modules.student_inference import StudentPolicy  # noqa: E402
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: E402

RUNNER_CLASSES = {
    "OnPolicyRunnerWithExtractor": OnPolicyRunnerWithExtractor,
    "OnPolicyRunnerMoE": OnPolicyRunnerMoE,
}
EXPERT_COLORS = [(80, 160, 255), (90, 220, 120), (255, 150, 70), (220, 110, 220)]


def render_frame(depth: np.ndarray, gates: np.ndarray | None, scale: int = 5) -> np.ndarray:
    """Colourise one 58x87 depth frame and draw the expert gates beneath it."""
    img = np.clip(depth + 0.5, 0.0, 1.0)  # the env stores depth as value/max - 0.5
    img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    if gates is None:
        return img

    bar_h, pad = 46, 6
    panel = np.full((bar_h, img.shape[1], 3), 24, np.uint8)
    n = len(gates)
    slot = (img.shape[1] - pad * (n + 1)) // n
    for i, g in enumerate(gates):
        x0 = pad + i * (slot + pad)
        h = int(round(float(g) * (bar_h - 18)))
        cv2.rectangle(panel, (x0, bar_h - 4), (x0 + slot, bar_h - 4 - h), EXPERT_COLORS[i % 4], -1)
        cv2.putText(panel, f"E{i}", (x0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    return np.vstack([img, panel])


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # the depth debug window needs a display; the archive keeps the frames anyway
    env_cfg.observations.depth_camera.depth_cam.params["debug_vis"] = False

    out_dir = args_cli.out_dir or os.path.join(os.path.dirname(args_cli.checkpoint), "playback")
    os.makedirs(out_dir, exist_ok=True)
    tag = "router" if args_cli.force_expert is None else f"expert{args_cli.force_expert}"

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    cfg_dict = agent_cfg.to_dict()
    runner_class = RUNNER_CLASSES[agent_cfg.runner_class_name]
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO]: Loading student checkpoint from: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint, load_optimizer=False)

    policy = StudentPolicy(runner, cfg_dict, agent_cfg.device, vel_source=args_cli.vel_source)
    policy.force_expert(args_cli.force_expert)
    print(f"[INFO]: variant={policy.variant} vel_source={policy.vel_source} forced_expert={args_cli.force_expert}")

    parkour = env.unwrapped.parkour_manager.get_term("base_parkour")
    terrain_names = np.asarray(parkour.env_per_terrain_name).reshape(-1)

    obs, extras = env.get_observations()
    obs = obs.to(agent_cfg.device)
    depth = extras["observations"]["depth_camera"].to(agent_cfg.device)

    writer, rec = None, []
    log = {k: [] for k in ["gates", "context", "priv_latent", "base_lin_vel", "goal_idx", "contacts", "dones"]}

    for step in range(args_cli.steps):
        is_perception_step = env.unwrapped.common_step_counter % 5 == 0
        actions = policy.act(obs, depth.clone(), is_perception_step)

        gates = policy.last_gates[0].cpu().numpy() if policy.last_gates is not None else None
        if not args_cli.no_video:
            rec.append(render_frame(depth[0].cpu().numpy(), gates))

        log["gates"].append(policy.last_gates.cpu().numpy() if policy.last_gates is not None else np.zeros((env.num_envs, 1)))
        log["context"].append(policy.last_context.cpu().numpy())
        log["priv_latent"].append(obs[:, 194:223].cpu().numpy())
        log["base_lin_vel"].append(obs[:, policy.priv_explicit_slice][:, :3].cpu().numpy())
        log["goal_idx"].append(parkour.cur_goal_idx.cpu().numpy().copy())
        log["contacts"].append(obs[:, 49:53].cpu().numpy())

        obs, _, dones, extras = env.step(actions)
        obs = obs.to(agent_cfg.device)
        dones = dones.to(agent_cfg.device)
        depth = extras["observations"]["depth_camera"].to(agent_cfg.device)
        policy.reset(dones)
        log["dones"].append(dones.cpu().numpy())

        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{args_cli.steps}")

    if rec:
        h, w = rec[0].shape[:2]
        path = os.path.join(out_dir, f"depth_{tag}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), args_cli.fps, (w, h))
        for frame in rec:
            writer.write(frame)
        writer.release()
        print(f"[INFO]: wrote {path}  ({len(rec)} frames, {w}x{h})")

    arch = os.path.join(out_dir, f"rollout_{tag}.npz")
    np.savez_compressed(arch, terrain_names=terrain_names, **{k: np.asarray(v) for k, v in log.items()})
    print(f"[INFO]: wrote {arch}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
