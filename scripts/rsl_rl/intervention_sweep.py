"""Run one checkpoint under a series of controlled deletions, in one simulator.

Each condition removes exactly one pathway and leaves every weight alone, so the
difference against the reference run is attributable to that pathway. The point
is to find out which parts of the architecture are doing work and which are
decoration -- the input to deciding what to change next.

Everything runs inside a single ``gym.make``: building the environment costs
about a minute, the conditions cost about two, so re-launching per condition
would dominate the run. Environments keep their terrain assignment for the whole
sweep, which makes the comparison paired -- condition A and condition B are
scored on the same 192 terrains -- and the bootstrap resamples environments to
match.

  python scripts/rsl_rl/intervention_sweep.py --checkpoint <model_*.pt> --headless
  python scripts/rsl_rl/intervention_sweep.py -c <ckpt> --steps 1200 \
      --conditions none,no_depth,uniform_attn,uniform_gates --headless

Writes ``sweep.json`` and one ``trace_<condition>.npz`` per condition next to the
checkpoint, so ``episode_metrics.py`` can be re-run on any of them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

ALL_CONDITIONS = [
    "none",           # reference
    "zero_context",   # does the actor use the encoder at all
    "no_depth",       # is the context vision, or proprioceptive memory
    "uniform_attn",   # is the attention better than a mean pool
    "uniform_gates",  # is the routing better than a plain ensemble
    "dense",          # does sparsity cost anything
    "top1",           # is the second expert earning its keep
    "expert0", "expert1", "expert2", "expert3",
]

parser = argparse.ArgumentParser(description="Ablate one pathway at a time and measure the cost.")
parser.add_argument("--task", type=str, default="Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Eval-v0")
parser.add_argument("--num_envs", type=int, default=192)
parser.add_argument("--steps", type=int, default=1000, help="control steps per condition")
parser.add_argument("--warmup", type=int, default=50,
                    help="steps discarded after switching condition, while the "
                         "robots settle out of the previous controller's state")
parser.add_argument("--conditions", type=str, default=",".join(ALL_CONDITIONS))
parser.add_argument("--out_dir", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# registers the gym tasks; `import parkour_tasks` alone hits an empty namespace
# package formed by the repo-root directory of the same name
import parkour_tasks.extreme_parkour_task.config.go2  # noqa: F401, E402
from scripts.analysis.episode_metrics import (  # noqa: E402
    CONTROL_DT, NUM_GOALS, _fall_hazard, _goals_per_min, bootstrap_over_envs, episode_table,
)
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


def paired_delta(rows_a, rows_b, stat, n_boot=2000, seed=0):
    """Bootstrap the difference b - a, resampling the shared environment set once.

    The two conditions were scored on the same environments, so resampling them
    jointly cancels the terrain draw -- the dominant source of variance here --
    and leaves a much tighter interval than two independent bootstraps.
    """
    rng = np.random.default_rng(seed)
    a_by, b_by = {}, {}
    for r in rows_a:
        a_by.setdefault(r["env"], []).append(r)
    for r in rows_b:
        b_by.setdefault(r["env"], []).append(r)
    envs = sorted(set(a_by) | set(b_by))
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(envs), len(envs))
        sa = [r for j in pick for r in a_by.get(envs[j], [])]
        sb = [r for j in pick for r in b_by.get(envs[j], [])]
        vals[i] = (stat(sb) if sb else np.nan) - (stat(sa) if sa else np.nan)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    conditions = [c.strip() for c in args_cli.conditions.split(",") if c.strip()]
    bad = [c for c in conditions if c not in ALL_CONDITIONS]
    if bad:
        raise SystemExit(f"unknown condition(s): {bad}\nknown: {ALL_CONDITIONS}")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.depth_camera.depth_cam.params["debug_vis"] = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # the runner consumes its config dict in place (rsl-rl pops "class_name" out
    # of the algorithm block), so it gets its own copy
    cfg_dict = agent_cfg.to_dict()
    runner = RUNNER_CLASSES[agent_cfg.runner_class_name](
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = StudentPolicy(runner, cfg_dict, agent_cfg.device)
    if not policy.is_moe:
        # the gate and expert conditions are meaningless for the dense student;
        # the perception ones still are, so allow it but say what is dropped
        skipped = [c for c in conditions if c.startswith("expert") or c in ("uniform_gates", "dense", "top1")]
        conditions = [c for c in conditions if c not in skipped]
        if skipped:
            print(f"[INFO]: {policy.variant} has no router; skipping {skipped}")

    out_dir = args_cli.out_dir or os.path.join(os.path.dirname(args_cli.checkpoint), "sweep")
    os.makedirs(out_dir, exist_ok=True)
    parkour = env.unwrapped.parkour_manager.get_term("base_parkour")

    obs, extras = env.get_observations()
    obs = obs.to(agent_cfg.device)
    depth = extras["observations"]["depth_camera"].to(agent_cfg.device)

    results, tables = {}, {}
    for cond in conditions:
        policy.set_intervention(cond)
        # clear temporal state so the previous condition's window does not leak in
        policy.reset(torch.ones(env.num_envs, dtype=torch.bool, device=agent_cfg.device))
        rec = {k: [] for k in ["dones", "goal_idx", "terrain", "gates"]}
        vocab: list[str] = []

        for step in tqdm(range(args_cli.steps + args_cli.warmup), desc=f"{cond:<14}"):
            is_perception_step = env.unwrapped.common_step_counter % 5 == 0
            actions = policy.act(obs, depth.clone(), is_perception_step)
            names = np.asarray(parkour.env_per_terrain_name).reshape(-1)
            codes = np.empty(len(names), dtype=np.int16)
            for i, nm in enumerate(names):
                if nm not in vocab:
                    vocab.append(nm)
                codes[i] = vocab.index(nm)
            goal = parkour.cur_goal_idx.clone().cpu().numpy().astype(np.int16)

            obs, _, dones, extras = env.step(actions)
            obs = obs.to(agent_cfg.device)
            dones = dones.to(agent_cfg.device)
            depth = extras["observations"]["depth_camera"].to(agent_cfg.device)
            policy.reset(dones)

            if step >= args_cli.warmup:
                rec["dones"].append(dones.cpu().numpy().astype(bool))
                rec["goal_idx"].append(goal)
                rec["terrain"].append(codes)
                g = policy.last_gates
                rec["gates"].append(g.cpu().numpy().astype(np.float16) if g is not None
                                    else np.zeros((env.num_envs, 1), np.float16))

        arrs = {k: np.asarray(v) for k, v in rec.items()}
        np.savez_compressed(os.path.join(out_dir, f"trace_{cond}.npz"),
                            terrain_vocab=np.asarray(vocab), **arrs)
        rows = episode_table(arrs["dones"], arrs["goal_idx"].astype(int))
        tables[cond] = rows
        gph, gse = bootstrap_over_envs(rows, _goals_per_min, 1000)
        fhz, fse = bootstrap_over_envs(rows, _fall_hazard, 1000)
        results[cond] = {
            "goals_per_min": _goals_per_min(rows), "goals_per_min_se": gph,
            "fall_hazard_per_1k": _fall_hazard(rows), "fall_hazard_se": fhz,
            "segments": len(rows),
            "censored_frac": float(np.mean([r["censored"] for r in rows])),
            "mean_gate": arrs["gates"].astype(np.float32).mean(axis=(0, 1)).tolist(),
        }
        r = results[cond]
        print(f"  {cond:<14} goals/min {r['goals_per_min']:6.2f}+-{gph:4.2f}   "
              f"fall hazard {r['fall_hazard_per_1k']:5.3f}+-{fhz:5.3f}")

    env.close()

    # ------------------------------------------------------------------ report
    ref = "none" if "none" in tables else conditions[0]
    print(f"\n{'condition':<15} | {'goals/env-min':>20} | {'fall hazard /1k':>21} | verdict")
    print("-" * 92)
    for cond in conditions:
        r = results[cond]
        if cond == ref:
            print(f"{cond + ' (ref)':<15} | {r['goals_per_min']:11.2f} +- {r['goals_per_min_se']:<5.2f} | "
                  f"{r['fall_hazard_per_1k']:12.3f} +- {r['fall_hazard_se']:<5.3f} | reference")
            continue
        dg, dgs = paired_delta(tables[ref], tables[cond], _goals_per_min)
        df, dfs = paired_delta(tables[ref], tables[cond], _fall_hazard)
        sg = abs(dg) / dgs if dgs else 0.0
        sf = abs(df) / dfs if dfs else 0.0
        faster, safer = dg > 0 and sg >= 2, df < 0 and sf >= 2
        slower, riskier = dg < 0 and sg >= 2, df > 0 and sf >= 2
        if sg < 2 and sf < 2:
            verdict = "no effect -- this pathway is not load-bearing"
        elif (faster and riskier) or (slower and safer):
            # the two metrics disagree, which is a trade and not a verdict:
            # calling it "degrades" would hide a real +20% in throughput
            verdict = f"trades ({dg:+.1f} pace / {df:+.2f} falls)"
        elif slower or riskier:
            verdict = f"degrades ({sg:.1f}s pace, {sf:.1f}s falls)"
        else:
            verdict = f"IMPROVES ({sg:.1f}s pace, {sf:.1f}s falls)"
        print(f"{cond:<15} | {dg:+11.2f} +- {dgs:<5.2f} | {df:+12.3f} +- {dfs:<5.3f} | {verdict}")
        results[cond].update(d_goals=dg, d_goals_se=dgs, d_hazard=df, d_hazard_se=dfs)

    print("\nDeltas are paired: the same environments and terrains under both conditions.")
    print(f"Reference condition: {ref}. {args_cli.steps} steps x {args_cli.num_envs} envs each.")
    results["_meta"] = {"checkpoint": args_cli.checkpoint, "steps": args_cli.steps,
                        "num_envs": args_cli.num_envs, "reference": ref,
                        "control_dt": CONTROL_DT, "num_goals": NUM_GOALS}
    with open(os.path.join(out_dir, "sweep.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out_dir}/sweep.json")


if __name__ == "__main__":
    main()
    simulation_app.close()
