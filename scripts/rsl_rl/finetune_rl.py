"""Refine a distilled MoE actor with PPO and a privileged training-only critic.

Perception is frozen: replay stores actor input vectors and original routing
inputs, not depth images. The critic receives 300 warmup iterations by default.
The inherited reward degraded traversal in the recorded experiment; this entry
point preserves that procedure for investigation, not as a recommended upgrade.
See CONCLUSIONS.md for experimental motivation, measurements, and limitations.

Run with a StudentMoE/StudentMoEDense -FT- task and a distilled checkpoint.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import deque

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="PPO fine-tuning of a distilled student.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=192)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--steps_per_env", type=int, default=24)
parser.add_argument("--lr", type=float, default=3.0e-4)
parser.add_argument(
    "--init_noise_std",
    type=float,
    default=0.5,
    help="Exploration around the distilled actions. Large values throw away "
    "the behaviour distillation just bought.",
)
parser.add_argument("--entropy_coef", type=float, default=0.002)
parser.add_argument("--clip_param", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument(
    "--kl_target",
    type=float,
    default=None,
    help="Left unset it is derived from the noise, because the two are not "
    "independent: the Gaussian KL carries a (mu_old-mu_new)^2 / 2*sigma^2 "
    "term, so halving sigma quadruples the KL reported for an identical "
    "step. A fixed target then pins the learning rate to its floor. The "
    "default is calibrated at sigma 0.5 and scaled by (0.5/sigma)^2.",
)
parser.add_argument(
    "--unfreeze_encoder",
    action="store_true",
    default=False,
    help="Unsupported with vector-only replay; rejected before simulation starts.",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=None,
    help="Override the mixture's top-k at the start of fine-tuning. "
    "top_k = num_experts is soft routing. Switching here rather than "
    "in a fresh distillation run costs nothing: PPO retrains the actor "
    "anyway, so the router and the experts reorganise for the new regime "
    "under the objective that actually matters.",
)
parser.add_argument(
    "--balance_coef",
    type=float,
    default=0.02,
    help="Small load-balancing term. Distillation had one; PPO does not, and "
    "without it nothing stops the gates collapsing onto one expert.",
)
parser.add_argument(
    "--critic_warmup",
    type=int,
    default=300,
    help="Iterations spent training only the value function, with the policy "
    "held fixed. The critic is seeded from the teacher's, which was fitted "
    "on the teacher's state distribution, not this student's; until it is "
    "refitted the advantages are noise, and PPO spends thousands of Adam "
    "steps on that noise. Measured without it: pace fell from 24.25 to 8.38 "
    "within 250 iterations.",
)
parser.add_argument("--save_interval", type=int, default=250)
# --run_name comes from cli_args.add_rsl_rl_args; do not redefine it
parser.add_argument("--log_dir", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required: fine-tuning starts from a distilled policy")
if args_cli.unfreeze_encoder:
    parser.error("--unfreeze_encoder is unsupported: replay stores features, not encoder inputs")
if args_cli.init_noise_std <= 0:
    parser.error("--init_noise_std must be positive")
if args_cli.top_k is not None and args_cli.top_k < 1:
    parser.error("--top_k must be at least 1")
for name in (
    "num_envs",
    "max_iterations",
    "steps_per_env",
    "epochs",
    "minibatches",
    "save_interval",
):
    if getattr(args_cli, name) < 1:
        parser.error(f"--{name} must be positive")
if "StudentMoE" not in args_cli.task:
    parser.error("PPO refinement requires a Transformer/MoE student task")
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.kl_target is None:
    args_cli.kl_target = 0.02 * (0.5 / max(args_cli.init_noise_std, 1e-6)) ** 2
    print(
        f"[INFO]: kl_target derived from noise {args_cli.init_noise_std}: {args_cli.kl_target:.4f}"
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.distributions import Normal  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import parkour_tasks.extreme_parkour_task.config.go2  # noqa: F401, E402
from scripts.rsl_rl.modules.on_policy_runner_moe import OnPolicyRunnerMoE  # noqa: E402
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import (  # noqa: E402
    OnPolicyRunnerWithExtractor,
)
from scripts.rsl_rl.modules.student_inference import StudentPolicy  # noqa: E402
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: E402

from scripts.rsl_rl.modules.student_observations import PERCEPTION_INTERVAL, YAW_SLICE  # noqa: E402

from scripts.rsl_rl.modules.ppo_utils import (  # noqa: E402
    Critic,
    flatten_rollout,
    generalized_advantage_estimate,
)

RUNNER_CLASSES = {
    "OnPolicyRunnerWithExtractor": OnPolicyRunnerWithExtractor,
    "OnPolicyRunnerMoE": OnPolicyRunnerMoE,
}


class ActorAdapter:
    """Replay a frozen-perception MoE actor using flat input and routing vectors.

    The current input feeds the expert MLPs. The stored perception-step input
    feeds the router, preserving the distribution used to sample PPO actions.
    """

    def __init__(self, policy: StudentPolicy):
        self.policy = policy

    @property
    def parameters(self) -> list[nn.Parameter]:
        """Return actor parameters without including the frozen encoder."""
        return list(self.policy.depth_actor.parameters())

    def rollout_input(self, obs: torch.Tensor, prop_actor: torch.Tensor) -> torch.Tensor:
        """Collect the exact input blocks consumed by the configured MoE actor."""
        velocity = (
            obs[:, self.policy.priv_explicit_slice]
            if self.policy.vel_source == "true"
            else self.policy.vel_pred
        )
        return self.policy.depth_actor.build_input(prop_actor, self.policy.context, velocity)

    def mean_action(
        self, inputs: torch.Tensor, update_routing: bool, route_u: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Evaluate experts on current inputs and gates on the original routing inputs."""
        prop_end = self.policy.num_prop
        context_end = prop_end + self.policy.depth_actor.context_dim
        return self.policy.depth_actor(
            inputs[:, :prop_end],
            inputs[:, prop_end:context_end],
            vel=inputs[:, context_end:],
            update_routing=update_routing,
            route_u=route_u,
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Refine a distilled MoE actor with frozen perception and a privileged value function."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.depth_camera.depth_cam.params["debug_vis"] = False
    device = agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    run = None

    try:
        active = [t for t in env.unwrapped.reward_manager.active_terms]
        print(
            f"[INFO]: {len(active)} reward terms active: {', '.join(active[:6])}"
            + (" ..." if len(active) > 6 else "")
        )
        if len(active) <= 1:
            raise SystemExit(
                "This task has no task reward, so there is nothing for PPO to optimise.\n"
                "Fine-tuning needs the full reward set -- use a -FT- task whose env cfg "
                "carries TeacherRewardsCfg, not the distillation config."
            )

        cfg_dict = agent_cfg.to_dict()
        runner = RUNNER_CLASSES[agent_cfg.runner_class_name](
            env, agent_cfg.to_dict(), log_dir=None, device=device
        )
        runner.load(args_cli.checkpoint, load_optimizer=False)
        policy = StudentPolicy(runner, cfg_dict, device)
        if not policy.is_moe:
            raise ValueError("PPO refinement currently supports only Transformer/MoE students")
        actor = ActorAdapter(policy)
        print(f"[INFO]: fine-tuning {policy.variant} from {args_cli.checkpoint}")

        if args_cli.top_k is not None and policy.is_moe:
            old = policy.depth_actor.top_k
            policy.depth_actor.top_k = min(args_cli.top_k, policy.depth_actor.num_experts)
            print(
                f"[INFO]: top_k {old} -> {policy.depth_actor.top_k}"
                + (
                    " (soft routing)"
                    if policy.depth_actor.top_k == policy.depth_actor.num_experts
                    else ""
                )
            )

        if policy.is_moe and policy.depth_actor.router_noise_std > 0.0:
            # In PPO the exploration is the action distribution. Noisy top-k gating
            # on top of it makes the policy irreproducible: replaying a stored
            # transition draws different gates, so the recomputed mean differs from
            # the one that was acted on and the probability ratio is wrong before a
            # single gradient step is taken.
            print(
                f"[INFO]: router noise {policy.depth_actor.router_noise_std} -> 0 for fine-tuning"
            )
            policy.depth_actor.router_noise_std = 0.0

        policy.depth_actor.train()
        for parameter in policy.depth_encoder.parameters():
            parameter.requires_grad_(False)
        policy.depth_encoder.eval()

        num_actions = env.unwrapped.action_manager.total_action_dim
        log_std = nn.Parameter(
            torch.log(torch.full((num_actions,), args_cli.init_noise_std, device=device))
        )
        obs, extras = env.get_observations()
        obs = obs.to(device)
        depth = extras["observations"]["depth_camera"].to(device)
        critic = Critic(obs.shape[1]).to(device)
        critic.load_from_teacher(runner.alg.policy)

        # Two optimisers, not one. Sharing a single gradient-norm clip couples them:
        # the value loss reaches 4.07 against a surrogate of ~0.01, so the critic's
        # gradient dominates the total norm and the clip rescales the actor by however
        # wrong the critic happens to be on that batch. The actor's step then stops
        # being governed by its own learning rate, which is what the adaptive rule is
        # trying to control -- measured as a KL oscillating between the 1e-5 floor and
        # the 1e-3 ceiling rather than settling.
        actor_params = actor.parameters + [log_std]
        critic_params = list(critic.parameters())
        actor_optimizer = optim.Adam(actor_params, lr=args_cli.lr)
        critic_optimizer = optim.Adam(critic_params, lr=1.0e-3)
        lr = args_cli.lr

        log_dir = args_cli.log_dir or os.path.join(
            "logs",
            "rsl_rl",
            f"{agent_cfg.experiment_name}_ft",
            f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name or 'ft'}",
        )
        os.makedirs(log_dir, exist_ok=True)
        print(f"[INFO]: logging to {log_dir}")

        # Same project as the distillation runs, so the fine-tuning curves sit beside
        # the runs they start from rather than in a separate place.
        try:
            import wandb

            run = wandb.init(
                project=getattr(agent_cfg, "wandb_project", "parkour-moe-transformer"),
                name=os.path.basename(log_dir),
                dir=log_dir,
                config={
                    "stage": "rl_finetune",
                    "checkpoint": args_cli.checkpoint,
                    "task": args_cli.task,
                    "num_envs": args_cli.num_envs,
                    "max_iterations": args_cli.max_iterations,
                    "steps_per_env": args_cli.steps_per_env,
                    "lr": args_cli.lr,
                    "init_noise_std": args_cli.init_noise_std,
                    "entropy_coef": args_cli.entropy_coef,
                    "kl_target": args_cli.kl_target,
                    "top_k": args_cli.top_k,
                    "balance_coef": args_cli.balance_coef,
                    "unfreeze_encoder": args_cli.unfreeze_encoder,
                },
            )
            print(f"[INFO]: wandb run {run.name}")
        except Exception as e:
            print(f"[WARN]: wandb unavailable ({e}); continuing with console logging only")

        # The FT config inherits the training curriculum, so terrain difficulty rises
        # during the run. Without logging it, a falling reward cannot be told apart
        # from the same policy being asked to do something harder.
        try:
            parkour_term = env.unwrapped.parkour_manager.get_term("base_parkour")
        except Exception:
            parkour_term = None

        def terrain_level() -> float:
            """Read mean curriculum difficulty, or NaN when the metric is unavailable."""
            # The parkour term stores this metric in its dictionary, not an attribute.
            if parkour_term is None:
                return float("nan")
            try:
                return float(parkour_term.metrics["terrain_levels"].float().mean())
            except Exception:
                return float("nan")

        rollout_steps, num_envs = args_cli.steps_per_env, env.num_envs
        rew_buf, len_buf = deque(maxlen=100), deque(maxlen=100)
        last_routing: dict = {}
        cur_rew = torch.zeros(num_envs, device=device)
        cur_len = torch.zeros(num_envs, device=device)

        for it in range(args_cli.max_iterations):
            (
                actor_inputs,
                routing_inputs,
                critic_observations,
                sampled_actions,
                old_log_probs,
                values,
                rewards,
                done_flags,
                old_means,
                old_stds,
                replayable,
            ) = [], [], [], [], [], [], [], [], [], [], []
            # gates stay uniform from a reset until the next perception step
            # refreshes them, which is up to four control steps, not one
            stale_gates = torch.zeros(num_envs, dtype=torch.bool, device=device)
            route_u = None
            t0 = time.time()

            with torch.no_grad():
                for _ in range(rollout_steps):
                    is_perception = env.unwrapped.common_step_counter % PERCEPTION_INTERVAL == 0
                    prop_ctx = policy._student_prop(obs, blank_yaw=True)
                    ctrl_rate = (
                        getattr(policy.depth_encoder, "context_rate", "perception") == "control"
                    )
                    if ctrl_rate or is_perception or policy.context is None:
                        out = policy.depth_encoder(
                            depth.clone(), prop_ctx, is_perception_step=is_perception
                        )
                        policy.context, policy.vel_pred, policy.yaw_pred = (
                            out["context"],
                            out["vel"],
                            out["yaw"],
                        )
                    policy.depth_encoder.push_proprioception(prop_ctx)
                    obs[:, YAW_SLICE] = policy.yaw_pred
                    prop_actor = policy._student_prop(obs, blank_yaw=False)

                    u = actor.rollout_input(obs, prop_actor)
                    if is_perception or route_u is None:
                        route_u = u  # this input is what sets the gates
                    mean = actor.mean_action(u, update_routing=is_perception)
                    dist = Normal(mean, mean * 0.0 + log_std.exp())
                    action = dist.sample()

                    if policy.is_moe and is_perception:
                        last_routing = policy.depth_actor.routing_stats()
                    # A step taken straight after a termination acts on gates that
                    # reset() forced to uniform, and the replay cannot reproduce that:
                    # it recomputes the router's own gates from route_u. Those samples
                    # are a fraction of a percent of the batch, each carries an
                    # enormous KL, and the KL is a mean -- enough on their own to
                    # collapse the adaptive learning rate. Excluded rather than
                    # approximated.
                    if is_perception:
                        stale_gates[:] = False  # the router has just recomputed them
                    replayable.append(~stale_gates)
                    actor_inputs.append(u)
                    routing_inputs.append(route_u)
                    critic_observations.append(obs.clone())  # privileged, for the critic replay
                    sampled_actions.append(action)
                    old_log_probs.append(dist.log_prob(action).sum(-1))
                    old_means.append(mean)
                    old_stds.append(mean * 0.0 + log_std.exp())
                    values.append(critic(obs))

                    obs, rew, dones, extras = env.step(action)
                    obs, rew, dones = obs.to(device), rew.to(device), dones.to(device)
                    depth = extras["observations"]["depth_camera"].to(device)
                    policy.reset(dones)
                    stale_gates |= dones.bool()
                    # Every termination in this environment is flagged time_out=True:
                    # terminate_episode merges falls, timeouts and *reaching the final
                    # goal* into one DoneTerm carrying that flag. Treating them all as
                    # terminal states with zero future value teaches the critic that
                    # finishing a course is worth nothing, and PPO then learns not to
                    # finish -- which is exactly what was measured, a policy moving
                    # less and less while its episodes grew longer.
                    #
                    # Bootstrap through them as rsl-rl does, by crediting the value of
                    # the state the action was taken in. That is also how the teacher
                    # this policy came from was trained.
                    raw_rew = rew.clone()  # what the environment actually paid
                    timeouts = extras.get("time_outs")
                    if timeouts is not None:
                        to = timeouts.to(device).float()
                        rew = rew + args_cli.gamma * values[-1] * to
                    rewards.append(rew)
                    done_flags.append(dones.float())

                    # the credit above belongs in the return, not in the statistic:
                    # accumulating it here would report a reward the environment never
                    # paid, rising with the value function rather than with the policy
                    cur_rew += raw_rew
                    cur_len += 1
                    fin = dones.bool()
                    if fin.any():
                        rew_buf.extend(cur_rew[fin].cpu().tolist())
                        len_buf.extend(cur_len[fin].cpu().tolist())
                        cur_rew[fin] = 0.0
                        cur_len[fin] = 0.0

                last_v = critic(obs)

            # ---- GAE ---------------------------------------------------------
            actor_inputs = torch.stack(actor_inputs)
            critic_observations = torch.stack(critic_observations)
            sampled_actions = torch.stack(sampled_actions)
            old_log_probs = torch.stack(old_log_probs)
            routing_inputs = torch.stack(routing_inputs)
            old_means = torch.stack(old_means)
            old_stds = torch.stack(old_stds)
            replayable = torch.stack(replayable)
            values = torch.stack(values)
            rewards = torch.stack(rewards)
            done_flags = torch.stack(done_flags)
            adv, ret = generalized_advantage_estimate(
                rewards, values, done_flags, last_v, args_cli.gamma, args_cli.lam
            )
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            (
                actor_batch,
                routing_batch,
                critic_batch,
                action_batch,
                old_log_prob_batch,
                advantage_batch,
                return_batch,
                old_mean_batch,
                old_std_batch,
            ) = map(
                flatten_rollout,
                (
                    actor_inputs,
                    routing_inputs,
                    critic_observations,
                    sampled_actions,
                    old_log_probs,
                    adv,
                    ret,
                    old_means,
                    old_stds,
                ),
            )
            keep = flatten_rollout(replayable).nonzero(as_tuple=True)[0]
            (
                actor_batch,
                routing_batch,
                critic_batch,
                action_batch,
                old_log_prob_batch,
                advantage_batch,
                return_batch,
                old_mean_batch,
                old_std_batch,
            ) = (
                x[keep]
                for x in (
                    actor_batch,
                    routing_batch,
                    critic_batch,
                    action_batch,
                    old_log_prob_batch,
                    advantage_batch,
                    return_batch,
                    old_mean_batch,
                    old_std_batch,
                )
            )
            batch_size = actor_batch.shape[0]
            if batch_size < args_cli.minibatches:
                raise RuntimeError(
                    "Too few replayable samples; reduce --minibatches or increase rollout size"
                )
            idx = torch.randperm(batch_size, device=device)
            minibatch_size = batch_size // args_cli.minibatches

            warming = it < args_cli.critic_warmup
            stats = {"surrogate": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "n": 0}
            for _ in range(args_cli.epochs):
                for k in range(args_cli.minibatches):
                    sel = idx[k * minibatch_size : (k + 1) * minibatch_size]
                    mean = actor.mean_action(
                        actor_batch[sel], update_routing=False, route_u=routing_batch[sel]
                    )
                    std = mean * 0.0 + log_std.exp()
                    dist = Normal(mean, std)
                    logp = dist.log_prob(action_batch[sel]).sum(-1)
                    ratio = torch.exp(logp - old_log_prob_batch[sel])
                    s1 = -advantage_batch[sel] * ratio
                    s2 = -advantage_batch[sel] * ratio.clamp(
                        1 - args_cli.clip_param, 1 + args_cli.clip_param
                    )
                    surrogate = torch.max(s1, s2).mean()
                    # The critic reads the privileged observation, which is not part of
                    # the actor input, so the rollout stores it and the value function
                    # is regressed on the returns here.
                    value_loss = (critic(critic_batch[sel]) - return_batch[sel]).pow(2).mean()
                    entropy = dist.entropy().sum(-1).mean()
                    actor_loss = surrogate - args_cli.entropy_coef * entropy
                    if policy.is_moe and args_cli.balance_coef > 0.0:
                        actor_loss = (
                            actor_loss
                            + args_cli.balance_coef * policy.depth_actor.load_balancing_loss()
                        )

                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    (value_loss if warming else actor_loss + value_loss).backward()
                    if not warming:
                        nn.utils.clip_grad_norm_(actor_params, 1.0)
                        actor_optimizer.step()
                    nn.utils.clip_grad_norm_(critic_params, 1.0)
                    critic_optimizer.step()

                    with torch.no_grad():
                        # analytic KL between the sampling policy and the updated one,
                        # the quantity the adaptive learning rate is supposed to track.
                        # An RMS log-ratio stands in for it badly: it grows with the
                        # action magnitude rather than with the change in behaviour.
                        kl = (
                            (
                                torch.log(std / old_std_batch[sel])
                                + (old_std_batch[sel].pow(2) + (old_mean_batch[sel] - mean).pow(2))
                                / (2.0 * std.pow(2))
                                - 0.5
                            )
                            .sum(-1)
                            .mean()
                        )
                    # Adapt per minibatch and apply immediately, as rsl-rl does.
                    # Averaging over all 20 gradient steps and adapting once per
                    # iteration measures cumulative drift instead of step size, which
                    # drove the rate to its floor and stopped learning outright.
                    if warming:
                        pass
                    elif kl > 2.0 * args_cli.kl_target:
                        lr = max(1e-5, lr / 1.5)
                    elif kl < 0.5 * args_cli.kl_target and kl > 0.0:
                        # 1e-3 was high enough that one rise produced a KL spike and an
                        # immediate crash back to the floor; the rate never settled
                        lr = min(3e-4, lr * 1.5)
                    for g in actor_optimizer.param_groups:
                        g["lr"] = lr

                    stats["surrogate"] += surrogate.item()
                    stats["value"] += value_loss.item()
                    stats["entropy"] += entropy.item()
                    stats["kl"] += kl.item()
                    stats["n"] += 1

            # The minibatches leave the mixture's cached routing sized to the
            # minibatch, and the next rollout would reuse it on a non-perception step
            # against a batch of num_envs. Clear it so the first step recomputes.
            if policy.is_moe:
                policy.depth_actor.gates = None
                policy.depth_actor.probs = None
                policy.depth_actor.prev_probs = None
            policy.context = None
            policy.depth_latent = None

            k = max(stats["n"], 1)
            mean_kl = stats["kl"] / k

            if run is not None:
                payload = {
                    "ft/reward": statistics.mean(rew_buf) if rew_buf else float("nan"),
                    "ft/episode_length": statistics.mean(len_buf) if len_buf else float("nan"),
                    "ft/surrogate": stats["surrogate"] / k,
                    "ft/value_loss": stats["value"] / k,
                    "ft/entropy": stats["entropy"] / k,
                    "ft/kl": mean_kl,
                    "ft/lr": lr,
                    "ft/action_std": log_std.exp().mean().item(),
                    "ft/steps_per_s": rollout_steps * num_envs / (time.time() - t0),
                    "ft/terrain_level": terrain_level(),
                }
                if policy.is_moe and last_routing:
                    payload["ft/routing_perplexity"] = last_routing.get(
                        "routing_perplexity", float("nan")
                    )
                    payload["ft/routing_entropy"] = last_routing.get(
                        "routing_entropy", float("nan")
                    )
                run.log(payload, step=it)

            if it % 10 == 0 or it == args_cli.max_iterations - 1:
                r = statistics.mean(rew_buf) if rew_buf else float("nan")
                mean_episode_length = statistics.mean(len_buf) if len_buf else float("nan")
                rt = ""
                if policy.is_moe and last_routing:
                    rt = f" | experts {last_routing.get('routing_perplexity', float('nan')):.2f}"
                print(
                    f"it {it:5d} | reward {r:8.3f} | ep_len {mean_episode_length:7.1f} | "
                    f"surrogate {stats['surrogate'] / k:+.4f} | value {stats['value'] / k:7.3f} | "
                    f"kl {mean_kl:.4f} | lr {'warmup' if warming else f'{lr:.2e}'} | std {log_std.exp().mean().item():.3f} | "
                    f"{rollout_steps * num_envs / (time.time() - t0):.0f} steps/s"
                    + rt
                    + (f" | lvl {terrain_level():.2f}" if parkour_term is not None else ""),
                    flush=True,
                )

            if it % args_cli.save_interval == 0 or it == args_cli.max_iterations - 1:
                torch.save(
                    {
                        "model_state_dict": runner.alg.policy.state_dict(),
                        "estimator_state_dict": runner.alg.estimator.state_dict(),
                        "depth_encoder_state_dict": policy.depth_encoder.state_dict(),
                        "depth_actor_state_dict": policy.depth_actor.state_dict(),
                        "critic_ft_state_dict": critic.state_dict(),
                        "log_std": log_std.detach().cpu(),
                        "iter": it,
                        "infos": None,
                    },
                    os.path.join(log_dir, f"model_{it}.pt"),
                )

    finally:
        try:
            if run is not None:
                run.finish()
        finally:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
