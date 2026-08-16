"""Fine-tune a distilled student's actor with PPO on the task reward.

Why this stage exists
---------------------
Distillation optimises the distance to the teacher's actions. Measured here, on
both students independently, that objective saturates: `terrain_levels` stops
moving around iteration 11 000 while the action loss keeps falling 3% per
thousand iterations, and the gap that remains against the teacher is falls --
2.6x to 3.2x its hazard -- not pace. An L2 on actions averages over every
timestep, and the timesteps that end an episode are well under 1% of them, so
fitting the teacher better does not buy fewer falls.

The task reward does penalise falls. Rudin et al., "Parkour in the Wild" (IJRR
2026), report a distilled policy at 40-60% success rising to 80-95% after exactly
this stage, and no recent parkour work stops at supervised distillation.

What is trained
---------------
The actor only; the perception encoder stays frozen at its distilled weights.
That is a deliberate scoping choice and the ablations support it: zeroing the
context costs 12.2 goals per env-minute and quadruples the falls, so perception
is load-bearing and working, while the deficit is behavioural. Freezing it also
makes the update cheap -- the policy input is a vector already computed during
the rollout, so no depth image enters the storage and no CNN runs in the update.

Pass --unfreeze_encoder to train it too, at roughly 4x the step time.

The critic reads the privileged observation and is initialised from the teacher's
own critic. Asymmetric actor-critic is standard in simulation: the value function
may see what the deployed policy cannot.

  python scripts/rsl_rl/finetune_rl.py --checkpoint <distilled model_*.pt> \
      --task Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-FT-v0 --headless
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
parser.add_argument("--init_noise_std", type=float, default=0.5,
                    help="Exploration around the distilled actions. Large values throw away "
                         "the behaviour distillation just bought.")
parser.add_argument("--entropy_coef", type=float, default=0.002)
parser.add_argument("--clip_param", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--kl_target", type=float, default=None,
                    help="Left unset it is derived from the noise, because the two are not "
                         "independent: the Gaussian KL carries a (mu_old-mu_new)^2 / 2*sigma^2 "
                         "term, so halving sigma quadruples the KL reported for an identical "
                         "step. A fixed target then pins the learning rate to its floor. The "
                         "default is calibrated at sigma 0.5 and scaled by (0.5/sigma)^2.")
parser.add_argument("--unfreeze_encoder", action="store_true", default=False)
parser.add_argument("--top_k", type=int, default=None,
                    help="Override the mixture's top-k at the start of fine-tuning. "
                         "top_k = num_experts is soft routing. Switching here rather than "
                         "in a fresh distillation run costs nothing: PPO retrains the actor "
                         "anyway, so the router and the experts reorganise for the new regime "
                         "under the objective that actually matters.")
parser.add_argument("--balance_coef", type=float, default=0.02,
                    help="Small load-balancing term. Distillation had one; PPO does not, and "
                         "without it nothing stops the gates collapsing onto one expert.")
parser.add_argument("--save_interval", type=int, default=250)
# --run_name comes from cli_args.add_rsl_rl_args; do not redefine it
parser.add_argument("--log_dir", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required: fine-tuning starts from a distilled policy")
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.kl_target is None:
    args_cli.kl_target = 0.02 * (0.5 / max(args_cli.init_noise_std, 1e-6)) ** 2
    print(f"[INFO]: kl_target derived from noise {args_cli.init_noise_std}: "
          f"{args_cli.kl_target:.4f}")

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

RUNNER_CLASSES = {
    "OnPolicyRunnerWithExtractor": OnPolicyRunnerWithExtractor,
    "OnPolicyRunnerMoE": OnPolicyRunnerMoE,
}


class Critic(nn.Module):
    """Value function on the privileged observation, seeded from the teacher's."""

    def __init__(self, num_obs: int, hidden=(512, 256, 128)):
        super().__init__()
        dims = [num_obs, *hidden]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ELU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def load_from_teacher(self, policy) -> bool:
        """Copy the teacher's critic if the shapes line up."""
        src = getattr(policy, "critic", None)
        if src is None:
            return False
        s_lin = [m for m in src.modules() if isinstance(m, nn.Linear)]
        d_lin = [m for m in self.net if isinstance(m, nn.Linear)]
        if len(s_lin) != len(d_lin) or any(
            a.weight.shape != b.weight.shape for a, b in zip(s_lin, d_lin)
        ):
            print("[INFO]: teacher critic shape differs; starting the value function from scratch")
            return False
        with torch.no_grad():
            for a, b in zip(s_lin, d_lin):
                b.weight.copy_(a.weight)
                b.bias.copy_(a.bias)
        print("[INFO]: critic initialised from the teacher's own")
        return True


class ActorAdapter:
    """One interface over the two students, for rollout and for replay.

    The rollout records whatever the actor needs as a flat vector, so the PPO
    update can re-evaluate the policy without touching the simulator or the
    frozen encoder.
    """

    def __init__(self, policy: StudentPolicy):
        self.p = policy
        self.is_moe = policy.is_moe

    @property
    def parameters(self):
        return list(self.p.depth_actor.parameters())

    def rollout_input(self, obs, prop_actor):
        """The vector the actor consumes at this control step."""
        if self.is_moe:
            vel = obs[:, self.p.priv_explicit_slice] if self.p.vel_source == "true" else self.p.vel_pred
            return torch.cat([prop_actor, self.p.context, vel], dim=-1)
        # GRU student: the actor reads the masked full observation plus the latent
        return torch.cat([self.p.runner.mask_student_obs(obs), self.p.depth_latent], dim=-1)

    def mean_action(self, u, update_routing: bool, route_u=None):
        if self.is_moe:
            n_p, n_c = self.p.num_prop, self.p.depth_actor.context_dim
            prop, ctx, vel = u[:, :n_p], u[:, n_p:n_p + n_c], u[:, n_p + n_c:]
            return self.p.depth_actor(prop, ctx, vel=vel, update_routing=update_routing,
                                      route_u=route_u)
        n_obs = u.shape[1] - self.p.depth_latent.shape[1]
        masked, latent = u[:, :n_obs], u[:, n_obs:]
        return self.p.depth_actor(masked, hist_encoding=True, scandots_latent=latent)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.depth_camera.depth_cam.params["debug_vis"] = False
    device = agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    active = [t for t in env.unwrapped.reward_manager.active_terms]
    print(f"[INFO]: {len(active)} reward terms active: {', '.join(active[:6])}"
          + (" ..." if len(active) > 6 else ""))
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
    actor = ActorAdapter(policy)
    print(f"[INFO]: fine-tuning {policy.variant} from {args_cli.checkpoint}")

    if args_cli.top_k is not None and policy.is_moe:
        old = policy.depth_actor.top_k
        policy.depth_actor.top_k = min(args_cli.top_k, policy.depth_actor.num_experts)
        print(f"[INFO]: top_k {old} -> {policy.depth_actor.top_k}"
              + (" (soft routing)" if policy.depth_actor.top_k == policy.depth_actor.num_experts else ""))

    if policy.is_moe and policy.depth_actor.router_noise_std > 0.0:
        # In PPO the exploration is the action distribution. Noisy top-k gating
        # on top of it makes the policy irreproducible: replaying a stored
        # transition draws different gates, so the recomputed mean differs from
        # the one that was acted on and the probability ratio is wrong before a
        # single gradient step is taken.
        print(f"[INFO]: router noise {policy.depth_actor.router_noise_std} -> 0 for fine-tuning")
        policy.depth_actor.router_noise_std = 0.0

    policy.depth_actor.train()
    if args_cli.unfreeze_encoder:
        policy.depth_encoder.train()
    else:
        for p in policy.depth_encoder.parameters():
            p.requires_grad_(False)
        policy.depth_encoder.eval()

    num_actions = env.unwrapped.action_manager.total_action_dim
    log_std = nn.Parameter(torch.log(torch.full((num_actions,), args_cli.init_noise_std, device=device)))
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
    if args_cli.unfreeze_encoder:
        actor_params += list(policy.depth_encoder.parameters())
    critic_params = list(critic.parameters())
    opt = optim.Adam(actor_params, lr=args_cli.lr)
    opt_critic = optim.Adam(critic_params, lr=1.0e-3)
    lr = args_cli.lr

    log_dir = args_cli.log_dir or os.path.join(
        "logs", "rsl_rl", f"{agent_cfg.experiment_name}_ft",
        f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name or 'ft'}",
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO]: logging to {log_dir}")

    # Same project as the distillation runs, so the fine-tuning curves sit beside
    # the runs they start from rather than in a separate place.
    run = None
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
        # the parkour term keeps it in its metrics dict, not as an attribute
        if parkour_term is None:
            return float("nan")
        try:
            return float(parkour_term.metrics["terrain_levels"].float().mean())
        except Exception:
            return float("nan")

    T, N = args_cli.steps_per_env, env.num_envs
    rew_buf, len_buf = deque(maxlen=100), deque(maxlen=100)
    last_routing: dict = {}
    cur_rew = torch.zeros(N, device=device)
    cur_len = torch.zeros(N, device=device)

    for it in range(args_cli.max_iterations):
        U, UR, O, A, LP, V, R, D, TO, MU, SG = [], [], [], [], [], [], [], [], [], [], []
        route_u = None
        t0 = time.time()

        with torch.no_grad():
            for _ in range(T):
                is_perception = env.unwrapped.common_step_counter % 5 == 0
                prop_ctx = policy._student_prop(obs, blank_yaw=True)
                if is_perception or policy.context is None:
                    out = policy.depth_encoder(depth.clone(), prop_ctx)
                    policy.context, policy.vel_pred, policy.yaw_pred = (
                        out["context"], out["vel"], out["yaw"],
                    )
                policy.depth_encoder.push_proprioception(prop_ctx)
                obs[:, 6:8] = policy.yaw_pred
                prop_actor = policy._student_prop(obs, blank_yaw=False)

                u = actor.rollout_input(obs, prop_actor)
                if is_perception or route_u is None:
                    route_u = u          # this input is what sets the gates
                mean = actor.mean_action(u, update_routing=is_perception)
                dist = Normal(mean, mean * 0.0 + log_std.exp())
                action = dist.sample()

                if policy.is_moe and is_perception:
                    last_routing = policy.depth_actor.routing_stats()
                U.append(u)
                UR.append(route_u)
                O.append(obs.clone())   # privileged, for the critic replay
                A.append(action)
                LP.append(dist.log_prob(action).sum(-1))
                MU.append(mean)
                SG.append(mean * 0.0 + log_std.exp())
                V.append(critic(obs))

                obs, rew, dones, extras = env.step(action)
                obs, rew, dones = obs.to(device), rew.to(device), dones.to(device)
                depth = extras["observations"]["depth_camera"].to(device)
                policy.reset(dones)
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
                raw_rew = rew.clone()   # what the environment actually paid
                timeouts = extras.get("time_outs")
                if timeouts is not None:
                    to = timeouts.to(device).float()
                    rew = rew + args_cli.gamma * V[-1] * to
                else:
                    to = torch.zeros_like(rew)
                R.append(rew)
                TO.append(to)
                D.append(dones.float())

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
        U = torch.stack(U); O = torch.stack(O); A = torch.stack(A); LP = torch.stack(LP)
        UR = torch.stack(UR); MU = torch.stack(MU); SG = torch.stack(SG)
        V = torch.stack(V); R = torch.stack(R); D = torch.stack(D)
        adv = torch.zeros_like(R)
        gae = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            nxt = last_v if t == T - 1 else V[t + 1]
            # D marks the end of a trajectory segment, so the GAE recursion still
            # stops there; the truncation credit above is what replaces the value
            # that would otherwise be lost.
            delta = R[t] + args_cli.gamma * nxt * (1.0 - D[t]) - V[t]
            gae = delta + args_cli.gamma * args_cli.lam * (1.0 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
        Uf, URf, Of, Af, LPf, advf, retf, MUf, SGf = map(flat, (U, UR, O, A, LP, adv, ret, MU, SG))
        n = Uf.shape[0]
        idx = torch.randperm(n, device=device)
        mb = n // args_cli.minibatches

        stats = {"surrogate": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "n": 0}
        for _ in range(args_cli.epochs):
            for k in range(args_cli.minibatches):
                sel = idx[k * mb: (k + 1) * mb]
                mean = actor.mean_action(Uf[sel], update_routing=False, route_u=URf[sel])
                std = mean * 0.0 + log_std.exp()
                dist = Normal(mean, std)
                logp = dist.log_prob(Af[sel]).sum(-1)
                ratio = torch.exp(logp - LPf[sel])
                s1 = -advf[sel] * ratio
                s2 = -advf[sel] * ratio.clamp(1 - args_cli.clip_param, 1 + args_cli.clip_param)
                surrogate = torch.max(s1, s2).mean()
                # The critic reads the privileged observation, which is not part of
                # the actor input, so the rollout stores it and the value function
                # is regressed on the returns here.
                value_loss = (critic(Of[sel]) - retf[sel]).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
                actor_loss = surrogate - args_cli.entropy_coef * entropy
                if policy.is_moe and args_cli.balance_coef > 0.0:
                    actor_loss = actor_loss + args_cli.balance_coef * policy.depth_actor.load_balancing_loss()

                opt.zero_grad()
                opt_critic.zero_grad()
                (actor_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(actor_params, 1.0)
                nn.utils.clip_grad_norm_(critic_params, 1.0)
                opt.step()
                opt_critic.step()

                with torch.no_grad():
                    # analytic KL between the sampling policy and the updated one,
                    # the quantity the adaptive learning rate is supposed to track.
                    # An RMS log-ratio stands in for it badly: it grows with the
                    # action magnitude rather than with the change in behaviour.
                    kl = (
                        torch.log(std / SGf[sel])
                        + (SGf[sel].pow(2) + (MUf[sel] - mean).pow(2)) / (2.0 * std.pow(2))
                        - 0.5
                    ).sum(-1).mean()
                # Adapt per minibatch and apply immediately, as rsl-rl does.
                # Averaging over all 20 gradient steps and adapting once per
                # iteration measures cumulative drift instead of step size, which
                # drove the rate to its floor and stopped learning outright.
                if kl > 2.0 * args_cli.kl_target:
                    lr = max(1e-5, lr / 1.5)
                elif kl < 0.5 * args_cli.kl_target and kl > 0.0:
                    # 1e-3 was high enough that one rise produced a KL spike and an
                    # immediate crash back to the floor; the rate never settled
                    lr = min(3e-4, lr * 1.5)
                for g in opt.param_groups:
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
                "ft/steps_per_s": T * N / (time.time() - t0),
                "ft/terrain_level": terrain_level(),
            }
            if policy.is_moe and last_routing:
                payload["ft/routing_perplexity"] = last_routing.get("routing_perplexity", float("nan"))
                payload["ft/routing_entropy"] = last_routing.get("routing_entropy", float("nan"))
            run.log(payload, step=it)

        if it % 10 == 0 or it == args_cli.max_iterations - 1:
            r = statistics.mean(rew_buf) if rew_buf else float("nan")
            L = statistics.mean(len_buf) if len_buf else float("nan")
            rt = ""
            if policy.is_moe and last_routing:
                rt = f" | experts {last_routing.get('routing_perplexity', float('nan')):.2f}"
            print(f"it {it:5d} | reward {r:8.3f} | ep_len {L:7.1f} | "
                  f"surrogate {stats['surrogate'] / k:+.4f} | value {stats['value'] / k:7.3f} | "
                  f"kl {mean_kl:.4f} | lr {lr:.2e} | std {log_std.exp().mean().item():.3f} | "
                  f"{T * N / (time.time() - t0):.0f} steps/s" + rt
                  + (f" | lvl {terrain_level():.2f}" if parkour_term is not None else ""),
                  flush=True)

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

    env.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
    simulation_app.close()
