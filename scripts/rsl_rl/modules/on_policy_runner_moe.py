"""Runner for the Transformer + MoE student.

Same environment, same teacher and same optimisation budget as the original
distillation runner; only the student architecture changes.
"""

from __future__ import annotations

import os
import time
from collections import deque
from copy import copy

import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

from .distillation_moe import DistillationMoE
from .on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from .student_observations import ENV_CLASS_SLICE, PERCEPTION_INTERVAL, YAW_SLICE


class OnPolicyRunnerMoE(OnPolicyRunnerWithExtractor):
    """Collect student trajectories and supervise context and actions with a frozen teacher."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.zero_env_class = bool(self.depth_encoder_cfg.get("zero_env_class", True))
        # "true"      -> the actor reads the same ground-truth base velocity the
        #                original student actor reads, keeping the comparison exact.
        # "predicted" -> the actor consumes the context encoder's own estimate.
        self.vel_source = self.depth_encoder_cfg.get("vel_source", "true")
        self.action_target_clip = float(self.depth_encoder_cfg.get("action_target_clip", 4.8))
        # the base runner binds `learn` as an instance attribute, so override it here
        self.learn = self.learn_vision_moe

    def learn_vision_moe(self, num_learning_iterations, init_at_random_ep_len=False):  # noqa: C901
        """Alternate on-student rollouts with one joint distillation update per iteration."""
        if not isinstance(self.alg, DistillationMoE):
            raise TypeError(f"The algorithm must be DistillationMoE, got {type(self.alg)}")
        self.alg: DistillationMoE

        self._init_writer()

        obs, extras = self.env.get_observations()
        delta_yaw_ok = extras["observations"]["delta_yaw_ok"].to(self.device)
        depth_image = extras["observations"]["depth_camera"].to(self.device)
        obs = obs.to(self.device)

        num_prop = self.alg.num_prop
        num_scan = self.alg.num_scan
        num_priv_explicit = self.alg.priv_states_dim
        priv_explicit_slice = slice(num_prop + num_scan, num_prop + num_scan + num_priv_explicit)
        num_priv_latent = self.depth_encoder_cfg["num_priv_latent"]
        priv_latent_slice = slice(
            num_prop + num_scan + num_priv_explicit,
            num_prop + num_scan + num_priv_explicit + num_priv_latent,
        )

        # A termination is a fall or a completed course; only the first is a
        # failure, and the goal counter is what distinguishes them. It has to be
        # read *before* env.step, because the reset that follows a termination
        # zeroes it.
        parkour_term = self.env.unwrapped.parkour_manager.get_term("base_parkour")
        num_goals = int(parkour_term.num_goals)
        risk_lead = int(self.depth_encoder_cfg.get("risk_lead", 25))
        risk_horizon = int(self.depth_encoder_cfg.get("risk_horizon", 75))

        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()
        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        lenbuffer = deque(maxlen=100)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        # The schedule spans depth_encoder.total_iters measured from here, not
        # the length of this invocation, so that a short resumed run reproduces
        # the learning rate the original run had at the same iteration.
        self.alg.schedule_offset = start_iter
        if self.alg.total_iters != tot_iter:
            print(
                f"[INFO]: LR schedule spans {self.alg.total_iters} iterations; "
                f"this run covers {start_iter}..{tot_iter}"
            )

        context = None
        risk_pred = None
        vel_pred = None
        yaw_pred = None

        for it in range(start_iter, tot_iter):
            start = time.time()
            self.alg.set_iteration(it)
            buffers: dict[str, list[torch.Tensor]] = {
                "action": [],
                "yaw": [],
                "terrain": [],
                "dyn": [],
                "vel": [],
                "balance": [],
                "smooth": [],
                "zloss": [],
                "risk_pred": [],
                "risk_label": [],
                "risk_valid": [],
            }
            # per-step termination record, used after the rollout to label each
            # step with whether a fall followed it
            fall_log: list[torch.Tensor] = []
            delta_yaw_ok_buffer = []
            # divergence diagnostics, accumulated on device to avoid per-step syncs
            diag = {
                k: torch.zeros((), device=self.device) for k in ("act_s", "act_t", "ctx", "obs")
            }

            for _ in range(self.depth_encoder_cfg["num_steps_per_env"]):
                is_perception_step = (
                    self.env.unwrapped.common_step_counter % PERCEPTION_INTERVAL == 0
                )

                # ---- student-visible proprioception -------------------------
                prop_ctx = obs[:, :num_prop].clone()
                prop_ctx[:, YAW_SLICE] = 0.0  # the heading must be inferred, not read
                if self.zero_env_class:
                    prop_ctx[:, ENV_CLASS_SLICE] = 0.0

                # ---- perception step: refresh the temporal context ----------
                # At control rate the encoder runs every step: the visual latent
                # is cached on the camera's clock inside it, only the proprio
                # branch and the transformer recompute.
                ctrl_rate = self.alg.depth_encoder.context_rate == "control"
                if ctrl_rate or is_perception_step or context is None:
                    out = self.alg.depth_encoder(
                        depth_image.clone(), prop_ctx, is_perception_step=is_perception_step
                    )
                    context, vel_pred, yaw_pred = out["context"], out["vel"], out["yaw"]
                    risk_pred = out.get("risk")

                # The auxiliary supervision stays on the perception clock even when
                # the context runs at control rate. Its targets come from the
                # teacher -- infer_scandots_latent is a full privileged forward --
                # so sampling them five times as often would cost five times as
                # much and silently reweight the objective against the action loss.
                if is_perception_step or len(buffers["terrain"]) == 0:
                    with torch.no_grad():
                        z_scan_teacher = self.alg.policy.actor.infer_scandots_latent(obs)
                        vel_target = obs[:, priv_explicit_slice].detach()
                        if self.alg.dyn_target == "raw_params":
                            # standardised physical parameters; the teacher's own
                            # privileged latent is constant across environments
                            raw = obs[:, priv_latent_slice][:, self.alg.dyn_dims].detach()
                            dyn_target = self.alg.dyn_normaliser(raw)
                        else:
                            dyn_target = self.alg.policy.actor.infer_priv_latent(obs)

                    buffers["yaw"].append(obs[:, YAW_SLICE].detach() - yaw_pred)
                    buffers["terrain"].append(out["terrain"] - z_scan_teacher)
                    buffers["dyn"].append(out["dyn"] - dyn_target)
                    buffers["vel"].append(out["vel"] - vel_target)

                self.alg.depth_encoder.push_proprioception(prop_ctx)

                # ---- teacher action on the untouched privileged observation --
                with torch.no_grad():
                    actions_teacher = self.alg.policy.act_inference(
                        obs, hist_encoding=True, scandots_latent=None
                    )
                    # Regress the action the environment will actually execute.
                    # Beyond this clip the teacher's output is unbounded garbage
                    # produced from saturated observations, and identical in
                    # effect to the clipped value.
                    action_target = actions_teacher.clamp(
                        -self.action_target_clip, self.action_target_clip
                    )
                    delta_yaw_ok_buffer.append(
                        torch.nonzero(delta_yaw_ok).size(0) / delta_yaw_ok.numel()
                    )

                # ---- mixture of teacher and student heading (Sec. 3.3) ------
                obs[delta_yaw_ok, YAW_SLICE] = yaw_pred.detach()[delta_yaw_ok]

                prop_actor = obs[:, :num_prop].clone()
                if self.zero_env_class:
                    prop_actor[:, ENV_CLASS_SLICE] = 0.0
                vel_input = (
                    obs[:, priv_explicit_slice].detach() if self.vel_source == "true" else vel_pred
                )

                actions_student = self.alg.depth_actor(
                    prop_actor, context, vel=vel_input, update_routing=is_perception_step
                )
                buffers["action"].append(action_target - actions_student)

                with torch.no_grad():
                    diag["act_s"] = torch.maximum(diag["act_s"], actions_student.abs().max())
                    diag["act_t"] = torch.maximum(diag["act_t"], actions_teacher.abs().max())
                    diag["ctx"] = torch.maximum(diag["ctx"], context.abs().max())
                    diag["obs"] = torch.maximum(diag["obs"], obs.abs().max())

                if is_perception_step:
                    buffers["balance"].append(self.alg.depth_actor.load_balancing_loss())
                    buffers["smooth"].append(self.alg.depth_actor.routing_smoothness_loss())
                    buffers["zloss"].append(self.alg.depth_actor.router_z_loss())

                if self.alg.use_risk_head and risk_pred is not None:
                    # the cached prediction applies to every control step until
                    # the next perception step, exactly as the context does
                    buffers["risk_pred"].append(risk_pred)
                    goal_before = parkour_term.cur_goal_idx.clone()

                # ---- environment step ---------------------------------------
                obs, _, dones, infos = self.env.step(actions_student.detach().to(self.env.device))
                obs, dones = obs.to(self.device), dones.to(self.device)
                delta_yaw_ok = infos["observations"]["delta_yaw_ok"].to(self.device)
                depth_image = infos["observations"]["depth_camera"].to(self.device)
                obs = self.obs_normalizer(obs)

                if self.alg.use_risk_head and risk_pred is not None:
                    fall_log.append(dones.bool() & (goal_before < num_goals - 1))

                # unlike the original GRU backbone, temporal state does not leak
                # across episode boundaries
                self.alg.depth_encoder.reset(dones)
                self.alg.depth_actor.reset(dones)

                if self.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            if self.alg.use_risk_head and fall_log:
                # Label each step with "a fall follows in [t+lead, t+horizon)".
                # The lead excludes the states where the robot is already past
                # the tilt cutoff -- predicting those is bookkeeping, and a head
                # trained on them would fire too late to be worth weighting by.
                #
                # A step is only labelled when its whole window fits inside the
                # rollout; beyond that a quiet window cannot be told apart from
                # an unobserved one, so those steps are masked out rather than
                # counted as negatives.
                falls = torch.stack(fall_log)  # (T, num_envs)
                T = falls.shape[0]
                labels, valids = [], []
                for t in range(T):
                    hi = t + risk_horizon
                    ok = hi <= T
                    window = falls[t + risk_lead : hi] if ok else falls[0:0]
                    labels.append(
                        window.any(dim=0) if window.numel() else torch.zeros_like(falls[0])
                    )
                    valids.append(torch.full_like(falls[0], ok))
                buffers["risk_label"] = labels
                buffers["risk_valid"] = valids

            delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
            routing_stats = self.alg.depth_actor.routing_stats()
            loss_dict = self.alg.update_depth_actor(buffers)
            loss_dict.update(routing_stats)
            loss_dict.update(
                {
                    "absmax_action_student": diag["act_s"].item(),
                    "absmax_action_teacher": diag["act_t"].item(),
                    "absmax_context": diag["ctx"].item(),
                    "absmax_obs": diag["obs"].item(),
                }
            )

            stop = time.time()
            learn_time = stop - start

            self.alg.depth_encoder.detach_buffers()
            self.alg.depth_actor.detach_routing()
            context = context.detach()
            vel_pred = vel_pred.detach()
            yaw_pred = yaw_pred.detach()
            obs = obs.detach()

            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                self.log_vision(locals())
                if (
                    (it - self.start_learning_iteration < 2500 and it % self.save_interval == 0)
                    or (
                        it - self.start_learning_iteration < 5000
                        and it % (2 * self.save_interval) == 0
                    )
                    or (
                        it - self.start_learning_iteration >= 5000
                        and it % (5 * self.save_interval) == 0
                    )
                ):
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ------------------------------------------------------------------ utils

    def _init_writer(self):
        """Lazily create the configured metrics writer and record the run configuration."""
        if self.log_dir is None or self.writer is not None or self.disable_logs:
            return
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()
        if self.logger_type == "neptune":
            from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

            self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise ValueError(
                "Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'."
            )

    def save(self, path: str, infos=None):
        """Save teacher, estimator, student, and optimizer state using the released key names."""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "estimator_state_dict": self.alg.estimator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "depth_encoder_state_dict": self.alg.depth_encoder.state_dict(),
            "depth_actor_state_dict": self.alg.depth_actor.state_dict(),
            "depth_actor_optimizer_state_dict": self.alg.depth_actor_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = (
                self.privileged_obs_normalizer.state_dict()
            )
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        """Load a teacher or student checkpoint; warm-start experts only for a fresh student."""
        loaded_dict = torch.load(path, weights_only=False)
        # the teacher always comes from a plain PPO checkpoint
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.estimator.load_state_dict(loaded_dict["estimator_state_dict"])
        if "depth_encoder_state_dict" in loaded_dict:
            print("Saved temporal context encoder detected, loading...")
            self.alg.depth_encoder.load_state_dict(loaded_dict["depth_encoder_state_dict"])
        resumed_actor = "depth_actor_state_dict" in loaded_dict
        if resumed_actor:
            print("Saved MoE actor detected, loading...")
            self.alg.depth_actor.load_state_dict(loaded_dict["depth_actor_state_dict"])
        elif self.depth_encoder_cfg.get("warm_start_experts", False):
            # Only when the experts are still at their random initialisation:
            # on a resume this would overwrite trained weights with the teacher's.
            self.alg.warm_start_experts_from_teacher()
        if load_optimizer and "depth_actor_optimizer_state_dict" in loaded_dict:
            self.alg.depth_actor_optimizer.load_state_dict(
                loaded_dict["depth_actor_optimizer_state_dict"]
            )
        if "depth_actor_state_dict" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_depth_policy(self, device=None):
        """Return the student actor in evaluation mode on the requested device."""
        self.alg.depth_encoder.eval()
        self.alg.depth_actor.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
            self.alg.depth_actor.to(device)
        return self.alg.depth_actor
