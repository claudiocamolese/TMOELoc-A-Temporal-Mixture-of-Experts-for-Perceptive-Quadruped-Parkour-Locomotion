"""Runner for the Transformer + MoE student.

Same environment, same teacher and same optimisation budget as the original
distillation runner; only the student architecture changes.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from copy import copy

import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

from .distillation_moe import DistillationMoE
from .on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

# proprioception slice holding the (env_class != flat, env_class == flat) one-hot.
ENV_CLASS_SLICE = slice(11, 13)
# proprioception slice holding (delta_yaw, delta_next_yaw) -- the student must predict these.
YAW_SLICE = slice(6, 8)


class OnPolicyRunnerMoE(OnPolicyRunnerWithExtractor):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.zero_env_class = bool(self.depth_encoder_cfg.get("zero_env_class", True))
        # "true"      -> the actor reads the same ground-truth base velocity the
        #                original student actor reads, keeping the comparison exact.
        # "predicted" -> the actor consumes the context encoder's own estimate.
        self.vel_source = self.depth_encoder_cfg.get("vel_source", "true")
        # the base runner binds `learn` as an instance attribute, so override it here
        self.learn = self.learn_vision_moe

    def learn_vision_moe(self, num_learning_iterations, init_at_random_ep_len=False):  # noqa: C901
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

        context = None
        vel_pred = None
        yaw_pred = None

        for it in range(start_iter, tot_iter):
            start = time.time()
            buffers: dict[str, list[torch.Tensor]] = {
                "action": [],
                "yaw": [],
                "terrain": [],
                "dyn": [],
                "vel": [],
                "balance": [],
                "smooth": [],
            }
            delta_yaw_ok_buffer = []

            for _ in range(self.depth_encoder_cfg["num_steps_per_env"]):
                is_perception_step = self.env.unwrapped.common_step_counter % 5 == 0

                # ---- student-visible proprioception -------------------------
                prop_ctx = obs[:, :num_prop].clone()
                prop_ctx[:, YAW_SLICE] = 0.0  # the heading must be inferred, not read
                if self.zero_env_class:
                    prop_ctx[:, ENV_CLASS_SLICE] = 0.0

                # ---- perception step: refresh the temporal context ----------
                if is_perception_step or context is None:
                    out = self.alg.depth_encoder(depth_image.clone(), prop_ctx)
                    context, vel_pred, yaw_pred = out["context"], out["vel"], out["yaw"]

                    with torch.no_grad():
                        z_scan_teacher = self.alg.policy.actor.infer_scandots_latent(obs)
                        z_priv_teacher = self.alg.policy.actor.infer_priv_latent(obs)
                        vel_target = obs[:, priv_explicit_slice].detach()

                    buffers["yaw"].append(obs[:, YAW_SLICE].detach() - yaw_pred)
                    buffers["terrain"].append(out["terrain"] - z_scan_teacher)
                    buffers["dyn"].append(out["dyn"] - z_priv_teacher)
                    buffers["vel"].append(out["vel"] - vel_target)

                self.alg.depth_encoder.push_proprioception(prop_ctx)

                # ---- teacher action on the untouched privileged observation --
                with torch.no_grad():
                    actions_teacher = self.alg.policy.act_inference(obs, hist_encoding=True, scandots_latent=None)
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
                buffers["action"].append(actions_teacher.detach() - actions_student)

                if is_perception_step:
                    buffers["balance"].append(self.alg.depth_actor.load_balancing_loss())
                    buffers["smooth"].append(self.alg.depth_actor.routing_smoothness_loss())

                # ---- environment step ---------------------------------------
                obs, _, dones, infos = self.env.step(actions_student.detach().to(self.env.device))
                obs, dones = obs.to(self.device), dones.to(self.device)
                delta_yaw_ok = infos["observations"]["delta_yaw_ok"].to(self.device)
                depth_image = infos["observations"]["depth_camera"].to(self.device)
                obs = self.obs_normalizer(obs)

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

            delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
            routing_stats = self.alg.depth_actor.routing_stats()
            loss_dict = self.alg.update_depth_actor(buffers)
            loss_dict.update(routing_stats)

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
                    or (it - self.start_learning_iteration < 5000 and it % (2 * self.save_interval) == 0)
                    or (it - self.start_learning_iteration >= 5000 and it % (5 * self.save_interval) == 0)
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
            raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def save(self, path: str, infos=None):
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
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # the teacher always comes from a plain PPO checkpoint
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.estimator.load_state_dict(loaded_dict["estimator_state_dict"])
        if "depth_encoder_state_dict" in loaded_dict:
            print("Saved temporal context encoder detected, loading...")
            self.alg.depth_encoder.load_state_dict(loaded_dict["depth_encoder_state_dict"])
        if "depth_actor_state_dict" in loaded_dict:
            print("Saved MoE actor detected, loading...")
            self.alg.depth_actor.load_state_dict(loaded_dict["depth_actor_state_dict"])
        if load_optimizer and "depth_actor_optimizer_state_dict" in loaded_dict:
            self.alg.depth_actor_optimizer.load_state_dict(loaded_dict["depth_actor_optimizer_state_dict"])
        if "depth_actor_state_dict" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_depth_policy(self, device=None):
        self.alg.depth_encoder.eval()
        self.alg.depth_actor.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
            self.alg.depth_actor.to(device)
        return self.alg.depth_actor
