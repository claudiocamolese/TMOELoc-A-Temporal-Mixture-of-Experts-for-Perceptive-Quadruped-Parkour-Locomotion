"""Distillation of a scandots teacher into a Transformer + MoE student.

Single joint phase (no separate context pre-training):

    L = la * L_action + ly * L_yaw
      + lt * L_terrain + ld * L_dyn + lv * L_vel
      + lb * L_balance + ls * L_smooth

``L_terrain`` / ``L_dyn`` regress the teacher's own scan-encoder and
privileged-encoder latents, so a single context vector has to carry both the
terrain geometry and the hidden robot dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic_with_encoder import ActorCriticRMA
from .feature_extractors.temporal_transformer import TemporalContextEncoder
from .moe_actor import MoEActor


class DistillationMoE:
    policy: ActorCriticRMA

    def __init__(
        self,
        policy,
        estimator,
        estimator_paras,
        learning_rate,
        depth_encoder_cfg,
        policy_cfg,
        device,
        max_grad_norm=1.0,
        multi_gpu_cfg: dict | None = None,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy = policy
        self.estimator: nn.Module = estimator
        self.priv_states_dim = estimator_paras["num_priv_explicit"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.rnd = None  # keeps the rsl-rl runner base class happy

        cfg = dict(depth_encoder_cfg)
        cfg["num_scan_latent"] = policy_cfg["scan_encoder_dims"][-1]
        cfg["num_priv_latent_out"] = policy_cfg["priv_encoder_dims"][-1]
        self.depth_encoder = TemporalContextEncoder(cfg).to(device)
        self.depth_actor = MoEActor(
            num_prop=cfg["num_prop"],
            context_dim=cfg["context_dim"],
            num_priv_explicit=cfg["num_priv_explicit"],
            num_actions=policy.actor.num_actions,
            num_experts=cfg["num_experts"],
            top_k=cfg["top_k"],
            expert_hidden_dims=tuple(cfg["expert_hidden_dims"]),
            router_hidden_dim=cfg["router_hidden_dim"],
            use_vel_in_input=cfg["use_vel_in_input"],
            router_noise_std=cfg["router_noise_std"],
        ).to(device)

        self.depth_encoder_cfg = cfg
        self.learning_rate = cfg["learning_rate"]
        self.max_grad_norm = max_grad_norm
        self.depth_actor_optimizer = optim.Adam(
            [*self.depth_actor.parameters(), *self.depth_encoder.parameters()],
            lr=self.learning_rate,
        )

        self.loss_coef = {
            "action": cfg["coef_action"],
            "yaw": cfg["coef_yaw"],
            "terrain": cfg["coef_terrain"],
            "dyn": cfg["coef_dyn"],
            "vel": cfg["coef_vel"],
            "balance": cfg["coef_balance"],
            "smooth": cfg["coef_smooth"],
        }

    # ----------------------------------------------------------------- update

    def update_depth_actor(self, buffers: dict[str, list[torch.Tensor]]) -> dict[str, float]:
        def _l2(name: str) -> torch.Tensor:
            if not buffers.get(name):
                return torch.zeros((), device=self.device)
            return torch.cat(buffers[name], dim=0).norm(p=2, dim=1).mean()

        def _mse(name: str) -> torch.Tensor:
            if not buffers.get(name):
                return torch.zeros((), device=self.device)
            return torch.cat(buffers[name], dim=0).pow(2).mean()

        def _scalar(name: str) -> torch.Tensor:
            if not buffers.get(name):
                return torch.zeros((), device=self.device)
            return torch.stack(buffers[name]).mean()

        action_loss = _l2("action")
        yaw_loss = _l2("yaw")
        terrain_loss = _mse("terrain")
        dyn_loss = _mse("dyn")
        vel_loss = _mse("vel")
        balance_loss = _scalar("balance")
        smooth_loss = _scalar("smooth")

        c = self.loss_coef
        loss = (
            c["action"] * action_loss
            + c["yaw"] * yaw_loss
            + c["terrain"] * terrain_loss
            + c["dyn"] * dyn_loss
            + c["vel"] * vel_loss
            + c["balance"] * balance_loss
            + c["smooth"] * smooth_loss
        )

        self.depth_actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [*self.depth_actor.parameters(), *self.depth_encoder.parameters()], self.max_grad_norm
        )
        self.depth_actor_optimizer.step()

        return {
            "depth_actor_loss": action_loss.item(),
            "yaw_loss": yaw_loss.item(),
            "terrain_loss": terrain_loss.item(),
            "dyn_loss": dyn_loss.item(),
            "vel_loss": vel_loss.item(),
            "balance_loss": balance_loss.item(),
            "smooth_loss": smooth_loss.item(),
            "total_loss": loss.item(),
        }

    # ------------------------------------------------------------- multi-gpu

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict(), self.depth_encoder.state_dict(), self.depth_actor.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        self.depth_encoder.load_state_dict(model_params[1])
        self.depth_actor.load_state_dict(model_params[2])

    def reduce_parameters(self):
        params = [*self.depth_actor.parameters(), *self.depth_encoder.parameters()]
        grads = [p.grad.view(-1) for p in params if p.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in params:
            if p.grad is not None:
                numel = p.numel()
                p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad.data))
                offset += numel
