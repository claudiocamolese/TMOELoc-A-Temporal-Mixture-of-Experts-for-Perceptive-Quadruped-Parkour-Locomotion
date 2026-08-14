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
from .optim_utils import RunningStandardiser, SpikeGuard, warmup_cosine_lr


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
        self.dyn_target = cfg.get("dyn_target", "raw_params")
        self.dyn_dims = list(cfg.get("dyn_dims", [0, 1, 2, 3, 4]))
        cfg["num_dyn_targets"] = (
            len(self.dyn_dims) if self.dyn_target == "raw_params" else cfg["num_priv_latent_out"]
        )
        self.dyn_normaliser = RunningStandardiser(cfg["num_dyn_targets"])
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

        self.peak_lr = self.learning_rate
        self.warmup_iters = cfg["warmup_iters"]
        self.total_iters = cfg["total_iters"]
        self.min_lr_ratio = cfg["min_lr_ratio"]
        self.spike_guard = SpikeGuard(factor=cfg["spike_factor"])
        self.last_grad_norm = 0.0
        # distillation inherits the teacher's iteration counter, so the schedule
        # is keyed on iterations since this run started
        self.schedule_offset = 0

        self.loss_coef = {
            "action": cfg["coef_action"],
            "yaw": cfg["coef_yaw"],
            "terrain": cfg["coef_terrain"],
            "dyn": cfg["coef_dyn"],
            "vel": cfg["coef_vel"],
            "balance": cfg["coef_balance"],
            "smooth": cfg["coef_smooth"],
            "zloss": cfg["coef_zloss"],
            "risk": cfg.get("coef_risk", 0.0),
        }
        self.use_risk_head = bool(cfg.get("use_risk_head", False))
        self.risk_weight = float(cfg.get("risk_weight", 0.0))
        self.risk_weight_cap = float(cfg.get("risk_weight_cap", 5.0))
        self.risk_weight_warmup = int(cfg.get("risk_weight_warmup", 0))
        self._iteration = 0

    def warm_start_experts_from_teacher(self) -> bool:
        """Seed every expert's upper layers with the teacher actor's own.

        The GRU baseline copies the teacher's actor wholesale into its student
        and therefore begins with a working controller, needing only to learn the
        perception that feeds it. Measured, that is worth a lot: it reaches a
        given terrain level roughly twice as fast, and it still leads at
        convergence. The mixture cannot copy wholesale -- an expert reads
        [proprioception; context; velocity], 190 dims, where the teacher's
        backbone reads 114 including the privileged scandots latent -- but the
        shapes line up from the second layer on:

            teacher actor_backbone  114 -> 512 -> 256 -> 128 -> 12
            expert                         190 -> 256 -> 128 -> 12

        so the mapping from internal features to joint targets transfers exactly,
        and only the first layer has to learn to produce features in the
        teacher's own representation. Experts stay distinct because their first
        layers keep their independent random initialisation.

        Must be called *after* the teacher checkpoint is loaded -- at
        construction time the policy still holds random weights -- and never on a
        resume, where it would overwrite trained experts.
        """
        backbone = getattr(self.policy.actor, "actor_backbone", None)
        if backbone is None:
            print("[WARN]: teacher has no actor_backbone; skipping expert warm start")
            return False
        linears = [m for m in backbone if isinstance(m, nn.Linear)]
        experts = self.depth_actor.experts
        # pair the teacher's trailing layers with the expert's, back to front
        pairs = list(zip(reversed(linears), reversed(range(len(experts.weights)))))
        copied = 0
        with torch.no_grad():
            for src, idx in pairs:
                w, b = experts.weights[idx], experts.biases[idx]
                # FusedExperts stores (num_experts, in, out); nn.Linear is (out, in)
                if tuple(w.shape[1:]) != (src.in_features, src.out_features):
                    break
                w.copy_(src.weight.data.t().unsqueeze(0).expand_as(w))
                b.copy_(src.bias.data.view(1, 1, -1).expand_as(b))
                copied += 1
        print(f"[INFO]: warm-started {copied} expert layer(s) from the teacher actor "
              f"({'x'.join(str(d) for d in (self.depth_actor.in_features,))} -> ... -> "
              f"{linears[-1].out_features})")
        return copied > 0

    # ----------------------------------------------------------------- update

    def current_risk_weight(self) -> float:
        """Risk weighting, ramped in linearly once the head has had time to learn.

        Trusting an untrained head to say which timesteps matter would just add
        noise to the action loss for the first few thousand iterations.
        """
        if not self.use_risk_head or self.risk_weight <= 0.0:
            return 0.0
        it = self._iteration - self.schedule_offset
        if self.risk_weight_warmup <= 0:
            return self.risk_weight
        return self.risk_weight * min(1.0, max(0.0, it / self.risk_weight_warmup))

    def set_iteration(self, iteration: int) -> None:
        """Apply the warmup/cosine learning-rate schedule for this iteration."""
        self._iteration = iteration
        self.learning_rate = warmup_cosine_lr(
            iteration - self.schedule_offset, self.peak_lr, self.warmup_iters, self.total_iters, self.min_lr_ratio
        )
        for group in self.depth_actor_optimizer.param_groups:
            group["lr"] = self.learning_rate

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

        # -- action loss, optionally weighted by how dangerous the step is ----
        #
        # The unweighted mean is dominated by ordinary trotting, which the
        # student already imitates well, while the steps that end episodes are
        # a fraction of a percent of the batch. Scaling each sample by its
        # predicted risk moves gradient onto those steps without changing the
        # overall step size: the weights are renormalised to mean 1.
        w = self.current_risk_weight()
        if w > 0.0 and buffers.get("risk_pred") and buffers.get("action"):
            resid = torch.cat(buffers["action"], dim=0).norm(p=2, dim=1)
            risk = torch.sigmoid(torch.cat(buffers["risk_pred"], dim=0).detach())
            weight = (1.0 + w * risk).clamp(max=self.risk_weight_cap)
            weight = weight / weight.mean().clamp_min(1e-8)
            action_loss = (weight * resid).mean()
            # The weighted mean is what the optimiser sees, but it is not
            # comparable to a run without weighting: if the dangerous states are
            # also the hardest to imitate -- the premise of the whole term -- the
            # weighted mean sits above the plain one by construction. Log both,
            # so "did imitation get worse" stays answerable across runs.
            action_loss_unweighted = resid.mean().detach()
        else:
            action_loss = _l2("action")
            action_loss_unweighted = action_loss.detach()
        yaw_loss = _l2("yaw")
        terrain_loss = _mse("terrain")
        dyn_loss = _mse("dyn")
        vel_loss = _mse("vel")
        balance_loss = _scalar("balance")
        smooth_loss = _scalar("smooth")
        z_loss = _scalar("zloss")

        # -- risk head: does this state fall within [lead, horizon) from now ---
        # Positives are well under 1% of samples, so the loss is balanced by the
        # observed positive rate; without that the head converges to "never".
        if self.use_risk_head and buffers.get("risk_pred") and buffers.get("risk_label"):
            logit = torch.cat(buffers["risk_pred"], dim=0)
            label = torch.cat(buffers["risk_label"], dim=0).float()
            valid = torch.cat(buffers["risk_valid"], dim=0).bool()
            if valid.any():
                logit, label = logit[valid], label[valid]
                pos = label.mean().clamp(1e-4, 1 - 1e-4)
                pos_weight = ((1 - pos) / pos).clamp(max=100.0)
                risk_loss = nn.functional.binary_cross_entropy_with_logits(
                    logit, label, pos_weight=pos_weight
                )
            else:
                risk_loss = torch.zeros((), device=self.device)
        else:
            risk_loss = torch.zeros((), device=self.device)

        c = self.loss_coef
        loss = (
            c["action"] * action_loss
            + c["yaw"] * yaw_loss
            + c["terrain"] * terrain_loss
            + c["dyn"] * dyn_loss
            + c["vel"] * vel_loss
            + c["balance"] * balance_loss
            + c["smooth"] * smooth_loss
            + c["zloss"] * z_loss
            + c["risk"] * risk_loss
        )

        params = [*self.depth_actor.parameters(), *self.depth_encoder.parameters()]
        self.depth_actor_optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.last_grad_norm = float(grad_norm)
        # A spike here is what killed an earlier run; skipping the step costs one
        # batch, letting it through costs the optimiser state.
        skipped = self.spike_guard.should_skip(loss.item())
        if not skipped:
            self.depth_actor_optimizer.step()

        return {
            "depth_actor_loss": action_loss.item(),
            "depth_actor_loss_unweighted": action_loss_unweighted.item(),
            "yaw_loss": yaw_loss.item(),
            "terrain_loss": terrain_loss.item(),
            "dyn_loss": dyn_loss.item(),
            "vel_loss": vel_loss.item(),
            "balance_loss": balance_loss.item(),
            "smooth_loss": smooth_loss.item(),
            "z_loss": z_loss.item(),
            "risk_loss": risk_loss.item(),
            "risk_weight": w,
            "total_loss": loss.item(),
            "grad_norm": self.last_grad_norm,
            "lr": self.learning_rate,
            "steps_skipped": float(self.spike_guard.num_skipped),
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
