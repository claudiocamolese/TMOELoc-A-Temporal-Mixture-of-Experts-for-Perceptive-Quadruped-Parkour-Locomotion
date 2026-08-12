"""Uniform inference wrapper for both distillation students.

`play.py` and `evaluation.py` upstream branch on
``algorithm.class_name != "DistillationWithExtractor"``, which silently routes
any *other* student -- including the MoE one -- into the teacher branch. That
does not crash: it runs the privileged scandots teacher and produces a perfectly
convincing video of the wrong policy. Everything downstream of a checkpoint
should go through this wrapper instead, which fails loudly on an unknown
variant.
"""

from __future__ import annotations

import torch

# see on_policy_runner_moe for the layout of the 53-dim proprioception vector
YAW_SLICE = slice(6, 8)
ENV_CLASS_SLICE = slice(11, 13)


class StudentPolicy:
    """Wraps a trained student so callers do not care which variant it is."""

    def __init__(self, runner, agent_cfg: dict, device, vel_source: str | None = None):
        self.device = device
        self.variant = agent_cfg["algorithm"]["class_name"]
        if self.variant not in ("DistillationWithExtractor", "DistillationMoE"):
            raise ValueError(
                f"{self.variant} is not a distillation student. Refusing to run: the upstream "
                "scripts would silently fall back to the privileged teacher here."
            )
        self.runner = runner
        self.alg = runner.alg
        self.depth_encoder = runner.alg.depth_encoder
        self.depth_actor = runner.alg.depth_actor

        enc_cfg = agent_cfg["depth_encoder"]
        est_cfg = agent_cfg["estimator"]
        self.num_prop = est_cfg["num_prop"]
        self.num_scan = est_cfg["num_scan"]
        self.num_priv_explicit = est_cfg["num_priv_explicit"]
        self.priv_explicit_slice = slice(
            self.num_prop + self.num_scan, self.num_prop + self.num_scan + self.num_priv_explicit
        )
        self.zero_env_class = bool(enc_cfg.get("zero_env_class", True))
        self.vel_source = vel_source or enc_cfg.get("vel_source", "true")

        self.is_moe = self.variant == "DistillationMoE"
        self.forced_expert: int | None = None

        self.context = None
        self.vel_pred = None
        self.yaw_pred = None
        self.depth_latent = None
        # last-step diagnostics, filled on every act()
        self.last_gates = None
        self.last_context = None

        self.depth_encoder.eval()
        self.depth_actor.eval()

    # ------------------------------------------------------------------ utils

    def force_expert(self, index: int | None) -> None:
        """Pin routing to a single expert, to check what it does on its own."""
        if index is not None and not self.is_moe:
            raise ValueError("--force_expert only applies to the Mixture-of-Experts student.")
        self.forced_expert = index

    def reset(self, dones=None) -> None:
        if not self.is_moe or dones is None:
            # NB: the GRU backbone is deliberately left alone. It was never reset
            # across episodes during training either, and evaluation has to match.
            return
        self.depth_encoder.reset(dones)
        self.depth_actor.reset(dones)

    def _student_prop(self, obs: torch.Tensor, blank_yaw: bool) -> torch.Tensor:
        prop = obs[:, : self.num_prop].clone()
        if blank_yaw:
            prop[:, YAW_SLICE] = 0.0
        if self.zero_env_class:
            prop[:, ENV_CLASS_SLICE] = 0.0
        return prop

    # ---------------------------------------------------------------- forward

    @torch.no_grad()
    def act(self, obs: torch.Tensor, depth_image: torch.Tensor, is_perception_step: bool) -> torch.Tensor:
        prop_ctx = self._student_prop(obs, blank_yaw=True)

        if self.is_moe:
            if is_perception_step or self.context is None:
                out = self.depth_encoder(depth_image, prop_ctx)
                self.context, self.vel_pred, self.yaw_pred = out["context"], out["vel"], out["yaw"]
            self.depth_encoder.push_proprioception(prop_ctx)
            obs[:, YAW_SLICE] = self.yaw_pred

            prop_actor = self._student_prop(obs, blank_yaw=False)
            vel = obs[:, self.priv_explicit_slice] if self.vel_source == "true" else self.vel_pred

            if self.forced_expert is None:
                actions = self.depth_actor(
                    prop_actor, self.context, vel=vel, update_routing=is_perception_step
                )
            else:
                # bypass the router entirely: run this one expert on its own
                gates = torch.zeros(
                    prop_actor.shape[0], self.depth_actor.num_experts, device=prop_actor.device
                )
                gates[:, self.forced_expert] = 1.0
                self.depth_actor.gates = gates
                actions = self.depth_actor(prop_actor, self.context, vel=vel, update_routing=False)
            self.last_gates = self.depth_actor.gates.clone()
            self.last_context = self.context.clone()
            return actions

        # original GRU + dense student
        if is_perception_step or self.depth_latent is None:
            latent_and_yaw = self.depth_encoder(depth_image, prop_ctx)
            self.depth_latent = latent_and_yaw[:, :-2]
            self.yaw_pred = 1.5 * latent_and_yaw[:, -2:]
        obs[:, YAW_SLICE] = self.yaw_pred
        masked = self.runner.mask_student_obs(obs)
        self.last_context = self.depth_latent.clone()
        return self.depth_actor(masked, hist_encoding=True, scandots_latent=self.depth_latent)
