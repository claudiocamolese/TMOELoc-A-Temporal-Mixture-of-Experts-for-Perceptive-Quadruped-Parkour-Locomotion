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


class TeacherPolicy:
    """The phase-1 teacher, exposed through the same interface as a student.

    This is the ceiling for distillation: no student can regress its way past
    the policy it is imitating. Evaluating it on the *same* environment as the
    students is what makes their MXD interpretable.

    Defaults match the teacher exactly as the distillation runner queries it --
    ``hist_encoding=True`` and the true privileged velocity straight from the
    observation -- so the number really is the target the students chase.
    """

    variant = "teacher"

    def __init__(self, runner, agent_cfg: dict, device, use_estimator: bool = False, **_):
        self.device = device
        self.runner = runner
        self.alg = runner.alg
        self.policy = runner.alg.policy
        self.estimator = getattr(runner.alg, "estimator", None)
        self.use_estimator = use_estimator
        est = agent_cfg["estimator"]
        self.num_prop = est["num_prop"]
        self.num_scan = est["num_scan"]
        self.num_priv_explicit = est["num_priv_explicit"]
        self.priv_explicit_slice = slice(
            self.num_prop + self.num_scan, self.num_prop + self.num_scan + self.num_priv_explicit
        )
        self.vel_source = "true"
        self.forced_expert = None
        self.last_gates = None
        self.last_context = None
        self.policy.eval()

    def force_expert(self, index):
        if index is not None:
            raise ValueError("--force_expert does not apply to the teacher.")

    def reset(self, dones=None):
        pass

    @torch.no_grad()
    def act(self, obs: torch.Tensor, depth_image: torch.Tensor, is_perception_step: bool) -> torch.Tensor:
        if self.use_estimator and self.estimator is not None:
            obs[:, self.priv_explicit_slice] = self.estimator.inference(obs[:, : self.num_prop])
        actions = self.policy.act_inference(obs, hist_encoding=True, scandots_latent=None)
        # the teacher has no context; keep the field shaped for the trace writer
        self.last_context = obs[:, : self.num_prop].clone()
        return actions


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
        # optional causal test: directions to project out of the context, so that
        # "c_t encodes the dynamics" can be separated from "the actor uses it"
        self.ablation_basis: torch.Tensor | None = None
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

    def ablate_context(self, directions) -> None:
        """Project the given directions out of the context before the actor sees it.

        ``directions`` is (k, context_dim). They are orthonormalised first, so
        overlapping probe directions are removed once rather than twice.
        """
        if directions is None:
            self.ablation_basis = None
            return
        d = torch.as_tensor(directions, dtype=torch.float32, device=self.device)
        if d.ndim == 1:
            d = d[None, :]
        q, _ = torch.linalg.qr(d.T)          # (context_dim, k) orthonormal columns
        self.ablation_basis = q
        print(f"[INFO]: ablating a {q.shape[1]}-dimensional subspace of the {q.shape[0]}-d context")

    def _apply_ablation(self, context: torch.Tensor) -> torch.Tensor:
        if self.ablation_basis is None:
            return context
        return context - (context @ self.ablation_basis) @ self.ablation_basis.T

    def reset(self, dones=None) -> None:
        if dones is None:
            return
        # Both students clear their temporal state on episode boundaries: the
        # Transformer its token window, the GRU its hidden state. Evaluation
        # mirrors training for each.
        self.depth_encoder.reset(dones)
        if self.is_moe:
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
                self.context = self._apply_ablation(out["context"])
                self.vel_pred, self.yaw_pred = out["vel"], out["yaw"]
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
