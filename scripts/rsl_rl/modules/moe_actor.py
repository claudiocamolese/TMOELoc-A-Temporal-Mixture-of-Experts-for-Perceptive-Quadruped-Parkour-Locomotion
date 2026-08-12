"""Sparse Top-k Mixture-of-Experts actor for quadruped parkour control.

The router observes ``u_t = [proprioception ; context ; estimated base velocity]``
and mixes ``K`` expert MLPs into a single joint-position target. Routing is
refreshed at the perception rate (10 Hz) and cached in between, while the
experts themselves run at the control rate (50 Hz) on fresh proprioception.
No expert is ever told which skill it should represent: specialisation is
expected to emerge from the context.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusedExperts(nn.Module):
    """``num_experts`` identical MLPs evaluated as a batched matmul.

    Mathematically identical to a ``ModuleList`` of ``nn.Sequential`` MLPs
    (including the initialisation), but it issues one ``baddbmm`` per layer
    instead of one GEMM per expert per layer. At these tensor sizes the actor is
    kernel-launch bound, so the fused form is markedly faster.
    """

    def __init__(self, num_experts: int, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, act):
        super().__init__()
        self.num_experts = num_experts
        dims = [in_dim, *hidden_dims, out_dim]
        self.act = act()
        weights, biases = [], []
        for i in range(len(dims) - 1):
            w = torch.empty(num_experts, dims[i], dims[i + 1])
            b = torch.empty(num_experts, 1, dims[i + 1])
            for k in range(num_experts):
                ref = nn.Linear(dims[i], dims[i + 1])
                w[k] = ref.weight.data.t()
                b[k, 0] = ref.bias.data
            weights.append(nn.Parameter(w))
            biases.append(nn.Parameter(b))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """(B, in) -> (B, num_experts, out)."""
        h = u.unsqueeze(0).expand(self.num_experts, -1, -1)
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b, h, w)
            if i < last:
                h = self.act(h)
        return h.transpose(0, 1)


class MoEActor(nn.Module):
    def __init__(
        self,
        num_prop: int,
        context_dim: int,
        num_priv_explicit: int,
        num_actions: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dims: tuple[int, ...] = (256, 128),
        router_hidden_dim: int = 128,
        use_vel_in_input: bool = True,
        router_noise_std: float = 0.3,
        activation: str = "elu",
    ):
        super().__init__()
        self.num_prop = num_prop
        self.context_dim = context_dim
        self.num_priv_explicit = num_priv_explicit if use_vel_in_input else 0
        self.use_vel_in_input = use_vel_in_input
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.in_features = num_prop + context_dim + self.num_priv_explicit

        act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        self.router_noise_std = router_noise_std
        self.router = nn.Sequential(
            nn.Linear(self.in_features, router_hidden_dim),
            act(),
            nn.Linear(router_hidden_dim, num_experts),
        )
        # Small (but not degenerate) init: routing starts near-uniform and
        # specialises gradually. With a gain of ~0.01 the logits are identical
        # for every environment and top-k degenerates into a constant choice,
        # which starves the unselected experts of gradient.
        nn.init.orthogonal_(self.router[-1].weight, gain=0.1)
        nn.init.zeros_(self.router[-1].bias)

        self.experts = FusedExperts(num_experts, self.in_features, tuple(expert_hidden_dims), num_actions, act)

        # cached routing state (refreshed at perception rate)
        self.gates: torch.Tensor | None = None  # sparse, renormalised top-k weights
        self.probs: torch.Tensor | None = None  # dense softmax, for the aux losses
        self.prev_probs: torch.Tensor | None = None

    # ------------------------------------------------------------------ utils

    def build_input(
        self, proprioception: torch.Tensor, context: torch.Tensor, vel: torch.Tensor | None = None
    ) -> torch.Tensor:
        parts = [proprioception, context]
        if self.use_vel_in_input:
            if vel is None:
                raise ValueError("MoEActor was configured with use_vel_in_input=True but no velocity was given.")
            parts.append(vel)
        return torch.cat(parts, dim=-1)

    def update_routing(self, u: torch.Tensor) -> None:
        """Recompute the gate distribution. Called once per perception step."""
        logits = self.router(u)
        if self.training and self.router_noise_std > 0.0:
            # noisy top-k gating (Shazeer et al., 2017): keeps every expert in
            # play early on, when the router cannot yet tell states apart
            logits = logits + self.router_noise_std * torch.randn_like(logits)
        probs = torch.softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        gates = torch.zeros_like(probs)
        gates.scatter_(1, topk_idx, topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8))

        self.prev_probs = self.probs
        self.probs = probs
        self.gates = gates

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Fall back to uniform mixing for environments that just terminated.

        Branchless on purpose: testing ``mask.any()`` would force a device
        synchronisation on every control step.
        """
        if dones is None or self.gates is None:
            return
        mask = dones.bool().view(-1, 1)
        uniform = 1.0 / self.num_experts
        self.gates = torch.where(mask, torch.full_like(self.gates, uniform), self.gates)
        if self.prev_probs is not None:
            self.prev_probs = torch.where(
                mask, torch.full_like(self.prev_probs, uniform), self.prev_probs
            )

    def detach_routing(self) -> None:
        if self.gates is not None:
            self.gates = self.gates.detach().clone()
        if self.probs is not None:
            self.probs = self.probs.detach().clone()
        self.prev_probs = None

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        proprioception: torch.Tensor,
        context: torch.Tensor,
        vel: torch.Tensor | None = None,
        update_routing: bool = False,
    ) -> torch.Tensor:
        u = self.build_input(proprioception, context, vel)
        if update_routing or self.gates is None:
            self.update_routing(u)

        # With K=4 experts on a few hundred environments, evaluating every expert
        # and weighting by the sparse gate is faster than gathering per-expert
        # batches, and is numerically identical to top-k dispatch.
        expert_out = self.experts(u)  # (B, K, A)
        return (self.gates.unsqueeze(-1) * expert_out).sum(dim=1)

    # ------------------------------------------------------------ aux losses

    def load_balancing_loss(self) -> torch.Tensor:
        """Switch-Transformer style auxiliary loss (Fedus et al., 2021)."""
        if self.probs is None:
            return torch.zeros((), device=next(self.parameters()).device)
        fraction = (self.gates > 0).float().mean(dim=0)  # dispatch fraction per expert
        mean_prob = self.probs.mean(dim=0)
        return self.num_experts * torch.sum(fraction.detach() * mean_prob)

    def routing_smoothness_loss(self) -> torch.Tensor:
        """Penalise abrupt expert switching between consecutive perception steps."""
        if self.probs is None or self.prev_probs is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return ((self.probs - self.prev_probs.detach()) ** 2).sum(dim=-1).mean()

    @torch.no_grad()
    def routing_stats(self) -> dict[str, float]:
        if self.probs is None:
            return {}
        usage = (self.gates > 0).float().mean(dim=0)
        entropy = -(self.probs * torch.log(self.probs + 1e-8)).sum(dim=-1).mean()
        stats = {f"expert_{i}_usage": usage[i].item() for i in range(self.num_experts)}
        stats["routing_entropy"] = entropy.item()
        return stats
