"""Simulator-independent PPO components for the frozen-perception refinement stage.

Policy inputs are collected under no_grad and replayed as vectors. GAE therefore
operates on stored scalar rewards/values; the critic never enters the deployed
student policy. Checkpoint keys and the critic layer layout remain unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


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
        """Return one state-value estimate per privileged observation."""
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


def flatten_rollout(tensor: torch.Tensor) -> torch.Tensor:
    """Merge (time, environment) axes while retaining trailing feature dimensions."""
    return tensor.reshape(-1, *tensor.shape[2:])


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unnormalized advantages and value targets for (time, env) rollouts.

    ``rewards`` must already contain the environment timeout bootstrap credit.
    ``dones`` still stops the recursive estimate at every reset, preventing the
    next episode from contributing to the previous trajectory. Normalization is
    left to the caller so the original PPO batch convention stays explicit.
    """
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for step in reversed(range(rewards.shape[0])):
        next_value = last_value if step == rewards.shape[0] - 1 else values[step + 1]
        continuation = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * continuation - values[step]
        gae = delta + gamma * lam * continuation * gae
        advantages[step] = gae
    return advantages, advantages + values
