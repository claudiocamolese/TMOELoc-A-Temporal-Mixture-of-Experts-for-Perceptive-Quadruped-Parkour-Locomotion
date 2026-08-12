"""Unified multimodal temporal context encoder.

Replaces both temporal modules of the original Extreme Parkour student:

* ``RecurrentDepthBackbone``  (CNN -> GRU -> terrain latent)
* ``StateHistoryEncoder``     (Conv1D over proprioception history -> dynamics latent)

with a single Pre-LN causal Transformer that consumes one multimodal token per
depth frame (10 Hz) and emits a context vector ``c_t`` supervised to contain
terrain geometry, hidden robot dynamics, base velocity and target heading.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .depth_backbone import DepthOnlyFCBackbone58x87


class PreLNTransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block (Xiong et al., 2020)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.norm_ff(x)
        x = x + self.dropout(self.ff(h))
        return x


class TemporalContextEncoder(nn.Module):
    """Depth + proprioception history -> unified context ``c_t``.

    The module owns its temporal state (proprioception window and token window),
    mirroring how ``RecurrentDepthBackbone`` owns its GRU hidden state. Callers
    must invoke :meth:`reset` on episode termination and :meth:`detach_buffers`
    at the end of every optimisation window.
    """

    def __init__(self, depth_cfg: dict):
        super().__init__()
        self.num_prop: int = depth_cfg["num_prop"]
        self.prop_window: int = depth_cfg["prop_window"]
        self.history_len: int = depth_cfg["history_len"]
        self.d_model: int = depth_cfg["d_model"]
        self.depth_latent_dim: int = depth_cfg["depth_latent_dim"]
        self.prop_latent_dim: int = depth_cfg["prop_latent_dim"]
        self.context_dim: int = depth_cfg["context_dim"]

        # -- visual branch: unchanged Extreme Parkour CNN so that the GRU/Transformer
        #    comparison isolates the temporal model.
        self.depth_backbone = DepthOnlyFCBackbone58x87(self.depth_latent_dim)

        # -- proprioceptive branch: last `prop_window` states at 50 Hz (=100 ms)
        self.prop_encoder = nn.Sequential(
            nn.Linear(self.num_prop * self.prop_window, 128),
            nn.ELU(),
            nn.Linear(128, self.prop_latent_dim),
            nn.ELU(),
        )

        # -- fusion into one multimodal token
        self.token_proj = nn.Linear(self.depth_latent_dim + self.prop_latent_dim, self.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.history_len, self.d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        # -- temporal model
        self.blocks = nn.ModuleList(
            [
                PreLNTransformerBlock(
                    self.d_model,
                    depth_cfg["n_heads"],
                    depth_cfg["d_ff"],
                    depth_cfg["dropout"],
                )
                for _ in range(depth_cfg["n_layers"])
            ]
        )
        self.norm_out = nn.LayerNorm(self.d_model)
        self.context_proj = (
            nn.Identity() if self.context_dim == self.d_model else nn.Linear(self.d_model, self.context_dim)
        )

        causal_mask = torch.triu(torch.ones(self.history_len, self.history_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # -- auxiliary heads (training-time supervision, droppable at deployment)
        def _head(out_dim: int, tanh: bool) -> nn.Sequential:
            layers = [nn.Linear(self.context_dim, 64), nn.ELU(), nn.Linear(64, out_dim)]
            if tanh:
                layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        # terrain head matches the teacher scan encoder, whose last layer is Tanh
        self.terrain_head = _head(depth_cfg["num_scan_latent"], tanh=True)
        self.dyn_head = _head(depth_cfg["num_priv_latent_out"], tanh=False)
        self.vel_head = _head(depth_cfg["num_priv_explicit"], tanh=False)
        self.yaw_head = _head(2, tanh=True)

        # temporal state, lazily sized on the first forward pass
        self.prop_buffer = torch.zeros(0, self.prop_window, self.num_prop)
        self.token_buffer = torch.zeros(0, self.history_len, self.d_model)
        self._initialised = False

    # ------------------------------------------------------------------ state

    def _lazy_init(self, num_envs: int, device: torch.device) -> None:
        self.prop_buffer = torch.zeros(num_envs, self.prop_window, self.num_prop, device=device)
        self.token_buffer = torch.zeros(num_envs, self.history_len, self.d_model, device=device)
        self._initialised = True

    def push_proprioception(self, proprioception: torch.Tensor) -> None:
        """Append one 50 Hz proprioceptive sample to the rolling window."""
        if not self._initialised:
            self._lazy_init(proprioception.shape[0], proprioception.device)
            self.prop_buffer = proprioception.detach().unsqueeze(1).repeat(1, self.prop_window, 1)
            return
        self.prop_buffer = torch.cat([self.prop_buffer[:, 1:], proprioception.detach().unsqueeze(1)], dim=1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Clear temporal state for terminated environments.

        The original GRU backbone never reset its hidden state across episode
        boundaries; here we do, so that ``c_t`` never mixes two episodes.
        Branchless on purpose: testing ``mask.any()`` would force a device
        synchronisation on every control step.
        """
        if not self._initialised or dones is None:
            return
        keep = (~dones.bool()).view(-1, 1, 1).to(self.prop_buffer.dtype)
        self.prop_buffer = self.prop_buffer * keep
        self.token_buffer = self.token_buffer * keep

    def detach_buffers(self) -> None:
        self.prop_buffer = self.prop_buffer.detach().clone()
        self.token_buffer = self.token_buffer.detach().clone()

    # ---------------------------------------------------------------- forward

    def encode_token(self, depth_image: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        """Build one multimodal token from a depth frame + the proprio window."""
        if not self._initialised:
            self._lazy_init(depth_image.shape[0], depth_image.device)
            self.prop_buffer = proprioception.detach().unsqueeze(1).repeat(1, self.prop_window, 1)

        z_v = self.depth_backbone(depth_image)
        # newest sample is the live (differentiable) observation, the rest is history
        window = torch.cat([self.prop_buffer[:, 1:], proprioception.unsqueeze(1)], dim=1)
        z_p = self.prop_encoder(window.reshape(window.shape[0], -1))
        return self.token_proj(torch.cat([z_v, z_p], dim=-1))

    def forward(self, depth_image: torch.Tensor, proprioception: torch.Tensor) -> dict[str, torch.Tensor]:
        """Advance the temporal window by one depth frame and return the context."""
        token = self.encode_token(depth_image, proprioception)
        self.token_buffer = torch.cat([self.token_buffer[:, 1:], token.unsqueeze(1)], dim=1)

        x = self.token_buffer + self.pos_embedding
        for block in self.blocks:
            x = block(x, attn_mask=self.causal_mask)
        c_t = self.context_proj(self.norm_out(x[:, -1]))

        return {
            "context": c_t,
            "terrain": self.terrain_head(c_t),
            "dyn": self.dyn_head(c_t),
            "vel": self.vel_head(c_t),
            # same 1.5 scaling as the original depth encoder yaw output
            "yaw": 1.5 * self.yaw_head(c_t),
        }
