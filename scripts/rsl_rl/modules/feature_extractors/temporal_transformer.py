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
        # Analysis-only. When set, the learned attention distribution is replaced
        # by a causal uniform average over the window, leaving every other weight
        # untouched. It answers the question the attention plot only hints at: if
        # performance is unchanged, the self-attention is doing the job of a mean
        # pool and the "Transformer" claim rests on the window, not the routing
        # of information inside it.
        self.uniform_attention = False

    def _uniform_attn(self, h: torch.Tensor) -> torch.Tensor:
        """Exactly the attention sublayer, with the softmax replaced by 1/(i+1).

        Uniform weights are identical across heads, so the per-head split is a
        no-op here and the value projection can be averaged directly before the
        output projection.
        """
        e = h.shape[-1]
        w = self.attn.in_proj_weight[2 * e :]  # value projection
        b = self.attn.in_proj_bias[2 * e :] if self.attn.in_proj_bias is not None else None
        v = torch.nn.functional.linear(h, w, b)  # (B, T, E)
        counts = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=v.dtype).view(1, -1, 1)
        v = v.cumsum(dim=1) / counts  # causal running mean
        return self.attn.out_proj(v)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, need_weights: bool = False
    ) -> torch.Tensor:
        """Apply causal attention and the feed-forward residual to a (batch, time, width) tensor."""
        h = self.norm_attn(x)
        if self.uniform_attention:
            attn_out = self._uniform_attn(h)
            t = h.shape[1]
            tri = torch.tril(torch.ones(t, t, device=h.device))
            self.last_attention = (tri / tri.sum(-1, keepdim=True)).expand(h.shape[0], t, t)
            x = x + self.dropout(attn_out)
            h = self.norm_ff(x)
            return x + self.dropout(self.ff(h))
        attn_out, weights = self.attn(
            h, h, h, attn_mask=attn_mask, need_weights=need_weights, average_attn_weights=True
        )
        # (batch, query, key) averaged over heads; kept for the attention probe
        self.last_attention = weights
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
            nn.Identity()
            if self.context_dim == self.d_model
            else nn.Linear(self.d_model, self.context_dim)
        )

        causal_mask = torch.triu(
            torch.ones(self.history_len, self.history_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # Training supervises all readouts. Heading is also required by the
        # actor at inference; velocity is required when vel_source='predicted'.
        def _head(out_dim: int, tanh: bool) -> nn.Sequential:
            """Construct a context readout, with bounded output where the target requires it."""
            layers = [nn.Linear(self.context_dim, 64), nn.ELU(), nn.Linear(64, out_dim)]
            if tanh:
                layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        # terrain head matches the teacher scan encoder, whose last layer is Tanh
        self.terrain_head = _head(depth_cfg["num_scan_latent"], tanh=True)
        # The dynamics head regresses the standardised physical parameters, not
        # the teacher's privileged latent. Measured on the released teacher:
        # its priv_encoder maps mass (std 1.13 across environments) and friction
        # (std 0.41) onto a *constant* 20-d vector -- across-environment std
        # 0.00000 on every one of the 20 dimensions. Regressing that target
        # teaches the context nothing; the head just learns a constant.
        self.dyn_head = _head(depth_cfg["num_dyn_targets"], tanh=False)
        self.vel_head = _head(depth_cfg["num_priv_explicit"], tanh=False)
        self.yaw_head = _head(2, tanh=True)
        # Optional: one logit for a fall in the configured future risk window.
        # Built only when enabled, so a checkpoint trained without it still loads
        # into this class unchanged.
        self.use_risk_head = bool(depth_cfg.get("use_risk_head", False))
        self.risk_head = _head(1, tanh=False) if self.use_risk_head else None

        # temporal state, lazily sized on the first forward pass
        self.prop_buffer = torch.zeros(0, self.prop_window, self.num_prop)
        self.token_buffer = torch.zeros(0, self.history_len, self.d_model)
        self._initialised = False
        # The depth camera runs at 10 Hz and the controller at 50 Hz. With the
        # context refreshed only on perception steps, the actor spends four
        # control steps out of five with a context up to 80 ms stale and no other
        # temporal information at all -- its own input is a single 53-dim
        # proprioception frame, where the GRU baseline's actor reads a 530-dim
        # proprioceptive history every step. A fall develops in 100-200 ms, so
        # that staleness is a large fraction of the timescale to react on.
        #
        # At "control" rate the visual latent stays on the camera's clock, cached
        # between frames, while the proprioceptive branch, the fusion and the
        # transformer recompute every control step. The window still spans 0.8 s:
        # the seven older tokens stay at 10 Hz and only the newest is live.
        # History at the camera's rate, present at the controller's.
        self.context_rate = depth_cfg.get("context_rate", "perception")
        self.depth_latent_cache: torch.Tensor | None = None

        # attention probe, off during training (need_weights costs a softmax copy)
        self.capture_attention = False
        self.last_attention: torch.Tensor | None = None

    # ------------------------------------------------------------------ state

    def _lazy_init(self, num_envs: int, device: torch.device) -> None:
        """Allocate per-environment temporal buffers on the input device."""
        self.prop_buffer = torch.zeros(num_envs, self.prop_window, self.num_prop, device=device)
        self.token_buffer = torch.zeros(num_envs, self.history_len, self.d_model, device=device)
        self._initialised = True

    def push_proprioception(self, proprioception: torch.Tensor) -> None:
        """Append one 50 Hz proprioceptive sample to the rolling window."""
        if not self._initialised:
            self._lazy_init(proprioception.shape[0], proprioception.device)
            self.prop_buffer = proprioception.detach().unsqueeze(1).repeat(1, self.prop_window, 1)
            return
        self.prop_buffer = torch.cat(
            [self.prop_buffer[:, 1:], proprioception.detach().unsqueeze(1)], dim=1
        )

    def reset_depth_cache(self) -> None:
        """Invalidate cached visual features before the next perception update."""
        self.depth_latent_cache = None

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
        """Truncate rollout autograd history while preserving the temporal buffer values."""
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

    def forward(
        self,
        depth_image: torch.Tensor,
        proprioception: torch.Tensor,
        is_perception_step: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return the context, advancing the 10 Hz window only on a new frame."""
        if not self._initialised:
            self._lazy_init(depth_image.shape[0], depth_image.device)
            self.prop_buffer = proprioception.detach().unsqueeze(1).repeat(1, self.prop_window, 1)

        fresh_frame = is_perception_step or self.depth_latent_cache is None
        if fresh_frame:
            self.depth_latent_cache = self.depth_backbone(depth_image)

        window = torch.cat([self.prop_buffer[:, 1:], proprioception.unsqueeze(1)], dim=1)
        z_p = self.prop_encoder(window.reshape(window.shape[0], -1))
        token = self.token_proj(torch.cat([self.depth_latent_cache, z_p], dim=-1))

        if fresh_frame:
            # commit: the previous live token becomes history, this one is live
            self.token_buffer = torch.cat([self.token_buffer[:, 1:], token.unsqueeze(1)], dim=1)
        else:
            # refresh the live slot in place, leaving the 10 Hz history untouched
            self.token_buffer = torch.cat([self.token_buffer[:, :-1], token.unsqueeze(1)], dim=1)

        x = self.token_buffer + self.pos_embedding
        for block in self.blocks:
            x = block(x, attn_mask=self.causal_mask, need_weights=self.capture_attention)
        c_t = self.context_proj(self.norm_out(x[:, -1]))

        if self.capture_attention:
            # attention paid by the newest token to each position of the window.
            # This is what decides whether the 0.8 s window is used at all, or
            # whether the model is effectively reading only the current frame.
            self.last_attention = torch.stack(
                [b.last_attention[:, -1, :] for b in self.blocks], dim=1
            )  # (batch, layers, history_len)

        out = {
            "context": c_t,
            "terrain": self.terrain_head(c_t),
            "dyn": self.dyn_head(c_t),
            "vel": self.vel_head(c_t),
            # same 1.5 scaling as the original depth encoder yaw output
            "yaw": 1.5 * self.yaw_head(c_t),
        }
        if self.risk_head is not None:
            out["risk"] = self.risk_head(c_t).squeeze(-1)  # logit, (batch,)
        return out
