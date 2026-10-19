"""Behavioral checks for routing, temporal context, PPO returns, and released weights."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
from _support import RELEASED_CHECKPOINT, load_core, released_rollout
from torch import nn

moe = load_core("moe_actor")
temporal = load_core("feature_extractors.temporal_transformer")
ppo = load_core("ppo_utils")
optim = load_core("optim_utils")


class RoutingTests(unittest.TestCase):
    """Check the semantics that PPO and control-rate gate caching depend on."""

    def setUp(self):
        """Use a small deterministic actor, keeping training noise out of comparisons."""
        torch.manual_seed(7)
        self.actor = moe.MoEActor(3, 4, 2, 2, expert_hidden_dims=(8, 4)).eval()
        self.prop, self.context, self.vel = torch.randn(5, 3), torch.randn(5, 4), torch.randn(5, 2)

    def test_fused_experts_match_independent_networks(self):
        """Batched kernels must equal four independent MLP evaluations."""
        fused = self.actor.experts
        inputs = self.actor.build_input(self.prop, self.context, self.vel)
        expected = []
        for expert in range(4):
            output = inputs
            for layer, (weight, bias) in enumerate(zip(fused.weights, fused.biases)):
                output = nn.functional.linear(output, weight[expert].T, bias[expert, 0])
                if layer < len(fused.weights) - 1:
                    output = nn.functional.elu(output)
            expected.append(output)
        torch.testing.assert_close(fused(inputs), torch.stack(expected, dim=1))

    def test_cached_top_two_gates_and_reset(self):
        """Control updates reuse gates, and reset affects only terminated environments."""
        self.actor(self.prop, self.context, self.vel, update_routing=True)
        gates = self.actor.gates.clone()
        self.assertTrue(torch.all((gates > 0).sum(-1) == 2))
        torch.testing.assert_close(gates.sum(-1), torch.ones(5))
        self.actor(self.prop * 4, self.context, self.vel)
        torch.testing.assert_close(self.actor.gates, gates)
        self.actor.reset(torch.tensor([True, False, False, False, False]))
        torch.testing.assert_close(self.actor.gates[0], torch.full((4,), 0.25))
        torch.testing.assert_close(self.actor.gates[1:], gates[1:])

    def test_replay_uses_original_routing_input(self):
        """PPO replay must reproduce the rollout distribution with unchanged weights."""
        route_input = self.actor.build_input(self.prop, self.context, self.vel)
        self.actor(self.prop, self.context, self.vel, update_routing=True)
        rollout = self.actor(self.prop * 3, self.context, self.vel)
        replay = self.actor(self.prop * 3, self.context, self.vel, route_u=route_input)
        torch.testing.assert_close(replay, rollout)

    def test_dense_balance_keeps_a_router_gradient(self):
        """Dense routing regularization must not become a constant Switch loss."""
        self.actor.top_k = 4
        self.actor(self.prop, self.context, self.vel, update_routing=True)
        self.actor.load_balancing_loss().backward()
        self.assertGreater(self.actor.router[-1].weight.grad.abs().sum().item(), 0)

    def test_invalid_routing_configuration_fails_early(self):
        """Zero experts or an empty top-k cannot define a valid action mixture."""
        with self.assertRaises(ValueError):
            moe.MoEActor(3, 4, 2, 2, top_k=0)


class TemporalTests(unittest.TestCase):
    """Protect causal attention and the two-clock temporal-buffer contract."""

    def test_attention_cannot_read_future_tokens(self):
        """Changing future tokens must leave earlier causal outputs unchanged."""
        torch.manual_seed(11)
        block = temporal.PreLNTransformerBlock(16, 4, 32).eval()
        tokens = torch.randn(2, 8, 16)
        mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
        changed = tokens.clone()
        changed[:, 4:] += 10
        torch.testing.assert_close(block(tokens, mask)[:, :4], block(changed, mask)[:, :4])

    def test_visual_cache_and_history_have_separate_clocks(self):
        """A control-only refresh changes the live token without advancing old tokens."""
        cfg = {
            "num_prop": 3,
            "prop_window": 5,
            "history_len": 8,
            "d_model": 16,
            "depth_latent_dim": 4,
            "prop_latent_dim": 8,
            "context_dim": 16,
            "n_heads": 4,
            "d_ff": 32,
            "dropout": 0,
            "n_layers": 2,
            "num_scan_latent": 4,
            "num_dyn_targets": 5,
            "num_priv_explicit": 2,
        }
        encoder = temporal.TemporalContextEncoder(cfg).eval()
        with torch.no_grad():
            encoder(torch.randn(2, 58, 87), torch.randn(2, 3))
            old_history = encoder.token_buffer[:, :-1].clone()
            cached_visual = encoder.depth_latent_cache.clone()
            encoder(torch.randn(2, 58, 87), torch.randn(2, 3), is_perception_step=False)
            torch.testing.assert_close(encoder.token_buffer[:, :-1], old_history)
            torch.testing.assert_close(encoder.depth_latent_cache, cached_visual)
            encoder.reset(torch.tensor([True, False]))
            self.assertEqual(encoder.token_buffer[0].count_nonzero().item(), 0)
            self.assertGreater(encoder.token_buffer[1].count_nonzero().item(), 0)


class PpoTests(unittest.TestCase):
    """Verify trajectory boundaries and privileged critic initialization."""

    def test_returns_do_not_cross_episode_boundaries(self):
        """A reset must separate the completed segment from the next episode."""
        rewards = torch.tensor([[1.0], [2.0], [100.0]])
        values = torch.zeros_like(rewards)
        dones = torch.tensor([[0.0], [1.0], [0.0]])
        advantages, returns = ppo.generalized_advantage_estimate(
            rewards, values, dones, torch.tensor([10.0]), gamma=0.5, lam=1.0
        )
        torch.testing.assert_close(advantages, torch.tensor([[2.0], [2.0], [105.0]]))
        torch.testing.assert_close(returns, advantages)

    def test_critic_warm_start_preserves_teacher_values(self):
        """Matching critic layers should transfer the complete value function."""
        teacher = ppo.Critic(6, hidden=(8, 4))
        critic = ppo.Critic(6, hidden=(8, 4))
        proxy = type("Teacher", (), {"critic": teacher.net})()
        self.assertTrue(critic.load_from_teacher(proxy))
        inputs = torch.randn(4, 6)
        torch.testing.assert_close(teacher(inputs), critic(inputs))

    def test_spike_guard_recovers_after_bounded_skips(self):
        """A lasting loss shift must eventually be accepted instead of freezing training."""
        guard = optim.SpikeGuard(window=100, min_history=3, max_consecutive=2)
        for _ in range(3):
            self.assertFalse(guard.should_skip(1.0))
        self.assertTrue(guard.should_skip(20.0))
        self.assertTrue(guard.should_skip(20.0))
        self.assertFalse(guard.should_skip(20.0))
        self.assertTrue(guard.should_skip(float("nan")))


@unittest.skipUnless(RELEASED_CHECKPOINT.is_file(), "Released checkpoint is not available")
class ReleasedPolicyTests(unittest.TestCase):
    """Compare published weights against outputs captured before the cleanup."""

    def test_released_policy_matches_pre_refactor_outputs(self):
        """Actions, context, and gates must survive perception steps and episode resets."""
        actual = released_rollout()
        fixture = Path(__file__).parent / "fixtures/released_policy_outputs.npz"
        with np.load(fixture) as expected:
            for name, values in actual.items():
                np.testing.assert_allclose(values, expected[name], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
