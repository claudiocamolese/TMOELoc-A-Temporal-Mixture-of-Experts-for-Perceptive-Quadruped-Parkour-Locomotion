"""CPU test loading that avoids the inherited simulator-importing package init."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "scripts/rsl_rl/modules"
RELEASED_CHECKPOINT = ROOT / (
    "logs/rsl_rl/unitree_go2_parkour_moe/2026-08-12_12-16-19_moe_transformer/slim/model_14999.pt"
)


def load_core(name: str, root: Path = ROOT, namespace: str = "_expertparkour_cpu"):
    """Import our core modules without executing upstream Isaac/RSL-RL imports.

    Namespace package paths point at the real source files; no module under
    test is mocked. The upstream depth CNN is loaded from the working tree when
    comparing an archived copy containing only ExpertParkour-owned files.
    """
    for suffix in ("", ".feature_extractors"):
        package_name = namespace + suffix
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            relative = "scripts/rsl_rl/modules" + suffix.replace(".", "/")
            package.__path__ = [str(root / relative), str(ROOT / relative)]
            sys.modules[package_name] = package
    return importlib.import_module(f"{namespace}.{name}")


def load_tool(relative_path: str):
    """Load an offline script directly, bypassing the inherited scripts package init."""
    path = ROOT / "scripts" / relative_path
    name = "_expertparkour_tool_" + relative_path.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def released_rollout(root: Path = ROOT, namespace: str = "_expertparkour_cpu"):
    """Run deterministic recorded-weight inputs across perception and reset steps."""
    encoder_class = load_core(
        "feature_extractors.temporal_transformer", root, namespace
    ).TemporalContextEncoder
    actor_class = load_core("moe_actor", root, namespace).MoEActor
    policy_class = load_core("student_inference", root, namespace).StudentPolicy
    depth_cfg = {
        "num_prop": 53,
        "prop_window": 5,
        "history_len": 8,
        "d_model": 128,
        "depth_latent_dim": 32,
        "prop_latent_dim": 64,
        "context_dim": 128,
        "n_heads": 4,
        "d_ff": 256,
        "dropout": 0.0,
        "n_layers": 2,
        "num_scan_latent": 32,
        "num_dyn_targets": 5,
        "num_priv_explicit": 9,
    }
    encoder = encoder_class(depth_cfg)
    actor = actor_class(53, 128, 9, 12)
    checkpoint = torch.load(RELEASED_CHECKPOINT, map_location="cpu", weights_only=False)
    encoder.load_state_dict(checkpoint["depth_encoder_state_dict"], strict=True)
    actor.load_state_dict(checkpoint["depth_actor_state_dict"], strict=True)
    runner = SimpleNamespace(alg=SimpleNamespace(depth_encoder=encoder, depth_actor=actor))
    config = {
        "algorithm": {"class_name": "DistillationMoE"},
        "depth_encoder": {"zero_env_class": True, "vel_source": "true"},
        "estimator": {"num_prop": 53, "num_scan": 132, "num_priv_explicit": 9},
    }
    policy = policy_class(runner, config, "cpu")
    rng = np.random.default_rng(2026)
    recorded = {"actions": [], "context": [], "gates": []}
    for step in range(8):
        observation = torch.from_numpy(rng.normal(0, 0.1, (2, 223)).astype(np.float32))
        depth = torch.from_numpy(rng.uniform(-0.5, 0.5, (2, 58, 87)).astype(np.float32))
        actions = policy.act(observation, depth, is_perception_step=step % 5 == 0)
        recorded["actions"].append(actions.numpy().copy())
        recorded["context"].append(policy.last_context.numpy().copy())
        recorded["gates"].append(policy.last_gates.numpy().copy())
        # Reset different environments between camera frames to exercise cached
        # context, uniform gates after reset, and the next perception refresh.
        policy.reset(torch.tensor([step == 3, step == 4]))
    return {key: np.asarray(value) for key, value in recorded.items()}
