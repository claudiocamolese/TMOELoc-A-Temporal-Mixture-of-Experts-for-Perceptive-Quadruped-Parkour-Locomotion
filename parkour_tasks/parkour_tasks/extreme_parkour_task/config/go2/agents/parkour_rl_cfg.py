# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
RslRlOnPolicyRunnerCfg, 
RslRlPpoActorCriticCfg, 
RslRlPpoAlgorithmCfg,
)
#########################
# Policy configurations #
#########################

@configclass
class ParkourRslRlBaseCfg:
    num_priv_explicit: int = 3 + 3 + 3 # 9
    num_priv_latent: int = 4 + 1 + 12 +12 # 29
    num_prop: int = 3 + 2 + 3 + 4 + 36 + 5 # 53
    num_scan: int = 132
    num_hist: int = 10
    
@configclass
class ParkourRslRlStateHistEncoderCfg(ParkourRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder" 
    channel_size: int = 10 
    
@configclass
class ParkourRslRlDepthEncoderCfg(ParkourRslRlBaseCfg):
    backbone_class_name: str = "DepthOnlyFCBackbone58x87" 
    encoder_class_name: str = "RecurrentDepthBackbone" 
    depth_shape: tuple[int] = (87, 58)
    hidden_dims: int = 512
    learning_rate: float = 5.e-4
    num_steps_per_env: int = 24 * 5
    # mask the terrain-class one-hot out of the student input (see MoE cfg)
    zero_env_class: bool = True
    # identical schedule to the MoE student: the two arms must differ only in
    # architecture, not in optimiser
    warmup_iters: int = 500
    total_iters: int = 15000
    min_lr_ratio: float = 0.1
    spike_factor: float = 5.0
    # Clamp the teacher action *target* to the range the environment applies
    # (ActionsCfg.joint_pos.clip). Two actions that differ only beyond that clip
    # drive identical torques, so the difference is not a behavioural error and
    # regressing it is fitting noise.
    #
    # This is what killed two runs. When the robot enters a violent state the
    # observation saturates at the +-100 observation clip; the teacher, a dense
    # MLP that never saw saturated inputs, then emits actions of magnitude ~700.
    # Measured at the divergence: |a|teacher 30 -> 415 -> 700 while the context
    # norm stayed flat at 5.4, so the temporal encoder was never the problem.
    # Upstream never hits this because its student is warm-started from the
    # teacher's own actor and therefore never leaves the well-behaved regime.
    action_target_clip: float = 4.8

@configclass
class ParkourRslRlTemporalMoECfg(ParkourRslRlBaseCfg):
    """Unified temporal Transformer + Mixture-of-Experts student.

    Replaces the GRU depth backbone and the Conv1D state history encoder with a
    single multimodal temporal Transformer, and the dense actor with a sparse
    top-k MoE.
    """

    depth_shape: tuple[int] = (87, 58)
    num_steps_per_env: int = 24 * 5
    # Peak LR, halved from the upstream 1e-3. That value is fine for a student
    # warm-started from the teacher's actor; this one trains from scratch, and
    # at 1e-3 without warmup a run diverged irrecoverably at iteration 1020.
    learning_rate: float = 5.e-4
    warmup_iters: int = 500
    total_iters: int = 15000
    min_lr_ratio: float = 0.1
    spike_factor: float = 5.0
    # see ParkourRslRlDepthEncoderCfg.action_target_clip
    action_target_clip: float = 4.8

    # -- token construction
    prop_window: int = 5          # 50 Hz proprioceptive samples per token (= 100 ms)
    depth_latent_dim: int = 32    # unchanged Extreme Parkour CNN output
    prop_latent_dim: int = 64

    # -- temporal transformer
    history_len: int = 8          # tokens at 10 Hz -> 0.8 s of context
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    context_dim: int = 128

    # -- mixture of experts
    num_experts: int = 4
    top_k: int = 2
    expert_hidden_dims: list[int] = [256, 128]
    router_hidden_dim: int = 128
    use_vel_in_input: bool = True
    vel_source: str = "true"      # "true" | "predicted"
    router_noise_std: float = 0.3  # noisy top-k gating, training only

    # -- ablation hooks (no cost when left at the defaults)
    zero_env_class: bool = True

    # -- loss weights
    #
    # Calibrated against a 235-iteration probe run. The action loss sits around
    # 6, so a coefficient only matters if its term reaches a comparable scale.
    coef_action: float = 1.0
    coef_yaw: float = 1.0
    # The context losses are mean-squared-per-element and landed near 0.05 /
    # 0.0004 / 0.014, i.e. ~1% of the objective at unit weight: effectively
    # unsupervised. Scaled up so that shaping c_t is an actual training goal.
    coef_terrain: float = 5.0
    coef_dyn: float = 5.0
    coef_vel: float = 5.0
    # The balance loss ranges over [2, 4] for K=4/top-2. The probe run pinned it
    # at 3.6-4.0 (worst case: all mass on a fixed pair) because at 0.01 it was
    # 0.6% of the objective and could only kick the router between collapsed
    # configurations.
    coef_balance: float = 0.15
    # Disabled: a constant gate is its global optimum, so it rewards exactly the
    # collapse the balance loss fights. Gate caching at the 10 Hz perception
    # rate already prevents mid-stride switching. Still logged.
    coef_smooth: float = 0.0
    coef_zloss: float = 1.e-3


@configclass
class ParkourRslRlEstimatorCfg(ParkourRslRlBaseCfg):
    class_name: str = "DefaultEstimator" 
    train_with_estimated_states: bool = True 
    learning_rate: float = 1.e-4 
    hidden_dims: list[int] = MISSING 
    
@configclass
class ParkourRslRlActorCfg(ParkourRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: ParkourRslRlStateHistEncoderCfg = MISSING


@configclass
class ParkourRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = 'ActorCriticRMA'
    tanh_encoder_output: bool = False 
    scan_encoder_dims: list[int] = MISSING
    priv_encoder_dims: list[int] = MISSING
    actor: ParkourRslRlActorCfg = MISSING

@configclass
class ParkourRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = 'PPOWithExtractor'
    dagger_update_freq: int = 1
    priv_reg_coef_schedual: list[float]= [0, 0.1, 2000, 3000]

@configclass
class ParkourRslRlDistillationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "DistillationWithExtractor"

@configclass
class ParkourRslRlDistillationMoEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "DistillationMoE"

@configclass
class ParkourRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: ParkourRslRlPpoActorCriticCfg = MISSING
    estimator: ParkourRslRlEstimatorCfg = MISSING
    depth_encoder: ParkourRslRlDepthEncoderCfg | ParkourRslRlTemporalMoECfg | None = None
    algorithm: (
        ParkourRslRlPpoAlgorithmCfg
        | ParkourRslRlDistillationAlgorithmCfg
        | ParkourRslRlDistillationMoEAlgorithmCfg
    ) = MISSING
    runner_class_name: str = "OnPolicyRunnerWithExtractor"

