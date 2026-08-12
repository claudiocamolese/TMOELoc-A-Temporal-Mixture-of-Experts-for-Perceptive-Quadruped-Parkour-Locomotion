from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
ParkourRslRlOnPolicyRunnerCfg,
ParkourRslRlPpoActorCriticCfg,
ParkourRslRlActorCfg,
ParkourRslRlStateHistEncoderCfg,
ParkourRslRlEstimatorCfg,
ParkourRslRlDistillationMoEAlgorithmCfg,
ParkourRslRlTemporalMoECfg
)
from isaaclab.utils import configclass


@configclass
class UnitreeGo2ParkourStudentMoEPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    """Temporal Mixture-of-Experts student.

    `policy` still describes the *teacher* (it is loaded from the phase-1
    checkpoint and frozen); the student lives entirely in `depth_encoder`.
    """

    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 100
    experiment_name = "unitree_go2_parkour_moe"
    run_name = "moe_transformer"
    empirical_normalization = False
    runner_class_name = "OnPolicyRunnerMoE"
    # The action term records the *raw* action in its history buffer before
    # clamping it (parkour_actions/joint_actions.py:42), and that history is fed
    # straight back as proprioception[37:49]. An unbounded student action
    # therefore re-enters its own observation, saturates the +-100 observation
    # clip and drives a runaway feedback loop -- measured: |obs| tracked
    # |action_student| exactly, at 41.1, 33.7, 62.0 on consecutive iterations.
    # Clipping at the wrapper makes the recorded action the executed one.
    clip_actions = 4.8
    logger = "wandb"
    wandb_project = "parkour-moe-transformer"

    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
        activation="elu",
        actor=ParkourRslRlActorCfg(
            class_name="Actor",
            state_history_encoder=ParkourRslRlStateHistEncoderCfg(
                class_name="StateHistoryEncoder"
            )
        )
    )
    estimator = ParkourRslRlEstimatorCfg(
        hidden_dims=[128, 64]
    )
    depth_encoder = ParkourRslRlTemporalMoECfg(
        learning_rate=1e-3,
        num_steps_per_env=24 * 5,
    )
    algorithm = ParkourRslRlDistillationMoEAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
