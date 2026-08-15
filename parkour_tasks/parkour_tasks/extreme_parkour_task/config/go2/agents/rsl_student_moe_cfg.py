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


@configclass
class UnitreeGo2ParkourStudentMoERiskPPORunnerCfg(UnitreeGo2ParkourStudentMoEPPORunnerCfg):
    """Failure-aware variant: same architecture, different objective.

    Everything that defines the comparison is inherited unchanged -- the frozen
    phase-1 teacher, the environment, 15 000 iterations, peak LR 1e-3 with the
    same warmup and cosine, the Transformer context and the top-2 mixture. The
    only difference is *which timesteps the action loss cares about*.

    The measurement that motivates it, on the 12.5k checkpoint of the first run:

      pace              31.2 vs the teacher's 32.6 goals per env-minute  (-4%)
      falls             0.188 vs 0.073 per 1000 steps                    (2.6x)
      fall foresight    a linear readout of c_t sees a fall coming one second
                        ahead at AUC 0.73, control 0.54
      convergence       terrain_levels flat since iteration 11 000 while the
                        action loss kept falling 3.4% per 1000 iterations

    Read together: the student is not slow and it is not blind, it fails rarely
    and expensively, the failures are foreseeable from the vector the actor
    already consumes, and the plateau says more gradient on the same objective
    buys nothing. So the objective is what changes.
    """

    experiment_name = "unitree_go2_parkour_moe_risk"
    run_name = "moe_transformer_risk"

    depth_encoder = ParkourRslRlTemporalMoECfg(
        learning_rate=1e-3,
        num_steps_per_env=24 * 5,
        use_risk_head=True,
        coef_risk=1.0,
        risk_weight=4.0,
        risk_weight_warmup=2000,
        # Drop com_z and friction from the dynamics targets. Probed on the first
        # run's context with an environment-wise split, they are not recoverable
        # at all -- linear R^2 0.07 and -0.25, and a head of the same shape as
        # the real one does worse still at -0.24 and -0.49, i.e. it overfits
        # noise. Friction is the clear case: nothing in a depth image or in a
        # gait that has not yet slipped carries it.
        #
        # The three that stay are real: mass 0.79, com_y 0.92, com_x 0.38 held
        # out, and projecting those directions out of the context at inference
        # slowed the policy by 2.2 sigma, so the actor does use them.
        #
        # Note that the training log's dyn_loss of 0.26 is an in-sample number:
        # mass and COM are constant per environment, so once c_t identifies the
        # environment the head can memorise 192 values. The held-out probe is
        # the honest measure, and it is what these indices follow.
        dyn_dims=[0, 1, 2],
        # Cut the auxiliary budget. The head-to-head is what motivates this: the
        # baseline has two loss terms, spends everything on imitation and reaches
        # an action loss of 0.690, while this student spends 36% of its objective
        # there and sits at 1.316 on the same quantity and the same target -- and
        # loses by 2.8 sigma on pace and 3.4 sigma on falls.
        #
        # The auxiliary heads exist because c_t is a unified context that nothing
        # else would ground. The baseline shows that grounding is available for
        # free: its depth latent occupies the slot the teacher fills with its
        # scandots encoding, so the action loss shapes it implicitly, and it ends
        # closer to the teacher than our explicitly supervised context does.
        #
        # Cut rather than removed. Zeroing the context costs 12.2 goals per
        # env-minute and quadruples the falls, so c_t is load-bearing and worth
        # keeping supervised -- just not at a third of the objective. At these
        # weights the action share goes from 36% to about 55%.
        coef_dyn=1.0,
        coef_terrain=1.0,
    )


@configclass
class UnitreeGo2ParkourStudentMoEWarmPPORunnerCfg(UnitreeGo2ParkourStudentMoERiskPPORunnerCfg):
    """The failure-aware objective, plus a warm-started actor.

    The fallback if cutting the auxiliary budget does not close the gap. The
    head-to-head says the baseline wins by 2.8 sigma on pace and 3.4 on falls,
    and the training curves say it gets there roughly twice as fast, reaching a
    given terrain level at half the iterations. The one structural advantage it
    has is that its student actor *is* the teacher's actor, copied; it begins
    with a working controller and has only to learn the perception feeding it.

    A mixture cannot copy wholesale -- its experts read a different input -- but
    the layer shapes line up from the second layer on, so the mapping from
    internal features to joint targets transfers exactly and only the first layer
    has to learn. Experts keep their independent random first layers, so they do
    not start identical.
    """

    experiment_name = "unitree_go2_parkour_moe_warm"
    run_name = "moe_transformer_warm"

    depth_encoder = ParkourRslRlTemporalMoECfg(
        learning_rate=1e-3,
        num_steps_per_env=24 * 5,
        use_risk_head=True,
        coef_risk=1.0,
        risk_weight=4.0,
        risk_weight_warmup=2000,
        dyn_dims=[0, 1, 2],
        coef_dyn=1.0,
        coef_terrain=1.0,
        warm_start_experts=True,
    )


@configclass
class UnitreeGo2ParkourStudentMoEDensePPORunnerCfg(UnitreeGo2ParkourStudentMoEPPORunnerCfg):
    """The mixture trained soft from the start, rather than switched at the end.

    Two measurements motivate dropping top-k. Run on the distilled weights at
    inference, soft routing over all four experts is worth +2.77 +- 0.67 goals per
    environment-minute -- four times the sweep's resolution floor -- at a fall cost
    of +0.070 +- 0.042, which sits on the floor. And in this implementation it is
    free: FusedExperts already evaluates every expert on every step and then
    weights them, so top-2 was zeroing two contributions it had already paid for.
    Sparsity was buying no compute here, only a constraint.

    Weights trained under top-2 still carry its imprint -- experts 1 and 3 ended
    at mean gate weight 0.09 and 0.14 precisely because they rarely made the cut --
    so the honest version of the experiment trains dense from the start instead of
    flipping a switch over a model shaped by a different regime.

    CTS-MoE (arXiv 2606.19633) makes the same choice for perceptive locomotion,
    reasoning that locomotion tasks share enough structure that combining experts
    beats selecting one.

    Note that the load-balancing loss changes form under dense routing: the Switch
    formulation has identically zero gradient there. See MoEActor.
    """

    experiment_name = "unitree_go2_parkour_moe_dense"
    run_name = "moe_transformer_dense"

    depth_encoder = ParkourRslRlTemporalMoECfg(
        learning_rate=1e-3,
        num_steps_per_env=24 * 5,
        # top_k == num_experts is soft routing
        top_k=4,
        # Everything else stays at the first run's settings, so this differs from
        # the published comparison in routing alone. The auxiliary budget is NOT
        # cut here: that change was tested on its own and lost by 2.2 sigma on
        # both axes at iteration 4000.
    )


@configclass
class UnitreeGo2ParkourStudentMoEDenseWarmPPORunnerCfg(UnitreeGo2ParkourStudentMoEDensePPORunnerCfg):
    """Dense routing plus the warm-started actor.

    Selected over the plain dense variant only if the warm start proves itself at
    iteration 4000 against the original run. Keeping the two as separate task ids
    rather than a flag rewritten by a script means the decision is recorded in
    what was launched, not in a file that has since changed.
    """

    experiment_name = "unitree_go2_parkour_moe_dense_warm"
    run_name = "moe_transformer_dense_warm"

    depth_encoder = ParkourRslRlTemporalMoECfg(
        learning_rate=1e-3,
        num_steps_per_env=24 * 5,
        top_k=4,
        warm_start_experts=True,
    )
