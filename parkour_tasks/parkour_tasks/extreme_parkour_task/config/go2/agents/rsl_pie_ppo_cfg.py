from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlOnPolicyRunnerCfg,
    ParkourRslRlPpoAlgorithmCfg,
)


@configclass
class ParkourRslRlPIEActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "SimpleActorCritic"
    init_noise_std: float = 1.0
    num_actor_obs: int = 118
    action_limit: float | None = 1.2
    actor_hidden_dims: list[int] = [256, 256, 128]
    critic_hidden_dims: list[int] = [256, 256, 128]
    activation: str = "elu"


@configclass
class ParkourRslRlPIELowNoiseActorCriticCfg(ParkourRslRlPIEActorCriticCfg):
    """Low initial exploration noise ablation for early-contact failures."""

    init_noise_std: float = 0.5


@configclass
class ParkourRslRlPIELowerNoiseActorCriticCfg(ParkourRslRlPIEActorCriticCfg):
    """Lower initial exploration noise while preserving the paper reward signal."""

    init_noise_std: float = 0.3


@configclass
class ParkourRslRlPIELimitedActionActorCriticCfg(ParkourRslRlPIEActorCriticCfg):
    """Bound early joint offsets while keeping enough policy exploration."""

    init_noise_std: float = 0.5
    action_limit: float | None = 0.6


@configclass
class ParkourRslRlPIEGentleActorCriticCfg(ParkourRslRlPIEActorCriticCfg):
    """Moderate early exploration for basic walking bootstrapping."""

    init_noise_std: float = 0.30
    action_limit: float | None = 0.8


@configclass
class ParkourRslRlPIEGentleLoadFixActorCriticCfg(ParkourRslRlPIEGentleActorCriticCfg):
    """GentleLoadFix actor with full PIE action range for long walking runs."""

    action_limit: float | None = 1.2


@configclass
class ParkourRslRlPIEFullParkourActorCriticCfg(ParkourRslRlPIEActorCriticCfg):
    """From-scratch FullParkour actor: Teacher-level exploration with PIE arch.

    The Gentle/GentleLoadFix variants lower ``init_noise_std`` to 0.30 because
    they fine-tune from an already-walking checkpoint. Training PIE from
    random init on the multi-terrain Teacher mix needs much more exploration,
    matching Teacher's 1.0. Hidden dims widened to ``[512, 256, 128]`` so the
    first MLP layer can absorb the asymmetric critic privileged input (220
    dims) and the actor's heterogeneous proprio + estimator-latent stream.
    """

    init_noise_std: float = 1.0
    action_limit: float | None = 1.2
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]


@configclass
class ParkourRslRlPIEBridgeActorCriticCfg(ParkourRslRlPIEActorCriticCfg):
    """Moderate exploration and action range after a stable Gentle warmup."""

    init_noise_std: float = 0.25
    action_limit: float | None = 0.8


@configclass
class ParkourRslRlPIEEstimatorCfg:
    class_name: str = "PIEEstimator"
    learning_rate: float = 1.0e-4
    proprio_dim: int = 47
    next_proprio_dim: int = 47
    sample_latent_in_training: bool = False
    train_with_estimated_states: bool = False
    use_pie_estimator_rollout: bool = True
    use_pie_actor_features: bool = True
    detach_pie_actor_features: bool = True
    pie_joint_actor_estimator: bool = False
    pie_policy_obs_dim: int = 47
    pie_actor_estimator_grad_scale: float = 1.0
    pie_actor_feature_clip: float | None = 5.0
    pie_actor_feature_keys: tuple[str, ...] = ("z_m", "z_mu", "v_hat", "h_f_hat")
    pie_train_gru_sequence: bool = True
    pie_num_learning_epochs: int = 1
    pie_num_mini_batches: int = 4
    loss_weights: dict[str, float] = {
        "v": 1.0,
        "h_f": 1.0,
        "height": 1.0,
        "next_proprio": 1.0,
        "kl": 1.0,
        # Terrain-adaptive emphasis for height/foot-clearance losses. 0.0 keeps
        # the original uniform MSE. Raise (e.g. 2.0) to focus estimator
        # gradient on rough terrain (steps/slopes/gaps), where audits showed
        # height/h_f error is 5-12x worse than on flat ground.
        "terrain_adaptive": 0.0,
    }


@configclass
class UnitreeGo2PIEParkourPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 2000
    clip_actions = 1.2
    experiment_name = "unitree_go2_pie_parkour"
    empirical_normalization = False

    policy = ParkourRslRlPIEActorCriticCfg()
    estimator = ParkourRslRlPIEEstimatorCfg()
    depth_encoder = None

    algorithm = ParkourRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        desired_kl=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        dagger_update_freq=1,
        priv_reg_coef_schedual=[0.0, 0.0, 0.0, 1.0],
    )


@configclass
class UnitreeGo2PIELowNoisePPORunnerCfg(UnitreeGo2PIEParkourPPORunnerCfg):
    """PIE runner ablation that only lowers actor initial action noise."""

    policy = ParkourRslRlPIELowNoiseActorCriticCfg()


@configclass
class UnitreeGo2PIELowerNoisePPORunnerCfg(UnitreeGo2PIEParkourPPORunnerCfg):
    """PIE runner ablation with lower actor initial action noise."""

    policy = ParkourRslRlPIELowerNoiseActorCriticCfg()


@configclass
class UnitreeGo2PIELimitedActionPPORunnerCfg(UnitreeGo2PIEParkourPPORunnerCfg):
    """PIE runner ablation with bounded sampled actions and moderate exploration."""

    clip_actions = 0.6
    policy = ParkourRslRlPIELimitedActionActorCriticCfg()


@configclass
class UnitreeGo2PIEGentlePPORunnerCfg(UnitreeGo2PIEParkourPPORunnerCfg):
    """PIE runner ablation for stable walking warmup with bounded actions."""

    save_interval = 500
    clip_actions = 0.8
    policy = ParkourRslRlPIEGentleActorCriticCfg()
    algorithm = ParkourRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        desired_kl=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        dagger_update_freq=1,
        priv_reg_coef_schedual=[0.0, 0.0, 0.0, 1.0],
    )


@configclass
class UnitreeGo2PIEGentleLoadFixPPORunnerCfg(UnitreeGo2PIEGentlePPORunnerCfg):
    """Gentle runner with full action range for long walking runs."""

    save_interval = 2000
    clip_actions = 1.2
    policy = ParkourRslRlPIEGentleLoadFixActorCriticCfg()


@configclass
class UnitreeGo2PIEFullParkourPPORunnerCfg(UnitreeGo2PIEParkourPPORunnerCfg):
    """From-scratch PIE training on the Teacher-style multi-terrain mix.

    Differences vs ``UnitreeGo2PIEGentleLoadFixPPORunnerCfg``:
    - ``init_noise_std=1.0`` and ``entropy_coef=0.01`` (Teacher levels) so the
      randomly-initialised actor explores enough to discover walking.
    - Inherits ``UnitreeGo2PIEParkourPPORunnerCfg`` (entropy_coef already 0.01)
      directly, so the Gentle warmup-tuned values do not leak in.
    - Keeps ``dagger_update_freq=1`` and the PIE-style
      ``priv_reg_coef_schedual=[0,0,0,1.0]`` because those are dictated by
      the PIE estimator architecture, not the training stage.
    - ``save_interval=2000`` matches what we have been using on 4090 server
      for long training runs.
    """

    save_interval = 2000
    clip_actions = 1.2
    policy = ParkourRslRlPIEFullParkourActorCriticCfg()


@configclass
class ParkourRslRlPIEFlatStage1ActorCriticCfg(ParkourRslRlPIEFullParkourActorCriticCfg):
    """Stage-1 actor with tighter action bounds.

    Cuts ``action_limit`` from 1.2 to 0.8 so the actor cannot park its
    output at the saturation rail and lean on PD overshoot. This was the
    root cause of the FlatStage1 v1/v2 failures: with action_limit=1.2 the
    target calf angle could be set 0.6 rad past the soft joint limit, which
    let the policy hide a degenerate "drag-and-shake" gait behind a high
    Train/mean_reward. Stage 2 (FullParkour) restores 1.2 to give the
    policy enough range for hurdles / gap jumps; until then 0.8 forces a
    proper trot during walking bootstrap.

    All other settings (init_noise_std=1.0, [512,256,128] hidden dims) are
    inherited unchanged from FullParkour.
    """

    action_limit: float | None = 0.8


@configclass
class UnitreeGo2PIEFlatStage1PPORunnerCfg(UnitreeGo2PIEFullParkourPPORunnerCfg):
    """Stage-1 PPO runner that uses the tighter-action ActorCritic above."""

    save_interval = 500
    clip_actions = 0.8
    policy = ParkourRslRlPIEFlatStage1ActorCriticCfg()


@configclass
class ParkourRslRlPIEFullStage2WarmActorCriticCfg(ParkourRslRlPIEFullParkourActorCriticCfg):
    """Stage-2 warm-start actor: keep Stage 1's action_limit=0.8.

    Earlier we tried 1.0 here to leave room for hurdles / gaps, but a
    Stage 1 ckpt loaded into action_limit=1.0 has every action
    instantly scaled 25% larger, which the policy was never trained for
    and the robot cannot stand within two steps. Stage 2a now keeps the
    same 0.8 as Stage 1 so the policy only has to adapt to new terrain
    + heading range, not action scaling. Stage 2b later switches to the
    original FullParkour cfg (action_limit=1.2) once Stage 2a stabilises.
    """

    action_limit: float | None = 0.8


@configclass
class UnitreeGo2PIEFullStage2WarmPPORunnerCfg(UnitreeGo2PIEFullParkourPPORunnerCfg):
    """Stage-2 warm-up runner: clip_actions=0.8, save_interval=1000."""

    save_interval = 1000
    clip_actions = 0.8
    policy = ParkourRslRlPIEFullStage2WarmActorCriticCfg()


@configclass
class ParkourRslRlPIETerrainAdaptiveEstimatorCfg(ParkourRslRlPIEEstimatorCfg):
    """Estimator cfg that turns on terrain-adaptive height/h_f loss weighting.

    Audits showed the estimator's height and foot-clearance error on rough
    sub-terrains (step / slope / gap) is 5-12x worse than on flat ground,
    because flat samples dominate the uniform-MSE loss. Setting
    ``terrain_adaptive=2.0`` reweights each sample by its height-scan spatial
    roughness (batch-mean-normalised to 1) so the estimator is pushed to
    estimate accurately exactly where parkour traversal needs it. This is a
    loss-only change: the network architecture is unchanged, so a checkpoint
    can resume into this runner.
    """

    loss_weights: dict[str, float] = {
        "v": 1.0,
        "h_f": 1.0,
        "height": 1.0,
        "next_proprio": 1.0,
        "kl": 1.0,
        "terrain_adaptive": 2.0,
    }


@configclass
class UnitreeGo2PIEFullStage2WarmTerrainAdaptivePPORunnerCfg(UnitreeGo2PIEFullStage2WarmPPORunnerCfg):
    """Stage-2 warm-up runner with terrain-adaptive estimator loss weighting.

    Identical to ``UnitreeGo2PIEFullStage2WarmPPORunnerCfg`` (same actor,
    clip_actions, save_interval) except the estimator uses
    ``terrain_adaptive=2.0`` to emphasise rough-terrain perception accuracy.
    Network shapes are unchanged, so this can resume from an existing
    FrontFast / Stage2Warm checkpoint.
    """

    estimator = ParkourRslRlPIETerrainAdaptiveEstimatorCfg()


# ---------------------------------------------------------------------------
# High-capacity perception variant (from-scratch experiment "Strategy B").
#
# Audits showed PIE height/foot-clearance error is 5-12x worse on rough
# sub-terrains (step/slope/gap) than on flat ground. Strategy B combines
# four changes aimed squarely at perception accuracy, all of which preserve
# the sim2real interface (depth input 2x58x87, proprio 47, action 12):
#   (1) terrain_adaptive=2.0   - focus loss gradient on rough terrain
#   (2) z_m_dim 32 -> 64 + height_decoder [128] -> [256,128]
#                              - more capacity for the 132-dim height map
#   (3) depth_feature_map 6x9 -> 8x12 (54 -> 96 visual tokens)
#                              - preserve terrain edge detail through the CNN
#   (6) h_f loss weight 1.0 -> 2.0  - push foot-clearance accuracy
#
# Because z_m grows, the actor input grows: 47 + 64 (z_m) + 32 (z_mu) +
# 3 (v) + 4 (h_f) = 150 (was 118). The matching actor cfg sets
# num_actor_obs=150. This is a from-scratch architecture (cannot resume old
# checkpoints).
# ---------------------------------------------------------------------------
@configclass
class ParkourRslRlPIEHighCapEstimatorCfg(ParkourRslRlPIEEstimatorCfg):
    """High-capacity PIE estimator: bigger terrain latent + finer depth map."""

    z_m_dim: int = 64
    depth_feature_map_shape: tuple[int, int] = (8, 12)
    height_decoder_hidden_dims: tuple[int, ...] = (256, 128)
    loss_weights: dict[str, float] = {
        "v": 1.0,
        "h_f": 2.0,
        "height": 1.0,
        "next_proprio": 1.0,
        "kl": 1.0,
        "terrain_adaptive": 2.0,
    }


@configclass
class ParkourRslRlPIEHighCapActorCriticCfg(ParkourRslRlPIEFullStage2WarmActorCriticCfg):
    """Actor for the high-capacity estimator: z_m=64 widens actor input to 150.

    actor input = proprio(47) + z_m(64) + z_mu(32) + v_hat(3) + h_f(4) = 150.
    Keeps Stage-2 warm action_limit=0.8 and Teacher-level init_noise_std=1.0
    for from-scratch exploration.
    """

    num_actor_obs: int = 150


@configclass
class UnitreeGo2PIEFullParkourHighCapPPORunnerCfg(UnitreeGo2PIEFullStage2WarmPPORunnerCfg):
    """From-scratch high-capacity perception runner (Strategy B).

    Pairs the high-capacity estimator (z_m=64, depth 8x12, wider height
    decoder, terrain_adaptive=2.0, h_f weight 2.0) with the matching
    num_actor_obs=150 actor. Trained from random init on the FrontFast
    curriculum. NOT checkpoint-compatible with prior PIE runs.

    save_interval=250 so a short 500-iteration smoke run produces
    model_250 / model_500 checkpoints to inspect early before committing to a
    long run.
    """

    save_interval = 250
    estimator = ParkourRslRlPIEHighCapEstimatorCfg()
    policy = ParkourRslRlPIEHighCapActorCriticCfg()


# ---------------------------------------------------------------------------
# Stage 0: flat-ground walking warmup for the high-capacity architecture.
#
# Audits + history show PIE from-scratch on the full obstacle mix fails to
# learn to even stand/walk (episode_length collapses). So we bootstrap the
# HighCap network on pure flat ground first. Crucially we DISABLE the h_f and
# height estimator losses during this stage (weights 0): on flat ground the
# foot-clearance / height targets are trivially predictable from proprio, so
# training them here would teach the estimator a proprio shortcut that ignores
# depth (exactly the failure the audit found: depth_shuffle->h_f ~ 0). By not
# training them at all on flat ground, the heads start from a clean slate when
# h_f/height (+terrain_adaptive) are turned back on in the obstacle stage.
#
# v_hat and next_proprio losses stay on (velocity tracking + delay robustness
# are meaningful even on flat ground). Network shapes are identical to the
# HighCap runner, so the obstacle stage can resume from this checkpoint.
# ---------------------------------------------------------------------------
@configclass
class ParkourRslRlPIEHighCapFlatWarmupEstimatorCfg(ParkourRslRlPIEHighCapEstimatorCfg):
    """HighCap estimator with terrain losses (h_f, height) disabled for the
    flat walking warmup stage."""

    loss_weights: dict[str, float] = {
        "v": 1.0,
        "h_f": 0.0,
        "height": 0.0,
        "next_proprio": 1.0,
        "kl": 1.0,
        "terrain_adaptive": 0.0,
    }


@configclass
class UnitreeGo2PIEHighCapFlatWarmupPPORunnerCfg(UnitreeGo2PIEFullParkourHighCapPPORunnerCfg):
    """Stage-0 flat walking warmup runner for the HighCap architecture.

    Same network as the HighCap runner (so the obstacle stage can resume from
    its checkpoint), but the estimator's h_f/height losses are off so flat
    ground does not teach a depth-ignoring proprio shortcut.
    """

    estimator = ParkourRslRlPIEHighCapFlatWarmupEstimatorCfg()


@configclass
class UnitreeGo2PIEBridgePPORunnerCfg(UnitreeGo2PIEParkourPPORunnerCfg):
    """Bridge runner that relaxes Gentle constraints before full PIE training."""

    save_interval = 500
    clip_actions = 0.8
    policy = ParkourRslRlPIEBridgeActorCriticCfg()
    algorithm = ParkourRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        desired_kl=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        dagger_update_freq=1,
        priv_reg_coef_schedual=[0.0, 0.0, 0.0, 1.0],
    )


@configclass
class UnitreeGo2PIEBridgeLoadFixPPORunnerCfg(UnitreeGo2PIEBridgePPORunnerCfg):
    """Bridge runner with denser checkpointing for load and propulsion diagnostics."""

    save_interval = 250
