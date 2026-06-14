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
    proprio_dim: int = 45
    next_proprio_dim: int = 45
    sample_latent_in_training: bool = False
    train_with_estimated_states: bool = False
    use_pie_estimator_rollout: bool = True
    use_pie_actor_features: bool = True
    detach_pie_actor_features: bool = True
    pie_joint_actor_estimator: bool = False
    pie_policy_obs_dim: int = 45
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
    # VAE implicit latent z_mu/z dim. Aligned to DreamWaQ's 16-dim context
    # latent (Nahrendra et al. 2023): audits showed z_mu was near-collapsed
    # (KL~0, |z_mu|~0.01) at 32 dims, i.e. half the capacity was unused and
    # diluted the actor input. 16 matches the reference and trims the actor
    # obs accordingly (z_m terrain code stays 64 -- that's the START heightmap
    # encoding, a separate design axis).
    latent_dim: int = 16
    depth_feature_map_shape: tuple[int, int] = (8, 12)
    height_decoder_hidden_dims: tuple[int, ...] = (256, 128)
    # h_f lowered 2.0 -> 1.0 and v raised 1.0 -> 1.5: the model_21750 audit
    # showed v_hat_rmse blew up to [0.33, 0.22, 0.84] (was ~0.10) while h_f/
    # height got accurate - i.e. the strong h_f weight (2.0) starved the shared
    # GRU/attention trunk of velocity-estimation capacity. v_hat is the
    # actor's most-relied-on feature, so its corruption stalled the policy
    # (terrain_level stuck ~2, episode_length collapsed). Rebalance toward v_hat.
    loss_weights: dict[str, float] = {
        "v": 1.5,
        "h_f": 1.0,
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
# HighCap Stage 2 with a hard noise-std ceiling.
#
# The Stage-2 collapse (2026-05-30_18-36-34) was diagnosed from the TB
# scalars: resuming the flat walker (mean_noise_std=0.033 at iter 3500) into
# PPO with entropy_coef=0.01, the action noise std climbed MONOTONICALLY every
# iteration (0.21 -> 0.30 -> 0.37 -> 0.41 -> 0.47 -> 0.54 -> 0.74). Episode
# length and terrain level peaked at iter 4500 (noise 0.41, terrain 2.0) and
# then collapsed exactly as noise crossed ~0.45. The estimator losses were
# never the cause - they kept improving throughout. Root cause: on noisy
# obstacle returns the surrogate gradient cannot pin the std down, so the
# entropy bonus inflates exploration without limit until the robot can no
# longer balance.
#
# Fix: cap the exploration noise std at 0.40 - right at the proven-productive
# peak (best terrain progress happened at 0.30-0.41) but a hard wall below the
# ~0.45 runaway tipping point. This is the single-variable change; PPO
# entropy_coef / KL schedule are left untouched. The cap defaults to None
# (legacy) everywhere else, so only this runner is affected and a checkpoint
# can resume into it (architecture unchanged).
# ---------------------------------------------------------------------------
@configclass
class ParkourRslRlPIEHighCapNoiseCapActorCriticCfg(ParkourRslRlPIEHighCapActorCriticCfg):
    """HighCap actor with a hard exploration-noise ceiling (anti-runaway)."""

    max_noise_std: float | None = 0.40


@configclass
class UnitreeGo2PIEFullParkourHighCapNoiseCapPPORunnerCfg(UnitreeGo2PIEFullParkourHighCapPPORunnerCfg):
    """HighCap Stage-2 runner that caps action noise std at 0.40.

    Identical to ``UnitreeGo2PIEFullParkourHighCapPPORunnerCfg`` (same
    estimator, save_interval, curriculum) except the actor clamps its
    exploration std to 0.40, breaking the entropy-driven noise runaway that
    collapsed the previous Stage-2 run around iter 4750-5000.
    """

    policy = ParkourRslRlPIEHighCapNoiseCapActorCriticCfg()


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


# ---------------------------------------------------------------------------
# START-style perception (arXiv 2409.15692 "Walking with Terrain
# Reconstruction", PIE same team). Two additions on top of HighCap:
#   (B) Two-stage heightmap reconstruction: z_m_head -> rough heightmap
#       (MLP, MSE) -> conv U-Net-lite refine (L1). The 132-dim height scan is
#       treated as a 12x11 grid (height_scanner GridPattern res=0.15
#       size=[1.65,1.5]). The refined map is what the actor's z_m encodes.
#   (A) AdaSmpl: during the estimator update, a fraction of samples encode the
#       GROUND-TRUTH heightmap into z_m instead of the reconstruction, with
#       probability p = min(tanh(CV(reward)), pie_adasmpl_max_prob). High early
#       reward variance -> more GT sampling; converged -> ~0 so deployment uses
#       pure reconstruction. The actor consumes z_m = heightmap_encoder(map),
#       so AdaSmpl genuinely changes what the policy sees (faithful to START).
#
# Because z_m now = encode(heightmap) rather than a free head, z_m semantics
# change -> NOT checkpoint-compatible with prior HighCap runs; train from the
# flat warmup. actor input is unchanged (z_m still 64) so num_actor_obs=150.
# height_rough/height_refined_l1 loss weights add the dual supervision; the
# base "height" MSE is kept small for stability.
# ---------------------------------------------------------------------------
@configclass
class ParkourRslRlPIESTARTEstimatorCfg(ParkourRslRlPIEHighCapEstimatorCfg):
    """HighCap estimator + START heightmap refine + heightmap-encoded z_m + AdaSmpl."""

    use_height_refine: bool = True
    use_heightmap_encoder: bool = True
    height_grid_shape: tuple[int, int] = (12, 11)
    height_refine_hidden_channels: int = 16
    pie_use_adasmpl: bool = True
    pie_adasmpl_max_prob: float = 0.8
    loss_weights: dict[str, float] = {
        "v": 1.0,
        "h_f": 1.0,
        "height": 0.0,            # START supervises the refined map with L1 ONLY
                                  # (eq 4); the extra refined-map MSE (was 0.5
                                  # "for stability") re-introduces the
                                  # over-smoothing START avoids and blurs the
                                  # terrain edges that sparse footholds
                                  # (beam/gap) depend on. Pure rough-MSE +
                                  # refined-L1 now == START eq 4.
        "next_proprio": 1.0,
        "kl": 1.0,
        "terrain_adaptive": 0.0,  # disabled: not in PIE/START (was a self-added
                                  # trick to upweight rough-terrain height/h_f
                                  # loss; height is now accurate everywhere and
                                  # it appeared to starve v_hat capacity -> vz
                                  # estimate degraded. Back to paper equal-weight.)
        "height_rough": 1.0,      # START rough-map MSE
        "height_refined_l1": 1.0,  # START refined-map L1
    }


@configclass
class ParkourRslRlPIESTARTFlatWarmupEstimatorCfg(ParkourRslRlPIESTARTEstimatorCfg):
    """START estimator for the flat warmup: terrain (h_f/height) losses OFF so
    flat ground does not teach a depth-ignoring proprio shortcut, but the
    refine/encoder modules still exist so the obstacle stage can resume."""

    loss_weights: dict[str, float] = {
        "v": 1.0,
        "h_f": 0.0,
        "height": 0.0,
        "next_proprio": 1.0,
        "kl": 1.0,
        "terrain_adaptive": 0.0,
        "height_rough": 0.0,
        "height_refined_l1": 0.0,
    }


@configclass
class UnitreeGo2PIESTARTFlatWarmupPPORunnerCfg(UnitreeGo2PIEHighCapFlatWarmupPPORunnerCfg):
    """Stage-0 flat warmup for the START architecture.

    Same network as the START Stage-2 runner (so the obstacle stage can resume
    from its checkpoint) but with terrain losses off and AdaSmpl effectively
    inert (no rough/height loss to benefit from GT on flat ground). NoiseCap
    actor (std<=0.40) carried for stable warmup exploration.
    """

    estimator = ParkourRslRlPIESTARTFlatWarmupEstimatorCfg()
    policy = ParkourRslRlPIEHighCapNoiseCapActorCriticCfg()


@configclass
class UnitreeGo2PIESTARTStage2PPORunnerCfg(UnitreeGo2PIEFullParkourHighCapNoiseCapPPORunnerCfg):
    """START Stage-2 obstacle runner: heightmap refine + heightmap-encoded z_m +
    AdaSmpl, on top of the NoiseCap (std<=0.40) actor. Resume from the START
    flat warmup checkpoint with --reset_optimizer_on_resume."""

    estimator = ParkourRslRlPIESTARTEstimatorCfg()
    policy = ParkourRslRlPIEHighCapNoiseCapActorCriticCfg()


# ---------------------------------------------------------------------------
# START per-foot heightmap (Ĥᶠ) variant: h_f_hat regresses a 4-leg x 3x3 = 36
# local heightmap instead of the 4-dim corridor demand (arXiv 2409.15692
# Appendix-C: "estimate the heightmap within 0.1m around each foot" beats a
# scalar clearance; Stepping Beams MEV 0.14 vs 0.52). Stacks on the full START
# estimator (refine + heightmap-encoded z_m + AdaSmpl). foot_height_dim=36
# widens the h_f head/target/storage and the actor input:
#   actor input = proprio(47) + z_m(64) + z_mu(32) + v_hat(3) + h_f(36) = 182.
# NOT checkpoint-compatible (h_f + actor dims changed); train from the START
# FootHmap flat warmup.
# ---------------------------------------------------------------------------
@configclass
class ParkourRslRlPIESTARTFootHmapEstimatorCfg(ParkourRslRlPIESTARTEstimatorCfg):
    """START estimator with per-foot 3x3 heightmap h_f (36-dim)."""

    foot_height_dim: int = 36


@configclass
class ParkourRslRlPIESTARTFootHmapFlatWarmupEstimatorCfg(ParkourRslRlPIESTARTFlatWarmupEstimatorCfg):
    """Flat-warmup START estimator with per-foot 3x3 heightmap h_f (36-dim)."""

    foot_height_dim: int = 36


@configclass
class ParkourRslRlPIESTARTFootHmapActorCriticCfg(ParkourRslRlPIEHighCapNoiseCapActorCriticCfg):
    """Actor for the per-foot heightmap variant: h_f=36 widens input to 182."""

    num_actor_obs: int = 164


@configclass
class UnitreeGo2PIESTARTFootHmapFlatWarmupPPORunnerCfg(UnitreeGo2PIESTARTFlatWarmupPPORunnerCfg):
    """Stage-0 flat warmup for the START per-foot-heightmap architecture."""

    estimator = ParkourRslRlPIESTARTFootHmapFlatWarmupEstimatorCfg()
    policy = ParkourRslRlPIESTARTFootHmapActorCriticCfg()


@configclass
class UnitreeGo2PIESTARTFootHmapStage2PPORunnerCfg(UnitreeGo2PIESTARTStage2PPORunnerCfg):
    """START Stage-2 obstacle runner with per-foot 3x3 heightmap h_f (36-dim)."""

    estimator = ParkourRslRlPIESTARTFootHmapEstimatorCfg()
    policy = ParkourRslRlPIESTARTFootHmapActorCriticCfg()


# ---------------------------------------------------------------------------
# START-aligned FootHmap estimator: per-foot 3x3 heightmap (36-dim) PLUS a
# lower AdaSmpl ceiling to fix the "z_m blind to depth" problem.
#
# Audit of the FootHmap from-scratch run (model_17500) showed
# depth_shuffle->z_m = 0.012 (z_m barely reacts to depth): because AdaSmpl fed
# the GROUND-TRUTH heightmap up to 80% of the time early on, the heightmap
# encoder learned to rely on clean GT input and stayed insensitive to the
# noisier depth reconstruction. On sparse-foothold terrain (stepping stones /
# balance beam) z_m MUST read terrain precisely, so this is critical. Fix:
# lower pie_adasmpl_max_prob 0.8 -> 0.5 so at least half the batch always
# encodes the reconstruction, forcing z_m to learn from depth from the start.
# ---------------------------------------------------------------------------
@configclass
class ParkourRslRlPIESTARTFootHmapLowAdaEstimatorCfg(ParkourRslRlPIESTARTFootHmapEstimatorCfg):
    """FootHmap estimator with AdaSmpl ceiling at 0.65.

    History: started at 0.8 (START paper value), dropped to 0.5 after a
    ``depth_shuffle->z_m`` audit suggested z_m was depth-blind (relying on the
    GT heightmap fed by AdaSmpl). 0.5 starved early exploration too much on the
    sparse-foothold from-scratch run, so this is a middle ground (0.65): enough
    GT heightmap sampling to ease early sparse-reward exploration (START's
    intent) while still forcing ~35%+ of the batch through the depth
    reconstruction so z_m stays sensitive to depth."""

    pie_adasmpl_max_prob: float = 0.65
    # Separated TR-Net (START two-stage): upstream reconstructs the heightmap
    # (supervised by depth), downstream policy estimator reads z_m from that
    # reconstruction (no GRU-memory shortcut to the actor). This restores the
    # visual feed-forward link so the actor can SEE a gap ahead instead of
    # inferring it from body state after stepping into it. NOT checkpoint
    # compatible with the single-GRU runs (new pol_* params + 2-layer hidden).
    use_separated_trnet: bool = True
    # START wean-off (from-scratch 20000-iter run with the separated TR-Net):
    # hold the GT-sampling ceiling at 0.65 for the first 6000 iters (build basic
    # gait + heightmap reconstruction while CV is high), linearly anneal the
    # ceiling 0.65 -> 0 over 6000..16000, then pure reconstruction 16000..20000.
    # Forces the downstream policy onto the TR-Net's depth-driven reconstruction
    # by late training (= deployment), so the actor's terrain code z_m is
    # grounded in current vision rather than GT.
    pie_adasmpl_anneal_start: int = 6000
    pie_adasmpl_anneal_end: int = 16000


@configclass
class ParkourRslRlPIESTARTFootHmapLowAdaFlatWarmupEstimatorCfg(
    ParkourRslRlPIESTARTFootHmapFlatWarmupEstimatorCfg
):
    """Flat-warmup FootHmap estimator with AdaSmpl ceiling 0.5."""

    pie_adasmpl_max_prob: float = 0.5


@configclass
class UnitreeGo2PIESTARTFootHmapLowAdaStage2PPORunnerCfg(UnitreeGo2PIESTARTFootHmapStage2PPORunnerCfg):
    """START Stage-2 FootHmap runner with START-aligned reward env + AdaSmpl 0.5.

    Uses the same FootHmap actor (182) but the estimator caps AdaSmpl at 0.5 so
    z_m is forced to read the depth reconstruction (not just the GT heightmap),
    fixing the depth-blind z_m observed in the 0.8-ceiling run. Pair this runner
    with the STARTAligned / STARTSparse env (START-form rewards + slope removed),
    whose 36-dim foot target matches foot_height_dim=36.
    """

    estimator = ParkourRslRlPIESTARTFootHmapLowAdaEstimatorCfg()


@configclass
class UnitreeGo2PIESTARTFootHmapLowAdaFlatWarmupPPORunnerCfg(
    UnitreeGo2PIESTARTFootHmapFlatWarmupPPORunnerCfg
):
    """Flat-warmup FootHmap runner with AdaSmpl ceiling 0.5."""

    estimator = ParkourRslRlPIESTARTFootHmapLowAdaFlatWarmupEstimatorCfg()
