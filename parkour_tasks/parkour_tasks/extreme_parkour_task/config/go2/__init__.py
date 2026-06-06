# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for velocity-based locomotion environments."""

# We leave this file empty since we don't want to expose any configs in this package directly.
# We still need this file to import the "config" module in the parent package.

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEParkourPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-LowNoise-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIELowNoisePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-LowerNoise-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIELowerNoisePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-LimitedAction-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIELimitedActionPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-TermFix-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_TermFix",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEParkourPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-FullFix-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_FullFix",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEParkourPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-ClipReward-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_ClipReward",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEParkourPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-LowNoise-ClipReward-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_ClipReward",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIELowNoisePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-LowNoise-TerminalPenalty-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_TerminalPenalty",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIELowNoisePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-StableEasy-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_StableEasy",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIELowNoisePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_StableEasyHeight",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIELowNoisePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-Gentle-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_StableEasyHeight",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEGentlePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-GentleLoadFix-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_StableEasyHeightGentleLoadFix",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEGentleLoadFixPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-FlatWalk-Unitree-Go2-GentleLoadFix-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFlatWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEGentleLoadFixPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-StairsBeam-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEStairsBeamEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEGentleLoadFixPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-StairsOnly-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEStairsOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEGentleLoadFixPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-GapOnly-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEGapOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEGentleLoadFixPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-FullParkour-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullParkourPPORunnerCfg",
    },
)

# Stage 2 warm-up variant: same env as FullParkour but action_limit=1.0
# (vs 1.2). Use this when fine-tuning a Stage-1 walking-bootstrap policy
# (action_limit=0.8) so the action scale jump is gentler. Once the
# policy stabilises on the obstacle mix, switch back to the
# Isaac-PIE-FullParkour-Unitree-Go2-v0 task with action_limit=1.2 to
# get the full joint range needed for harder hurdles / gaps.
gym.register(
    id="Isaac-PIE-FullParkour-Stage2Warm-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourStage2WarmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullStage2WarmPPORunnerCfg",
    },
)

# Easy variant: same env as Stage2Warm, but every sub-terrain difficulty=0
# corresponds to a 5 cm obstacle (gap / hurdle / step / incline). Lets a
# Stage-1 walker that has only seen a flat floor adapt to the multi-corridor
# layout while the obstacle is small enough that a normal trot stride clears
# it. difficulty=1 reaches roughly half of the full Stage2Warm peak so the
# curriculum still has plenty of room to ramp before switching to the regular
# Stage2Warm cfg for the final stage.
gym.register(
    id="Isaac-PIE-FullParkour-Easy-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourEasyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullStage2WarmPPORunnerCfg",
    },
)

# HighCap architecture on the Easy terrain. Use this to PLAY a HighCap
# checkpoint (z_m=64, num_actor_obs=150) on the gentler Easy curriculum, since
# the regular Easy task uses the z_m=32 runner whose actor dims do not match a
# HighCap checkpoint.
gym.register(
    id="Isaac-PIE-FullParkour-HighCap-Easy-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourEasyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullParkourHighCapPPORunnerCfg",
    },
)

# Front-fast variant: identical to Stage2Warm (same peak obstacle sizes at
# difficulty=1) but every sub-terrain difficulty formula is remapped with a
# two-slope knee at terrain level 4 (knee_value=0.6): obstacles grow FAST
# through the early levels the warm-start policy has already mastered, then
# grow SLOWLY above level 4 where the genuinely hard, new terrain begins.
# Use this to fine-tune a walker (e.g. easy_v5/model_19000 that reached
# ~level 4) so it blows past the easy region and gets a gentle ramp exactly
# where the linear Stage2Warm run diverged. Final (difficulty=1) target is
# unchanged.
gym.register(
    id="Isaac-PIE-FullParkour-Stage2WarmFrontFast-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourStage2WarmFrontFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullStage2WarmPPORunnerCfg",
    },
)

# HighCap Stage 0: pure flat walking warmup. FlatParkour env (single
# parkour_flat sub-terrain, no roughness, no domain randomisation) + HighCap
# flat-warmup runner (h_f/height losses OFF so flat ground does not teach a
# depth-ignoring proprio shortcut). Train this from scratch first to get the
# big HighCap network walking, then resume into HighCap on the obstacle mix
# with h_f/height/terrain_adaptive turned on.
gym.register(
    id="Isaac-PIE-FullParkour-HighCap-FlatWarmup-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFlatParkourWarmupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEHighCapFlatWarmupPPORunnerCfg",
    },
)

# Strategy B: high-capacity perception variant, from scratch. Same FrontFast
# curriculum env, but the runner uses the high-capacity estimator (z_m=64,
# depth feature map 8x12, wider height decoder, terrain_adaptive=2.0, h_f
# weight 2.0) and a matching num_actor_obs=150 actor. Aimed at the audited
# root cause: 5-12x worse height/h_f error on rough terrain. NOT checkpoint
# compatible with prior PIE runs (train from scratch).
gym.register(
    id="Isaac-PIE-FullParkour-HighCap-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourStage2WarmFrontFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullParkourHighCapPPORunnerCfg",
    },
)

# HighCap Stage 2: full obstacle terrain, resumed from the flat warmup. Same
# HighCap network + FrontFast curriculum, but reward uses tracking_goal_vel=1.5
# (carried from the flat warmup) and the estimator's h_f/height/terrain_adaptive
# losses are ON (HighCap runner). Resume the flat-warmup checkpoint into this
# task with --reset_optimizer_on_resume so Adam adapts to the newly-enabled
# terrain losses.
gym.register(
    id="Isaac-PIE-FullParkour-HighCap-Stage2-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourFrontFastStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullParkourHighCapPPORunnerCfg",
    },
)


# HighCap Stage 2 with a hard exploration-noise ceiling (std capped at 0.40).
# Same env + estimator + reward as the regular HighCap Stage 2, but the actor
# clamps its action noise std. This fixes the diagnosed Stage-2 collapse: the
# previous run's mean_noise_std climbed monotonically 0.03->0.74 and balance
# was lost once it crossed ~0.45 (episode length collapsed at iter ~4750-5000).
# Resume the flat-warmup model_3500 into this task with
# --reset_optimizer_on_resume (architecture is identical to the HighCap runner,
# so the checkpoint loads cleanly).
gym.register(
    id="Isaac-PIE-FullParkour-HighCap-Stage2-NoiseCap-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourFrontFastStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullParkourHighCapNoiseCapPPORunnerCfg",
    },
)


# START-aligned reward variant of the HighCap Stage2 NoiseCap task. Same
# HighCap NoiseCap network (resumes the same flat-warmup / Stage2 checkpoints)
# and same FrontFast curriculum, but: dof_error -0.04 -> -0.01, lin_vel_z and
# orientation swapped to the START paper forms (-2.0 / -1.0, applied on ALL
# terrain instead of being relaxed on obstacles), and the sloped-stone
# ``parkour`` sub-terrain removed (its 0.2 share split across gap/hurdle/flat/
# step). Use this to compare against the default Stage2 reward stack.
gym.register(
    id="Isaac-PIE-FullParkour-HighCap-Stage2-NoiseCap-STARTAligned-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourFrontFastStage2STARTAlignedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullParkourHighCapNoiseCapPPORunnerCfg",
    },
)


# START-sparse variant: the START-aligned Stage2 task (START-form rewards +
# sloped ``parkour`` sub-terrain removed) with the two START sparse-foothold
# terrains ADDED to the same full-parkour tile mix: stepping_stones (梅花桩)
# and balance_beam (独木桥). The six active obstacle terrains (gap / hurdle /
# flat / step / stepping_stones / balance_beam) share proportion evenly. Same
# HighCap NoiseCap network, so it resumes the same flat-warmup / Stage2
# checkpoints.
gym.register(
    id="Isaac-PIE-FullParkour-HighCap-Stage2-NoiseCap-STARTSparse-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourFrontFastStage2STARTSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullParkourHighCapNoiseCapPPORunnerCfg",
    },
)


# PLAY-ONLY task: same HighCap NoiseCap network + Stage2 obstacle curriculum,
# but relaxed termination cutoffs (min_height 0.12, roll/pitch 1.6 rad) so a
# visual play session is not cut short by brief dynamic poses. Use this only
# to watch a Stage2 checkpoint traverse obstacles; never train with it.
gym.register(
    id="Isaac-PIE-FullParkour-HighCap-Stage2-NoiseCap-Play-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourFrontFastStage2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullParkourHighCapNoiseCapPPORunnerCfg",
    },
)

# Terrain-adaptive (loss-only) variant: same FrontFast env, estimator uses
# terrain_adaptive=2.0 but unchanged network shapes, so it can resume from a
# FrontFast checkpoint (no architecture change).
gym.register(
    id="Isaac-PIE-FullParkour-Stage2WarmFrontFastTA-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourStage2WarmFrontFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFullStage2WarmTerrainAdaptivePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-FlatParkour-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFlatParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEFlatStage1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-Bridge-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_StableEasyHeightBridge",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEBridgePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-BridgeLoadFix-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_StableEasyHeightBridgeLoadFix",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEBridgeLoadFixPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-BridgeGaitFix-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_StableEasyHeightBridgeGaitFix",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEBridgePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-StableWarmup-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_StableWarmup",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIELowNoisePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEParkourPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-Parkour-Unitree-Go2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIEParkourPPORunnerCfg",
    },
)

# START architecture (arXiv 2409.15692) Stage 0: flat walking warmup. Same
# FlatParkour env, runner adds heightmap refine + heightmap-encoded z_m +
# AdaSmpl (terrain losses off on flat ground). Train from scratch first, then
# resume into the START Stage-2 task.
gym.register(
    id="Isaac-PIE-FullParkour-START-FlatWarmup-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFlatParkourWarmupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIESTARTFlatWarmupPPORunnerCfg",
    },
)

# START Stage 2: full obstacle terrain with two-stage heightmap reconstruction
# (rough MSE + U-Net-lite refine L1), heightmap-encoded z_m, AdaSmpl GT sampling,
# and noise-cap 0.40. Resume from the START flat warmup checkpoint with
# --reset_optimizer_on_resume. NOT compatible with HighCap checkpoints because
# z_m now encodes the heightmap rather than being a free head.
gym.register(
    id="Isaac-PIE-FullParkour-START-Stage2-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourFrontFastStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIESTARTStage2PPORunnerCfg",
    },
)

# START per-foot heightmap (Ĥᶠ) variants: h_f_hat regresses a 4-leg x 3x3 = 36
# local heightmap (arXiv 2409.15692 Appendix-C) instead of the 4-dim corridor
# demand. actor input grows to 182. NOT compatible with corridor/HighCap
# checkpoints. Two-stage: train FootHmap flat warmup, then resume FootHmap
# Stage-2 with --reset_optimizer_on_resume.
gym.register(
    id="Isaac-PIE-FullParkour-START-FootHmap-FlatWarmup-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFlatParkourWarmupFootHmapEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIESTARTFootHmapFlatWarmupPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PIE-FullParkour-START-FootHmap-Stage2-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourFrontFastStage2FootHmapEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIESTARTFootHmapStage2PPORunnerCfg",
    },
)

# START-sparse FootHmap variant: STARTSparse env (START-form rewards + slope
# removed + stepping_stones/balance_beam added) with the 36-dim per-foot
# heightmap estimator target, paired with the FootHmap actor (182) and a
# LOWERED AdaSmpl ceiling (0.5) to fix the depth-blind z_m seen at ceiling 0.8.
# This is the "full START alignment + sparse footholds + per-foot heightmap"
# task. NOT compatible with corridor (150) checkpoints.
gym.register(
    id="Isaac-PIE-FullParkour-START-FootHmap-Sparse-Stage2-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_pie_cfg:UnitreeGo2PIEFullParkourFrontFastStage2STARTSparseFootHmapEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_pie_ppo_cfg:UnitreeGo2PIESTARTFootHmapLowAdaStage2PPORunnerCfg",
    },
)
