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
