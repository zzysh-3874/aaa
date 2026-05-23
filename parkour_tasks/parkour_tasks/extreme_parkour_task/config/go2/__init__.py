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
