# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
from parkour_isaaclab.envs.mdp import ParkourEvent
 
if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

def terminate_episode(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_roll: float = 1.5,
    max_pitch: float = 1.5,
    minimum_height: float | None = None,
):  
    reset_buf = torch.zeros((env.num_envs, ), dtype=torch.bool, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_state_w[:,3:7])
    roll_cutoff = torch.abs(wrap_to_pi(roll)) > max_roll
    pitch_cutoff = torch.abs(wrap_to_pi(pitch)) > max_pitch
    time_out_buf = env.episode_length_buf >= env.max_episode_length
    parkour_event: ParkourEvent =  env.parkour_manager.get_term('base_parkour')    
    reach_goal_cutoff = parkour_event.cur_goal_idx >= env.scene.terrain.cfg.terrain_generator.num_goals
    height_cutoff = asset.data.root_state_w[:, 2] < -0.25
    if minimum_height is not None:
        height_cutoff = torch.logical_or(
            height_cutoff,
            asset.data.root_state_w[:, 2] < minimum_height,
        )
    time_out_buf |= reach_goal_cutoff
    reset_buf |= time_out_buf
    reset_buf |= roll_cutoff
    reset_buf |= pitch_cutoff
    reset_buf |= height_cutoff
    return reset_buf


def time_out(env: ParkourManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode due to the external episode time limit."""
    return env.episode_length_buf >= env.max_episode_length


def goal_reached(env: ParkourManagerBasedRLEnv, parkour_name: str = "base_parkour") -> torch.Tensor:
    """Terminate when the parkour goal sequence is completed."""
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    return parkour_event.cur_goal_idx >= env.scene.terrain.cfg.terrain_generator.num_goals


def bad_base_orientation(
    env: ParkourManagerBasedRLEnv,
    max_roll: float = 0.9,
    max_pitch: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when roll or pitch exceeds the configured limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_state_w[:, 3:7])
    roll_cutoff = torch.abs(wrap_to_pi(roll)) > max_roll
    pitch_cutoff = torch.abs(wrap_to_pi(pitch)) > max_pitch
    return torch.logical_or(roll_cutoff, pitch_cutoff)


def base_height_below_terrain(
    env: ParkourManagerBasedRLEnv,
    minimum_height: float = 0.18,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Terminate when base height relative to the local scanned terrain is too low."""
    asset: Articulation = env.scene[asset_cfg.name]
    ray_sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    terrain_z = torch.median(ray_sensor.data.ray_hits_w[..., 2], dim=1).values
    terrain_z = torch.nan_to_num(terrain_z, nan=0.0, posinf=0.0, neginf=0.0)
    return asset.data.root_state_w[:, 2] - terrain_z < minimum_height


def illegal_body_contact(
    env: ParkourManagerBasedRLEnv,
    threshold: float = 5.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["base", ".*_hip", ".*_thigh", ".*_calf", "Head_upper", "Head_lower"],
    ),
) -> torch.Tensor:
    """Terminate when non-foot body contact force exceeds the configured threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contact_force = torch.linalg.norm(net_contact_forces, dim=-1)
    return torch.any(torch.max(contact_force, dim=1).values > threshold, dim=1)
