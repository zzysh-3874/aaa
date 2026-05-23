from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi, quat_apply
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from collections.abc import Sequence

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

import cv2
import numpy as np 

class reward_feet_edge(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.parkour_event: ParkourEvent =  env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.body_id = self.contact_sensor.find_bodies('base')[0]
        self.horizontal_scale = env.scene.terrain.cfg.terrain_generator.horizontal_scale
        size_x, size_y = env.scene.terrain.cfg.terrain_generator.size
        self.rows_offset = (size_x * env.scene.terrain.cfg.terrain_generator.num_rows/2)
        self.cols_offset = (size_y * env.scene.terrain.cfg.terrain_generator.num_cols/2)
        total_x_edge_maskes = torch.from_numpy(self.parkour_event.terrain.terrain_generator_class.x_edge_maskes).to(device = self.device)
        self.x_edge_masks_tensor = total_x_edge_maskes.permute(0, 2, 1, 3).reshape(
            env.scene.terrain.terrain_generator_class.total_width_pixels, env.scene.terrain.terrain_generator_class.total_length_pixels
        )

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        ) -> torch.Tensor:
        feet_pos_x = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,0] + self.rows_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_y = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,1] + self.cols_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_x = torch.clip(feet_pos_x, 0, self.x_edge_masks_tensor.shape[0]-1)
        feet_pos_y = torch.clip(feet_pos_y, 0, self.x_edge_masks_tensor.shape[1]-1)
        feet_at_edge = self.x_edge_masks_tensor[feet_pos_x, feet_pos_y]
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        contact = torch.norm(contact_forces, dim=-1) > 2.
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        self.feet_at_edge = contact_filt & feet_at_edge
        rew = (self.parkour_event.terrain.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        ## This is for debugging to matching index and x_edge_mask
        # origin = self.x_edge_masks_tensor.detach().cpu().numpy().astype(np.uint8) * 255
        # cv2.imshow('origin',origin)
        # origin[feet_pos_x.detach().cpu().numpy(), feet_pos_y.detach().cpu().numpy()] -= 100
        # cv2.imshow('feet_edge',origin)
        # cv2.waitKey(1)
        return rew

def reward_torques(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)

def reward_dof_error(    
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

def reward_hip_pos(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] \
                                    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)


def reward_foot_mirror_diagonal(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fl_joint_ids: tuple[int, ...] = (4, 8),   # FL thigh, FL calf
    fr_joint_ids: tuple[int, ...] = (5, 9),   # FR thigh, FR calf
    rl_joint_ids: tuple[int, ...] = (6, 10),  # RL thigh, RL calf
    rr_joint_ids: tuple[int, ...] = (7, 11),  # RR thigh, RR calf
) -> torch.Tensor:
    """Penalise asymmetry between diagonal legs in joint-offset space.

    Trot gait is characterised by FL+RR moving together and FR+RL moving
    together. We compare the offset from the default joint pose so that the
    built-in front/back thigh bias (0.8 vs 1.0) does not produce a constant
    penalty. Hip joints are intentionally excluded so yaw commands do not get
    artificially suppressed.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    offset = asset.data.joint_pos - asset.data.default_joint_pos
    fl = offset[:, list(fl_joint_ids)]
    fr = offset[:, list(fr_joint_ids)]
    rl = offset[:, list(rl_joint_ids)]
    rr = offset[:, list(rr_joint_ids)]
    diff1 = torch.sum(torch.square(fl - rr), dim=1)  # FL should track RR
    diff2 = torch.sum(torch.square(fr - rl), dim=1)  # FR should track RL
    return 0.5 * (diff1 + diff2)


def reward_no_fly(
    env: ParkourManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalise frames where all four feet are simultaneously in the air.

    Returns 1.0 for env-steps where every foot has vertical force below
    ``contact_threshold`` Newtons, and 0.0 otherwise. Use a negative weight to
    punish pronk / aerial gaits while leaving regular trot (diagonal pair
    support) unaffected.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    vertical_force = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2]
    in_air = (vertical_force < contact_threshold).all(dim=1)
    return in_air.float()


def reward_feet_clearance_stairs(
    env: ParkourManagerBasedRLEnv,
    foot_sensor_names: tuple[str, str, str, str] = (
        "foot_scanner_fl",
        "foot_scanner_fr",
        "foot_scanner_rl",
        "foot_scanner_rr",
    ),
    contact_sensor_cfg: SceneEntityCfg | None = None,
    target_clearance: float = 0.10,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward lifting swing feet to at least ``target_clearance`` above the
    terrain directly beneath them.

    For each foot:
        clearance = foot_z - terrain_z  (from the per-foot ray caster)
        if foot is in swing (not in contact):
            reward += min(clearance, target_clearance) / target_clearance

    Sum across the four feet. Each foot contributes at most 1.0 per step. The
    clearance is clipped at target_clearance so the policy is not rewarded for
    lifting feet to absurd heights, only for reaching a clear clearance margin.
    Feet that are currently in contact (stance phase) do not contribute, so
    trot-style gaits don't get penalised.
    """
    from isaaclab.sensors import RayCaster, ContactSensor

    clearances = []
    for name in foot_sensor_names:
        ray: RayCaster = env.scene.sensors[name]
        foot_z = ray.data.pos_w[:, 2]
        terrain_z = ray.data.ray_hits_w[:, 0, 2]
        clearances.append(foot_z - terrain_z)
    clearance = torch.stack(clearances, dim=-1).clamp(min=0.0, max=target_clearance)

    # Only reward swing feet.
    if contact_sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
        vertical_force = contact_sensor.data.net_forces_w_history[:, 0, contact_sensor_cfg.body_ids, 2]
        in_swing = (vertical_force < contact_threshold).float()
    else:
        in_swing = torch.ones_like(clearance)

    reward = (clearance / target_clearance) * in_swing
    return reward.sum(dim=-1)


def reward_stand_still(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Legged-gym style stand-still reward.

    Only active when the commanded velocity norm is below ``command_threshold``.
    In that case, penalise any joint deviation from the default pose so the
    policy converges to a clean standing posture instead of leaning forward.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_offset = asset.data.joint_pos - asset.data.default_joint_pos
    dev = torch.sum(torch.abs(joint_offset), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    mask = (cmd_norm < command_threshold).float()
    return dev * mask


class reward_forward_distance(ManagerTermBase):
    """Reward forward progress in the +x direction (body frame projected to world).

    Each step, reward = clamp(delta_x, min=0). Only positive progress counts;
    going backwards gives 0 (not negative). This directly incentivises the
    policy to move forward through obstacles rather than standing still or
    walking in circles.
    """

    def __init__(self, cfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.prev_pos_x = self.asset.data.root_pos_w[:, 0].clone()

    def reset(self, env_ids=None):
        if env_ids is None:
            self.prev_pos_x[:] = self.asset.data.root_pos_w[:, 0]
        else:
            self.prev_pos_x[env_ids] = self.asset.data.root_pos_w[env_ids, 0]

    def __call__(self, env: ParkourManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        cur_x = self.asset.data.root_pos_w[:, 0]
        delta_x = cur_x - self.prev_pos_x
        self.prev_pos_x = cur_x.clone()
        return delta_x.clamp(min=0.0)

def reward_ang_vel_xy(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:,:2]), dim=1)

def reward_lin_vel_xy_command_tracking(
    env: ParkourManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = command[:, :2] - asset.data.root_lin_vel_b[:, :2]
    return torch.exp(-4.0 * torch.sum(torch.square(error), dim=1))


def reward_ang_vel_yaw_command_tracking(
    env: ParkourManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = command[:, 2] - asset.data.root_ang_vel_b[:, 2]
    return torch.exp(-4.0 * torch.square(error))

def reward_lin_vel_z_paper(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def reward_orientation_paper(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def reward_joint_power(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque) * torch.abs(asset.data.joint_vel), dim=1)


def reward_joint_power_gap_aware(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    parkour_name: str = "base_parkour",
    in_zone_scale: float = 0.25,
) -> torch.Tensor:
    """Joint power with the per-env penalty scaled down inside the gap zone.

    Returns ``Σ|τ_i|·|q̇_i| * scale`` per env, where ``scale`` is
    ``in_zone_scale`` (default 0.25) when the env is currently inside a
    padded gap zone (i.e. about to or just crossed a gap), and ``1.0``
    otherwise. With the default in_zone_scale=0.25 and a Reward weight of
    ``-8e-5``, the effective per-step joint-power penalty becomes:
        in zone : -8e-5 * 0.25 = -2e-5  (legacy gap-friendly value)
        out zone: -8e-5 * 1.00 = -8e-5  (4x stronger to discourage the
                                          jump-everywhere "v9" gait on
                                          flat ground / between gaps)

    The sub-terrain frame x is read from ``ParkourEvent.is_in_gap_zone``;
    parkour_flat tiles always return False so this reduces to the legacy
    ``-8e-5 * power`` everywhere on flat sub-terrains.
    """
    from parkour_isaaclab.envs.mdp.parkours import ParkourEvent

    asset: Articulation = env.scene[asset_cfg.name]
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    power = torch.sum(
        torch.abs(asset.data.applied_torque) * torch.abs(asset.data.joint_vel),
        dim=1,
    )
    in_zone = parkour_event.is_in_gap_zone()
    scale = torch.where(
        in_zone,
        torch.full_like(power, in_zone_scale),
        torch.ones_like(power),
    )
    return power * scale

def reward_action_rate_squared(
    env: ParkourManagerBasedRLEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    action_history = action_term.action_history_buf
    return torch.sum(torch.square(action_history[:, -1] - action_history[:, -2]), dim=1)

def reward_action_smoothness(
    env: ParkourManagerBasedRLEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    action_history = action_term.action_history_buf
    action_acc = action_history[:, -1] - 2.0 * action_history[:, -2] + action_history[:, -3]
    return torch.sum(torch.square(action_acc), dim=1)

def reward_action_magnitude(
    env: ParkourManagerBasedRLEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(action_term.raw_actions), dim=1)


def reward_failure_terminal_penalty(
    env: ParkourManagerBasedRLEnv,
    term_names: Sequence[str] = ("bad_base_orientation", "low_base_height", "illegal_body_contact"),
) -> torch.Tensor:
    failed = torch.zeros_like(env.reset_terminated, dtype=torch.bool)
    for term_name in term_names:
        failed |= env.termination_manager.get_term(term_name)
    return failed.float()


def reward_base_height_below_target(
    env: ParkourManagerBasedRLEnv,
    target_height: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ray_sensor = env.scene.sensors[sensor_cfg.name]
    terrain_z = torch.median(ray_sensor.data.ray_hits_w[..., 2], dim=1).values
    terrain_z = torch.nan_to_num(terrain_z, nan=0.0, posinf=0.0, neginf=0.0)
    base_height = asset.data.root_state_w[:, 2] - terrain_z
    return torch.clamp(target_height - base_height, min=0.0)


class reward_action_rate(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.previous_actions = torch.zeros(env.num_envs, 2, asset.num_joints, dtype=torch.float, device=self.device)
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.previous_actions.zero_()
        else:
            self.previous_actions[env_ids, 0, :] = 0.0
            self.previous_actions[env_ids, 1, :] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg | None = None,
        action_name: str = "joint_pos",
        ) -> torch.Tensor:
        self.previous_actions[:, 0, :] = self.previous_actions[:, 1, :]
        self.previous_actions[:, 1, :] = env.action_manager.get_term(action_name).raw_actions
        return torch.norm(self.previous_actions[:, 1, :] - self.previous_actions[:, 0, :], dim=1)

class reward_feet_contact_balance(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.contact_duty = torch.zeros(env.num_envs, 4, dtype=torch.float, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.contact_duty[env_ids, :] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        ema_alpha: float = 0.02,
    ) -> torch.Tensor:
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids]
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids]
        contact = torch.norm(contact_forces, dim=-1) > 2.0
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.0
        contact_filt = torch.logical_or(contact, last_contacts).float()
        self.contact_duty = (1.0 - ema_alpha) * self.contact_duty + ema_alpha * contact_filt
        return torch.square(self.contact_duty.max(dim=1).values - self.contact_duty.min(dim=1).values)

class reward_feet_vertical_force_balance(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.force_ema = torch.zeros(env.num_envs, 4, dtype=torch.float, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.force_ema[env_ids, :] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        ema_alpha: float = 0.02,
        min_total_force: float = 20.0,
    ) -> torch.Tensor:
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids]
        vertical_force = torch.clamp(contact_forces[..., 2], min=0.0)
        self.force_ema = (1.0 - ema_alpha) * self.force_ema + ema_alpha * vertical_force
        total_force = self.force_ema.sum(dim=1, keepdim=True)
        force_share = self.force_ema / total_force.clamp_min(1.0)
        balance_error = torch.sum(torch.square(force_share - 0.25), dim=1)
        return balance_error * (total_force.squeeze(1) > min_total_force).float()

class reward_feet_min_force_share(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.force_ema = torch.zeros(env.num_envs, 4, dtype=torch.float, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.force_ema.zero_()
        else:
            self.force_ema[env_ids, :] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        ema_alpha: float = 0.02,
        min_share: float = 0.14,
        min_total_force: float = 20.0,
    ) -> torch.Tensor:
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids]
        vertical_force = torch.clamp(contact_forces[..., 2], min=0.0)
        self.force_ema = (1.0 - ema_alpha) * self.force_ema + ema_alpha * vertical_force
        total_force = self.force_ema.sum(dim=1, keepdim=True)
        force_share = self.force_ema / total_force.clamp_min(1.0)
        share_deficit = torch.clamp(min_share - force_share, min=0.0)
        penalty = torch.sum(torch.square(share_deficit), dim=1)
        return penalty * (total_force.squeeze(1) > min_total_force).float()
    
class reward_dof_acc(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_joint_vel = torch.zeros(env.num_envs, 2,  asset.num_joints, dtype= torch.float ,device=self.device)
        self.dt = env.cfg.decimation * env.cfg.sim.dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.previous_joint_vel.zero_()
        else:
            self.previous_joint_vel[env_ids, 0, :] = 0.0
            self.previous_joint_vel[env_ids, 1, :] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        self.previous_joint_vel[:, 0, :] = self.previous_joint_vel[:, 1, :]
        self.previous_joint_vel[:, 1, :] = asset.data.joint_vel
        return torch.sum(torch.square((self.previous_joint_vel[:, 1, :] - self.previous_joint_vel[:,0,:]) / self.dt), dim=1)
        
def reward_lin_vel_z(
    env: ParkourManagerBasedRLEnv,        
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.square(asset.data.root_lin_vel_b[:, 2])
    rew[(terrain_names !='parkour_flat')[:,-1]] *= 0.5
    return rew

def reward_orientation(
    env: ParkourManagerBasedRLEnv,   
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    rew[(terrain_names !='parkour_flat')[:,-1]] = 0.
    return rew

def reward_feet_stumble(
    env: ParkourManagerBasedRLEnv,        
    sensor_cfg: SceneEntityCfg ,
    horizontal_force_ratio: float = 4.0,
    min_vertical_force: float = 1.0,
    ) -> torch.Tensor: 
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
    horizontal_force = torch.norm(net_contact_forces[:, :, :2], dim=2)
    vertical_force = torch.abs(net_contact_forces[:, :, 2]).clamp_min(min_vertical_force)
    rew = torch.any(horizontal_force > horizontal_force_ratio * vertical_force, dim=1)
    return rew.float()

def reward_feet_slip(
    env: ParkourManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    horizontal_force_ratio: float = 2.0,
    min_vertical_force: float = 5.0,
    contact_force_threshold: float = 2.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    horizontal_force = torch.norm(net_contact_forces[:, :, :2], dim=2)
    vertical_force = torch.abs(net_contact_forces[:, :, 2])
    contact = torch.norm(net_contact_forces, dim=-1) > contact_force_threshold
    excess_ratio = horizontal_force / vertical_force.clamp_min(min_vertical_force) - horizontal_force_ratio
    return torch.sum(torch.square(torch.clamp(excess_ratio, min=0.0)) * contact.float(), dim=1)

def reward_tracking_goal_vel(
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    target_pos_rel = parkour_event.target_pos_rel
    target_vel = target_pos_rel / (torch.norm(target_pos_rel, dim=-1, keepdim=True) + 1e-5)
    cur_vel = asset.data.root_vel_w[:, :2]
    proj_vel = torch.sum(target_vel * cur_vel, dim=-1)
    command_vel = env.command_manager.get_command('base_velocity')[:, 0]
    rew_move = torch.minimum(proj_vel, command_vel) / (command_vel + 1e-5)
    # Only reward positive progress toward the goal; do NOT penalise transient
    # negative projected velocity. When the robot crosses a goal waypoint the
    # next target_pos_rel can briefly flip sign (if the upcoming goal is close
    # or even slightly behind, e.g. beam waypoints), which would produce a
    # spurious negative reward that confuses the policy.
    rew_move = rew_move.clamp(min=0.0)
    return rew_move

def reward_tracking_yaw(     
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    yaw = torch.atan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
                    1 - 2*(q[:,2]**2 + q[:,3]**2))
    return torch.exp(-torch.abs((parkour_event.target_yaw - yaw)))


class reward_delta_torques(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_torque = torch.zeros(env.num_envs, 2,  self.asset.num_joints, dtype= torch.float ,device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.previous_torque.zero_()
        else:
            self.previous_torque[env_ids, 0, :] = 0.0
            self.previous_torque[env_ids, 1, :] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_torque[:, 0, :] = self.previous_torque[:, 1, :]
        self.previous_torque[:, 1, :] = self.asset.data.applied_torque
        return torch.sum(torch.square((self.previous_torque[:, 1, :] - self.previous_torque[:,0,:])), dim=1)

def reward_collision(
    env: ParkourManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg ,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
    return torch.sum(1.*(torch.norm(net_contact_forces, dim=-1) > 0.1), dim=1)


def reward_contact_force_above_threshold(
    env: ParkourManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contact_force = torch.linalg.norm(net_contact_forces, dim=-1)
    peak_force = torch.max(contact_force, dim=1).values
    return torch.sum(torch.clamp(peak_force - threshold, min=0.0), dim=1)


class reward_goal_reached(ManagerTermBase):
    """One-shot bonus when the parkour goal index advances.

    Detects ``cur_goal_idx`` increment between consecutive steps. Each
    increment yields +1.0 (multiply by RewTerm weight in cfg). Useful for
    sparse goal rewards on top of dense ``tracking_goal_vel``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.parkour_event: ParkourEvent = env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.previous_goal_idx = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.previous_goal_idx.zero_()
        else:
            self.previous_goal_idx[env_ids] = 0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        parkour_name: str = "base_parkour",
    ) -> torch.Tensor:
        cur = self.parkour_event.cur_goal_idx
        # Reward = number of new goals reached this step (usually 0 or 1).
        rew = (cur > self.previous_goal_idx).float()
        self.previous_goal_idx = cur.clone()
        return rew
