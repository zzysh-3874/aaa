"""Diagnose per-leg usage for a PIE checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != _SCRIPT_DIR]
sys.path.insert(0, str(_REPO_ROOT))

_CLI_ARGS_PATH = _SCRIPT_DIR / "rsl_rl" / "cli_args.py"
_CLI_ARGS_SPEC = importlib.util.spec_from_file_location("parkour_rsl_cli_args", _CLI_ARGS_PATH)
cli_args = importlib.util.module_from_spec(_CLI_ARGS_SPEC)
assert _CLI_ARGS_SPEC.loader is not None
_CLI_ARGS_SPEC.loader.exec_module(cli_args)


parser = argparse.ArgumentParser(description="Diagnose per-leg contact, load, action, and joint usage.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--steps", type=int, default=1000, help="Policy rollout steps to summarize.")
parser.add_argument("--zero_steps", type=int, default=300, help="Zero-action baseline steps to summarize.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Steps to skip before recording each mode.")
parser.add_argument("--contact_threshold", type=float, default=2.0, help="Contact force norm threshold.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_tasks.utils import parse_env_cfg
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import parkour_tasks  # noqa: F401
import parkour_tasks.extreme_parkour_task.config.go2  # noqa: F401


LEG_NAMES = ("FL", "FR", "RL", "RR")
FOOT_NAMES = tuple(f"{leg}_foot" for leg in LEG_NAMES)
JOINT_NAMES = tuple(
    tuple(f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf")) for leg in LEG_NAMES
)


@dataclass
class LegStats:
    samples: int
    contact_sum: torch.Tensor
    vertical_force_sum: torch.Tensor
    force_share_sum: torch.Tensor
    vertical_force_when_contact_sum: torch.Tensor
    forward_force_sum: torch.Tensor
    forward_force_positive_sum: torch.Tensor
    forward_force_negative_sum: torch.Tensor
    horizontal_force_sum: torch.Tensor
    foot_slip_when_contact_sum: torch.Tensor
    action_sq_sum: torch.Tensor
    action_abs_sum: torch.Tensor
    joint_pos_sq_sum: torch.Tensor
    joint_vel_sq_sum: torch.Tensor
    torque_sq_sum: torch.Tensor
    torque_abs_sum: torch.Tensor
    joint_power_positive_sum: torch.Tensor
    joint_power_abs_sum: torch.Tensor
    joint_action_sum: torch.Tensor
    joint_action_sq_sum: torch.Tensor
    joint_action_abs_sum: torch.Tensor
    joint_pos_offset_sum: torch.Tensor
    joint_pos_offset_sq_sum: torch.Tensor
    joint_vel_sum: torch.Tensor
    joint_vel_sq_detail_sum: torch.Tensor
    joint_torque_sum: torch.Tensor
    joint_torque_sq_sum: torch.Tensor
    joint_torque_abs_sum: torch.Tensor
    joint_power_detail_positive_sum: torch.Tensor
    joint_power_detail_abs_sum: torch.Tensor

    @classmethod
    def create(cls, device: torch.device | str) -> "LegStats":
        return cls(
            samples=0,
            contact_sum=torch.zeros(4, device=device),
            vertical_force_sum=torch.zeros(4, device=device),
            force_share_sum=torch.zeros(4, device=device),
            vertical_force_when_contact_sum=torch.zeros(4, device=device),
            forward_force_sum=torch.zeros(4, device=device),
            forward_force_positive_sum=torch.zeros(4, device=device),
            forward_force_negative_sum=torch.zeros(4, device=device),
            horizontal_force_sum=torch.zeros(4, device=device),
            foot_slip_when_contact_sum=torch.zeros(4, device=device),
            action_sq_sum=torch.zeros(4, device=device),
            action_abs_sum=torch.zeros(4, device=device),
            joint_pos_sq_sum=torch.zeros(4, device=device),
            joint_vel_sq_sum=torch.zeros(4, device=device),
            torque_sq_sum=torch.zeros(4, device=device),
            torque_abs_sum=torch.zeros(4, device=device),
            joint_power_positive_sum=torch.zeros(4, device=device),
            joint_power_abs_sum=torch.zeros(4, device=device),
            joint_action_sum=torch.zeros(4, 3, device=device),
            joint_action_sq_sum=torch.zeros(4, 3, device=device),
            joint_action_abs_sum=torch.zeros(4, 3, device=device),
            joint_pos_offset_sum=torch.zeros(4, 3, device=device),
            joint_pos_offset_sq_sum=torch.zeros(4, 3, device=device),
            joint_vel_sum=torch.zeros(4, 3, device=device),
            joint_vel_sq_detail_sum=torch.zeros(4, 3, device=device),
            joint_torque_sum=torch.zeros(4, 3, device=device),
            joint_torque_sq_sum=torch.zeros(4, 3, device=device),
            joint_torque_abs_sum=torch.zeros(4, 3, device=device),
            joint_power_detail_positive_sum=torch.zeros(4, 3, device=device),
            joint_power_detail_abs_sum=torch.zeros(4, 3, device=device),
        )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    if not getattr(runner.alg, "use_pie_actor_features", False):
        raise RuntimeError("This diagnostic expects a PIE actor inference wrapper.")
    policy = runner.get_pie_inference_policy(device=env.unwrapped.device)

    asset = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    foot_ids, foot_names = contact_sensor.find_bodies(FOOT_NAMES, preserve_order=True)
    action_joint_ids = getattr(action_term, "_joint_ids", None)
    action_joint_names = getattr(action_term, "_joint_names", None)
    torque_source = "applied_torque" if getattr(asset.data, "applied_torque", None) is not None else "computed_torque"
    asset_foot_ids, asset_foot_names = asset.find_bodies(FOOT_NAMES, preserve_order=True)
    joint_ids_by_leg = []
    joint_names_by_leg = []
    for names in JOINT_NAMES:
        joint_ids, joint_names = asset.find_joints(names, preserve_order=True)
        joint_ids_by_leg.append(torch.tensor(joint_ids, device=env.device, dtype=torch.long))
        joint_names_by_leg.append(tuple(joint_names))

    print("Resolved feet:", list(zip(LEG_NAMES, foot_ids, foot_names)))
    print("Resolved asset feet:", list(zip(LEG_NAMES, asset_foot_ids, asset_foot_names)))
    print("Action joint ids:", action_joint_ids)
    print("Action joint names:", action_joint_names)
    print("Torque source:", torque_source)
    print("Resolved joints:")
    for leg, ids, names in zip(LEG_NAMES, joint_ids_by_leg, joint_names_by_leg):
        print(f"  {leg}: ids={ids.detach().cpu().tolist()} names={names}")

    zero_stats = run_mode(env, policy, mode="zero", num_steps=args_cli.zero_steps, warmup_steps=args_cli.warmup_steps)
    print_summary("ZERO ACTION BASELINE", zero_stats)

    policy_stats = run_mode(env, policy, mode="policy", num_steps=args_cli.steps, warmup_steps=args_cli.warmup_steps)
    print_summary("POLICY", policy_stats)

    env.close()


def run_mode(
    env: ParkourRslRlVecEnvWrapper,
    policy,
    mode: str,
    num_steps: int,
    warmup_steps: int,
) -> LegStats:
    obs, extras = env.reset()
    policy.reset()
    stats = LegStats.create(env.device)
    for step in range(num_steps + warmup_steps):
        with torch.inference_mode():
            if mode == "zero":
                actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            elif mode == "policy":
                actions = policy(extras["observations"], hist_encoding=True)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        obs, _, dones, extras = env.step(actions)
        if mode == "policy":
            policy.reset(dones)
        if step >= warmup_steps:
            accumulate_stats(env, stats)
    return stats


def accumulate_stats(env: ParkourRslRlVecEnvWrapper, stats: LegStats) -> None:
    asset = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    foot_ids, _ = contact_sensor.find_bodies(FOOT_NAMES, preserve_order=True)
    foot_ids_tensor = torch.tensor(foot_ids, device=env.device, dtype=torch.long)
    joint_ids_by_leg = [
        torch.tensor(asset.find_joints(names, preserve_order=True)[0], device=env.device, dtype=torch.long)
        for names in JOINT_NAMES
    ]

    contact_forces = contact_sensor.data.net_forces_w_history[:, 0, foot_ids_tensor]
    force_norm = torch.linalg.norm(contact_forces, dim=-1)
    vertical_force = torch.clamp(contact_forces[..., 2], min=0.0)
    total_vertical = vertical_force.sum(dim=1, keepdim=True).clamp_min(1.0)
    force_share = vertical_force / total_vertical
    contact = (force_norm > args_cli.contact_threshold).float()
    contact_count = contact.sum(dim=0).clamp_min(1.0)

    root_yaw = yaw_quat(asset.data.root_quat_w)
    repeated_root_yaw = root_yaw[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
    contact_forces_b = quat_apply_inverse(repeated_root_yaw, contact_forces.reshape(-1, 3)).reshape(-1, 4, 3)
    forward_force = contact_forces_b[..., 0]
    horizontal_force = torch.linalg.norm(contact_forces_b[..., :2], dim=-1)

    asset_foot_ids, _ = asset.find_bodies(FOOT_NAMES, preserve_order=True)
    asset_foot_ids_tensor = torch.tensor(asset_foot_ids, device=env.device, dtype=torch.long)
    foot_lin_vel_b = None
    foot_lin_vel_w = getattr(asset.data, "body_lin_vel_w", None)
    if foot_lin_vel_w is not None:
        foot_lin_vel_w = foot_lin_vel_w[:, asset_foot_ids_tensor]
        foot_lin_vel_b = quat_apply_inverse(repeated_root_yaw, foot_lin_vel_w.reshape(-1, 3)).reshape(-1, 4, 3)

    raw_actions = action_term.raw_actions
    joint_pos_error = asset.data.joint_pos - asset.data.default_joint_pos
    joint_vel = asset.data.joint_vel

    stats.contact_sum += contact.mean(dim=0)
    stats.vertical_force_sum += vertical_force.mean(dim=0)
    stats.force_share_sum += force_share.mean(dim=0)
    stats.vertical_force_when_contact_sum += (vertical_force * contact).sum(dim=0) / contact_count
    stats.forward_force_sum += forward_force.mean(dim=0)
    stats.forward_force_positive_sum += torch.clamp(forward_force, min=0.0).mean(dim=0)
    stats.forward_force_negative_sum += torch.clamp(-forward_force, min=0.0).mean(dim=0)
    stats.horizontal_force_sum += horizontal_force.mean(dim=0)
    if foot_lin_vel_b is not None:
        foot_slip = torch.linalg.norm(foot_lin_vel_b[..., :2], dim=-1)
        stats.foot_slip_when_contact_sum += (foot_slip * contact).sum(dim=0) / contact_count
    for leg_idx, joint_ids in enumerate(joint_ids_by_leg):
        leg_actions = raw_actions[:, joint_ids]
        leg_joint_pos_error = joint_pos_error[:, joint_ids]
        leg_joint_vel = joint_vel[:, joint_ids]
        stats.action_sq_sum[leg_idx] += torch.mean(torch.square(leg_actions))
        stats.action_abs_sum[leg_idx] += torch.mean(torch.abs(leg_actions))
        stats.joint_pos_sq_sum[leg_idx] += torch.mean(torch.square(leg_joint_pos_error))
        stats.joint_vel_sq_sum[leg_idx] += torch.mean(torch.square(leg_joint_vel))
        stats.joint_action_sum[leg_idx] += leg_actions.mean(dim=0)
        stats.joint_action_sq_sum[leg_idx] += torch.square(leg_actions).mean(dim=0)
        stats.joint_action_abs_sum[leg_idx] += torch.abs(leg_actions).mean(dim=0)
        stats.joint_pos_offset_sum[leg_idx] += leg_joint_pos_error.mean(dim=0)
        stats.joint_pos_offset_sq_sum[leg_idx] += torch.square(leg_joint_pos_error).mean(dim=0)
        stats.joint_vel_sum[leg_idx] += leg_joint_vel.mean(dim=0)
        stats.joint_vel_sq_detail_sum[leg_idx] += torch.square(leg_joint_vel).mean(dim=0)
        torque = getattr(asset.data, "applied_torque", None)
        if torque is None:
            torque = getattr(asset.data, "computed_torque", None)
        if torque is not None:
            leg_torque = torque[:, joint_ids]
            stats.torque_sq_sum[leg_idx] += torch.mean(torch.square(leg_torque))
            stats.torque_abs_sum[leg_idx] += torch.mean(torch.abs(leg_torque))
            stats.joint_torque_sum[leg_idx] += leg_torque.mean(dim=0)
            stats.joint_torque_sq_sum[leg_idx] += torch.square(leg_torque).mean(dim=0)
            stats.joint_torque_abs_sum[leg_idx] += torch.abs(leg_torque).mean(dim=0)
            joint_power = leg_torque * leg_joint_vel
            stats.joint_power_positive_sum[leg_idx] += torch.mean(torch.clamp(joint_power, min=0.0))
            stats.joint_power_abs_sum[leg_idx] += torch.mean(torch.abs(joint_power))
            stats.joint_power_detail_positive_sum[leg_idx] += torch.clamp(joint_power, min=0.0).mean(dim=0)
            stats.joint_power_detail_abs_sum[leg_idx] += torch.abs(joint_power).mean(dim=0)
    stats.samples += 1


def print_summary(title: str, stats: LegStats) -> None:
    denom = max(stats.samples, 1)
    contact = stats.contact_sum / denom
    vertical_force = stats.vertical_force_sum / denom
    force_share = stats.force_share_sum / denom
    vertical_force_when_contact = stats.vertical_force_when_contact_sum / denom
    forward_force = stats.forward_force_sum / denom
    forward_force_positive = stats.forward_force_positive_sum / denom
    forward_force_negative = stats.forward_force_negative_sum / denom
    horizontal_force = stats.horizontal_force_sum / denom
    foot_slip_when_contact = stats.foot_slip_when_contact_sum / denom
    action_rms = torch.sqrt(stats.action_sq_sum / denom)
    action_abs = stats.action_abs_sum / denom
    joint_pos_rms = torch.sqrt(stats.joint_pos_sq_sum / denom)
    joint_vel_rms = torch.sqrt(stats.joint_vel_sq_sum / denom)
    torque_rms = torch.sqrt(stats.torque_sq_sum / denom)
    torque_abs = stats.torque_abs_sum / denom
    joint_power_positive = stats.joint_power_positive_sum / denom
    joint_power_abs = stats.joint_power_abs_sum / denom
    joint_action_mean = stats.joint_action_sum / denom
    joint_action_rms = torch.sqrt(stats.joint_action_sq_sum / denom)
    joint_action_abs = stats.joint_action_abs_sum / denom
    joint_pos_offset_mean = stats.joint_pos_offset_sum / denom
    joint_pos_offset_rms = torch.sqrt(stats.joint_pos_offset_sq_sum / denom)
    joint_vel_mean = stats.joint_vel_sum / denom
    joint_vel_detail_rms = torch.sqrt(stats.joint_vel_sq_detail_sum / denom)
    joint_torque_mean = stats.joint_torque_sum / denom
    joint_torque_rms = torch.sqrt(stats.joint_torque_sq_sum / denom)
    joint_torque_abs = stats.joint_torque_abs_sum / denom
    joint_power_detail_positive = stats.joint_power_detail_positive_sum / denom
    joint_power_detail_abs = stats.joint_power_detail_abs_sum / denom

    print(f"\n=== {title} ===")
    print(
        "leg | contact | force_z | force_z_ct | force_share | action_rms | action_abs | "
        "joint_pos_rms | joint_vel_rms | torque_rms | torque_abs"
    )
    for idx, leg in enumerate(LEG_NAMES):
        print(
            f"{leg:>2}  | "
            f"{contact[idx].item():7.3f} | "
            f"{vertical_force[idx].item():7.2f} | "
            f"{vertical_force_when_contact[idx].item():10.2f} | "
            f"{force_share[idx].item():11.3f} | "
            f"{action_rms[idx].item():10.3f} | "
            f"{action_abs[idx].item():10.3f} | "
            f"{joint_pos_rms[idx].item():13.3f} | "
            f"{joint_vel_rms[idx].item():13.3f} | "
            f"{torque_rms[idx].item():10.3f} | "
            f"{torque_abs[idx].item():10.3f}"
        )
    print("leg | fwd_force | fwd_pos | fwd_neg | horiz_force | slip_ct | joint_pwr_pos | joint_pwr_abs")
    for idx, leg in enumerate(LEG_NAMES):
        print(
            f"{leg:>2}  | "
            f"{forward_force[idx].item():9.2f} | "
            f"{forward_force_positive[idx].item():7.2f} | "
            f"{forward_force_negative[idx].item():7.2f} | "
            f"{horizontal_force[idx].item():11.2f} | "
            f"{foot_slip_when_contact[idx].item():7.3f} | "
            f"{joint_power_positive[idx].item():13.3f} | "
            f"{joint_power_abs[idx].item():13.3f}"
        )
    print("joint | action_mean | action_rms | action_abs | pos_mean | pos_rms | vel_mean | vel_rms | torque_mean | torque_rms | torque_abs | pwr_pos | pwr_abs")
    for leg_idx, leg in enumerate(LEG_NAMES):
        for joint_idx, joint in enumerate(("hip", "thigh", "calf")):
            print(
                f"{leg}_{joint:<5} | "
                f"{joint_action_mean[leg_idx, joint_idx].item():11.3f} | "
                f"{joint_action_rms[leg_idx, joint_idx].item():10.3f} | "
                f"{joint_action_abs[leg_idx, joint_idx].item():10.3f} | "
                f"{joint_pos_offset_mean[leg_idx, joint_idx].item():8.3f} | "
                f"{joint_pos_offset_rms[leg_idx, joint_idx].item():7.3f} | "
                f"{joint_vel_mean[leg_idx, joint_idx].item():8.3f} | "
                f"{joint_vel_detail_rms[leg_idx, joint_idx].item():7.3f} | "
                f"{joint_torque_mean[leg_idx, joint_idx].item():11.3f} | "
                f"{joint_torque_rms[leg_idx, joint_idx].item():10.3f} | "
                f"{joint_torque_abs[leg_idx, joint_idx].item():10.3f} | "
                f"{joint_power_detail_positive[leg_idx, joint_idx].item():7.3f} | "
                f"{joint_power_detail_abs[leg_idx, joint_idx].item():7.3f}"
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
