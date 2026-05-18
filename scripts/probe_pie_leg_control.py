"""Probe rear-leg control response with scripted actions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
import time

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


parser = argparse.ArgumentParser(description="Probe leg control symmetry with scripted joint-position actions.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--steps", type=int, default=300, help="Scripted rollout steps to summarize.")
parser.add_argument("--warmup_steps", type=int, default=60, help="Steps to skip before recording.")
parser.add_argument("--amplitude", type=float, default=0.35, help="Thigh/calf sinusoid amplitude in action units.")
parser.add_argument("--frequency", type=float, default=1.0, help="Sinusoid frequency in Hz.")
parser.add_argument("--contact_threshold", type=float, default=2.0, help="Contact force norm threshold.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
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
class ProbeStats:
    samples: int
    contact_sum: torch.Tensor
    vertical_force_sum: torch.Tensor
    force_share_sum: torch.Tensor
    vertical_force_when_contact_sum: torch.Tensor
    joint_pos_sq_sum: torch.Tensor
    joint_vel_sq_sum: torch.Tensor
    torque_sq_sum: torch.Tensor
    torque_abs_sum: torch.Tensor
    joint_power_positive_sum: torch.Tensor
    joint_power_abs_sum: torch.Tensor

    @classmethod
    def create(cls, device: torch.device | str) -> "ProbeStats":
        return cls(
            samples=0,
            contact_sum=torch.zeros(4, device=device),
            vertical_force_sum=torch.zeros(4, device=device),
            force_share_sum=torch.zeros(4, device=device),
            vertical_force_when_contact_sum=torch.zeros(4, device=device),
            joint_pos_sq_sum=torch.zeros(4, 3, device=device),
            joint_vel_sq_sum=torch.zeros(4, 3, device=device),
            torque_sq_sum=torch.zeros(4, 3, device=device),
            torque_abs_sum=torch.zeros(4, 3, device=device),
            joint_power_positive_sum=torch.zeros(4, 3, device=device),
            joint_power_abs_sum=torch.zeros(4, 3, device=device),
        )


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    action_joint_names = list(getattr(action_term, "_joint_names", ()))
    print("Action joint names:", action_joint_names)
    print("Probe action: hip=0, thigh=+A*sin(2*pi*f*t), calf=-A*sin(2*pi*f*t)")
    print(f"Amplitude={args_cli.amplitude:.3f}, frequency={args_cli.frequency:.3f} Hz")

    probes = (
        ("ZERO", ()),
        ("RL_ONLY", ("RL",)),
        ("RR_ONLY", ("RR",)),
        ("RL_RR_BOTH", ("RL", "RR")),
    )
    for title, target_legs in probes:
        stats = run_probe(env, action_joint_names, target_legs)
        print_summary(title, stats)

    env.close()


def run_probe(
    env: ParkourRslRlVecEnvWrapper,
    action_joint_names: list[str],
    target_legs: tuple[str, ...],
) -> ProbeStats:
    env.reset()
    stats = ProbeStats.create(env.device)
    dt = env.unwrapped.step_dt
    for step in range(args_cli.steps + args_cli.warmup_steps):
        start_time = time.time()
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        phase = math.sin(2.0 * math.pi * args_cli.frequency * step * dt)
        for leg in target_legs:
            thigh_idx = action_joint_names.index(f"{leg}_thigh_joint")
            calf_idx = action_joint_names.index(f"{leg}_calf_joint")
            actions[:, thigh_idx] = args_cli.amplitude * phase
            actions[:, calf_idx] = -args_cli.amplitude * phase
        _, _, _, _ = env.step(actions)
        if step >= args_cli.warmup_steps:
            accumulate_stats(env, stats)
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    return stats


def accumulate_stats(env: ParkourRslRlVecEnvWrapper, stats: ProbeStats) -> None:
    asset = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
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

    stats.contact_sum += contact.mean(dim=0)
    stats.vertical_force_sum += vertical_force.mean(dim=0)
    stats.force_share_sum += force_share.mean(dim=0)
    stats.vertical_force_when_contact_sum += (vertical_force * contact).sum(dim=0) / contact_count

    joint_pos_error = asset.data.joint_pos - asset.data.default_joint_pos
    joint_vel = asset.data.joint_vel
    torque = getattr(asset.data, "applied_torque", None)
    if torque is None:
        torque = getattr(asset.data, "computed_torque", None)

    for leg_idx, joint_ids in enumerate(joint_ids_by_leg):
        leg_joint_pos_error = joint_pos_error[:, joint_ids]
        leg_joint_vel = joint_vel[:, joint_ids]
        stats.joint_pos_sq_sum[leg_idx] += torch.square(leg_joint_pos_error).mean(dim=0)
        stats.joint_vel_sq_sum[leg_idx] += torch.square(leg_joint_vel).mean(dim=0)
        if torque is not None:
            leg_torque = torque[:, joint_ids]
            stats.torque_sq_sum[leg_idx] += torch.square(leg_torque).mean(dim=0)
            stats.torque_abs_sum[leg_idx] += torch.abs(leg_torque).mean(dim=0)
            joint_power = leg_torque * leg_joint_vel
            stats.joint_power_positive_sum[leg_idx] += torch.clamp(joint_power, min=0.0).mean(dim=0)
            stats.joint_power_abs_sum[leg_idx] += torch.abs(joint_power).mean(dim=0)

    stats.samples += 1


def print_summary(title: str, stats: ProbeStats) -> None:
    denom = max(stats.samples, 1)
    contact = stats.contact_sum / denom
    vertical_force = stats.vertical_force_sum / denom
    force_share = stats.force_share_sum / denom
    vertical_force_when_contact = stats.vertical_force_when_contact_sum / denom
    joint_pos_rms = torch.sqrt(stats.joint_pos_sq_sum / denom)
    joint_vel_rms = torch.sqrt(stats.joint_vel_sq_sum / denom)
    torque_rms = torch.sqrt(stats.torque_sq_sum / denom)
    torque_abs = stats.torque_abs_sum / denom
    joint_power_positive = stats.joint_power_positive_sum / denom
    joint_power_abs = stats.joint_power_abs_sum / denom

    print(f"\n=== {title} ===")
    print("leg | contact | force_z | force_z_ct | force_share | pos_rms_mean | vel_rms_mean | torque_rms_mean | pwr_pos_mean | pwr_abs_mean")
    for idx, leg in enumerate(LEG_NAMES):
        print(
            f"{leg:>2}  | "
            f"{contact[idx].item():7.3f} | "
            f"{vertical_force[idx].item():7.2f} | "
            f"{vertical_force_when_contact[idx].item():10.2f} | "
            f"{force_share[idx].item():11.3f} | "
            f"{joint_pos_rms[idx].mean().item():12.3f} | "
            f"{joint_vel_rms[idx].mean().item():12.3f} | "
            f"{torque_rms[idx].mean().item():15.3f} | "
            f"{joint_power_positive[idx].mean().item():12.3f} | "
            f"{joint_power_abs[idx].mean().item():12.3f}"
        )
    print("joint | pos_rms | vel_rms | torque_rms | torque_abs | pwr_pos | pwr_abs")
    for leg_idx, leg in enumerate(LEG_NAMES):
        for joint_idx, joint in enumerate(("hip", "thigh", "calf")):
            print(
                f"{leg}_{joint:<5} | "
                f"{joint_pos_rms[leg_idx, joint_idx].item():7.3f} | "
                f"{joint_vel_rms[leg_idx, joint_idx].item():7.3f} | "
                f"{torque_rms[leg_idx, joint_idx].item():10.3f} | "
                f"{torque_abs[leg_idx, joint_idx].item():10.3f} | "
                f"{joint_power_positive[leg_idx, joint_idx].item():7.3f} | "
                f"{joint_power_abs[leg_idx, joint_idx].item():7.3f}"
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
