"""Audit PIE action-to-joint mapping with scripted single-joint actions."""

from __future__ import annotations

import argparse
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


parser = argparse.ArgumentParser(description="Audit single action dimension mapping for PIE tasks.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--settle_steps", type=int, default=40, help="Zero-action settling steps before each probe.")
parser.add_argument("--hold_steps", type=int, default=40, help="Steps to hold each single-joint action.")
parser.add_argument("--amplitude", type=float, default=0.25, help="Raw action amplitude for +/- probes.")
parser.add_argument("--joints", type=str, default=".*_hip_joint", help="Regex-like substring filter for joints to probe.")
parser.add_argument("--disable_delay", action="store_true", default=True, help="Disable action delay for mapping audit.")
parser.add_argument("--keep_delay", action="store_false", dest="disable_delay", help="Keep configured action delay.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils import parse_env_cfg
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import parkour_tasks  # noqa: F401
import parkour_tasks.extreme_parkour_task.config.go2  # noqa: F401


LEG_NAMES = ("FL", "FR", "RL", "RR")
FOOT_NAMES = tuple(f"{leg}_foot" for leg in LEG_NAMES)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.disable_delay:
        env_cfg.actions.joint_pos.use_delay = False
        env_cfg.actions.joint_pos.action_delay_steps = [0]
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    asset = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    action_joint_names = list(getattr(action_term, "_joint_names", ()))
    action_joint_ids = getattr(action_term, "_joint_ids", None)
    scale = getattr(action_term, "_scale", None)
    offset = getattr(action_term, "_offset", None)

    print("=== ACTION TERM ===")
    print(f"task={args_cli.task}")
    print(f"num_envs={env.num_envs}, num_actions={env.num_actions}, clip_actions={agent_cfg.clip_actions}")
    print(f"delay_disabled={args_cli.disable_delay}")
    print(f"action_joint_ids={action_joint_ids}")
    print("idx | action_joint | asset_joint_id | asset_joint_name | scale_mean | default_offset_mean")
    for idx, joint_name in enumerate(action_joint_names):
        asset_ids, asset_names = asset.find_joints([joint_name], preserve_order=True)
        scale_value = _index_action_value(scale, idx)
        offset_value = _index_action_value(offset, idx)
        asset_id = asset_ids[0] if asset_ids else None
        asset_name = asset_names[0] if asset_names else None
        print(f"{idx:>3} | {joint_name:<16} | {str(asset_id):>14} | {str(asset_name):<16} | {scale_value:10.3f} | {offset_value:19.3f}")

    probe_indices = [
        idx for idx, joint_name in enumerate(action_joint_names) if _matches_filter(joint_name, args_cli.joints)
    ]
    if not probe_indices:
        raise RuntimeError(f"No joints matched --joints={args_cli.joints!r}")

    foot_ids, foot_names = asset.find_bodies(FOOT_NAMES, preserve_order=True)
    print("\nResolved feet:", list(zip(LEG_NAMES, foot_ids, foot_names)))
    print("\n=== SINGLE-JOINT RESPONSE ===")
    print(
        "idx | joint | sign | target_delta | joint_delta | leak_max | foot | "
        "foot_dx | foot_dy | foot_dz"
    )
    for idx in probe_indices:
        for sign in (1.0, -1.0):
            result = run_single_probe(env, idx, sign * args_cli.amplitude, foot_ids)
            leg = _leg_from_joint(action_joint_names[idx])
            foot_idx = LEG_NAMES.index(leg)
            print(
                f"{idx:>3} | {action_joint_names[idx]:<16} | {sign:+.0f} | "
                f"{result['target_delta']:12.4f} | {result['joint_delta']:11.4f} | "
                f"{result['leak_max']:8.4f} | {leg:>2} | "
                f"{result['foot_delta'][foot_idx, 0].item():7.4f} | "
                f"{result['foot_delta'][foot_idx, 1].item():7.4f} | "
                f"{result['foot_delta'][foot_idx, 2].item():7.4f}"
            )

    env.close()


def _matches_filter(joint_name: str, pattern: str) -> bool:
    if pattern == ".*":
        return True
    if pattern.startswith(".*"):
        return joint_name.endswith(pattern[2:])
    if pattern.endswith(".*"):
        return joint_name.startswith(pattern[:-2])
    return pattern in joint_name


def _index_action_value(value, idx: int) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float | int):
        return float(value)
    if hasattr(value, "ndim"):
        if value.ndim == 0:
            return value.item()
        if value.ndim == 1:
            return value[idx].item()
        return value[:, idx].mean().item()
    return float("nan")


def _leg_from_joint(joint_name: str) -> str:
    for leg in LEG_NAMES:
        if joint_name.startswith(f"{leg}_"):
            return leg
    raise RuntimeError(f"Unable to resolve leg from joint name: {joint_name}")


def run_single_probe(env: ParkourRslRlVecEnvWrapper, action_idx: int, action_value: float, foot_ids: list[int]) -> dict:
    env.reset()
    zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(args_cli.settle_steps):
        env.step(zero)

    asset = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    foot_before = foot_pos_in_base(asset, foot_ids)
    joint_before = asset.data.joint_pos.clone()
    target_before = action_term.processed_actions.clone()

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    actions[:, action_idx] = action_value
    for _ in range(args_cli.hold_steps):
        env.step(actions)

    foot_after = foot_pos_in_base(asset, foot_ids)
    joint_after = asset.data.joint_pos.clone()
    target_after = action_term.processed_actions.clone()

    joint_delta_all = joint_after - joint_before
    target_delta_all = target_after - target_before
    mask = torch.ones(env.num_actions, dtype=torch.bool, device=env.device)
    mask[action_idx] = False
    return {
        "target_delta": target_delta_all[:, action_idx].mean().item(),
        "joint_delta": joint_delta_all[:, action_idx].mean().item(),
        "leak_max": joint_delta_all[:, mask].abs().mean(dim=0).max().item(),
        "foot_delta": (foot_after - foot_before).mean(dim=0),
    }


def foot_pos_in_base(asset, foot_ids: list[int]) -> torch.Tensor:
    foot_ids_tensor = torch.tensor(foot_ids, device=asset.data.root_pos_w.device, dtype=torch.long)
    foot_pos_w = asset.data.body_pos_w[:, foot_ids_tensor]
    rel_pos_w = foot_pos_w - asset.data.root_pos_w[:, None, :]
    repeated_quat = asset.data.root_quat_w[:, None, :].expand(-1, len(foot_ids), -1).reshape(-1, 4)
    foot_pos_b = quat_apply_inverse(repeated_quat, rel_pos_w.reshape(-1, 3)).reshape(-1, len(foot_ids), 3)
    return foot_pos_b


if __name__ == "__main__":
    main()
    simulation_app.close()
