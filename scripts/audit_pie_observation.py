"""Audit PIE proprioceptive observation layout and values."""

from __future__ import annotations

import argparse
import importlib.util
import math
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


parser = argparse.ArgumentParser(description="Audit PIE 45-dim proprioception observation.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--steps", type=int, default=240, help="Steps to summarize.")
parser.add_argument("--warmup_steps", type=int, default=40, help="Warmup steps before recording.")
parser.add_argument("--amplitude", type=float, default=0.35, help="Scripted action amplitude.")
parser.add_argument("--frequency", type=float, default=0.7, help="Scripted action frequency in Hz.")
parser.add_argument("--clip_actions_override", type=float, default=None, help="Override wrapper clip_actions.")
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
from isaaclab_tasks.utils import parse_env_cfg
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import parkour_tasks  # noqa: F401
import parkour_tasks.extreme_parkour_task.config.go2  # noqa: F401


SLICES = {
    "ang_vel_scaled": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "command": slice(6, 9),
    "joint_pos_offset": slice(9, 21),
    "joint_vel_scaled": slice(21, 33),
    "previous_action": slice(33, 45),
}


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.clip_actions_override is not None:
        agent_cfg.clip_actions = args_cli.clip_actions_override

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    asset = env.unwrapped.scene["robot"]
    action_joint_names = list(getattr(action_term, "_joint_names", ()))
    print("=== OBSERVATION CONFIG ===")
    print(f"task={args_cli.task}")
    print(f"num_envs={env.num_envs}, num_actions={env.num_actions}, clip_actions={agent_cfg.clip_actions}")
    print(f"policy_dim={env.num_obs}, critic_dim={env.num_privileged_obs}")
    print(f"action_use_delay={getattr(action_term, '_use_delay', None)}")
    print(f"action_joint_names={action_joint_names}")
    print("slice layout:")
    for name, slc in SLICES.items():
        print(f"  {name}: [{slc.start}:{slc.stop}]")

    obs, extras = env.reset()
    sums = {name: torch.zeros((), device=env.device) for name in SLICES}
    maxes = {name: torch.zeros((), device=env.device) for name in SLICES}
    checks = {
        "policy_vs_history_latest": torch.zeros((), device=env.device),
        "policy_vs_target_next_proprio": torch.zeros((), device=env.device),
        "policy_vs_critic_prefix": torch.zeros((), device=env.device),
        "critic_base_velocity": torch.zeros((), device=env.device),
        "critic_height_scan": torch.zeros((), device=env.device),
        "previous_action_vs_history_last": torch.zeros((), device=env.device),
        "previous_action_vs_raw_delayed": torch.zeros((), device=env.device),
    }
    max_checks = {key: torch.zeros((), device=env.device) for key in checks}
    sample_count = 0

    for step in range(args_cli.steps + args_cli.warmup_steps):
        actions = scripted_actions(env, step)
        obs, _, _, extras = env.step(actions)
        if step < args_cli.warmup_steps:
            continue

        obs_dict = extras["observations"]
        policy_obs = obs_dict["policy"]
        expected = expected_policy_obs(env)
        for name, slc in SLICES.items():
            diff = policy_obs[:, slc] - expected[:, slc]
            sums[name] += torch.square(diff).mean()
            maxes[name] = torch.maximum(maxes[name], diff.abs().max())

        history = obs_dict["proprioception_history"]
        targets = obs_dict["estimator_targets"]
        critic = obs_dict["critic"]
        height_scan = targets["height_scan"]
        base_velocity = targets["base_velocity"]
        raw_action = action_term.raw_actions
        history_last = action_term.action_history_buf[:, -1]

        check_tensors = {
            "policy_vs_history_latest": policy_obs - history[:, -1, :],
            "policy_vs_target_next_proprio": policy_obs - targets["next_proprioception"],
            "policy_vs_critic_prefix": policy_obs - critic[:, :45],
            "critic_base_velocity": base_velocity - critic[:, 45:48],
            "critic_height_scan": height_scan - critic[:, 48:180],
            "previous_action_vs_history_last": policy_obs[:, SLICES["previous_action"]] - history_last,
            "previous_action_vs_raw_delayed": policy_obs[:, SLICES["previous_action"]] - raw_action,
        }
        for key, diff in check_tensors.items():
            checks[key] += torch.square(diff).mean()
            max_checks[key] = torch.maximum(max_checks[key], diff.abs().max())
        sample_count += 1

    print_summary(sample_count, sums, maxes, checks, max_checks)
    env.close()


def scripted_actions(env: ParkourRslRlVecEnvWrapper, step: int) -> torch.Tensor:
    dt = env.unwrapped.step_dt
    phase = math.sin(2.0 * math.pi * args_cli.frequency * step * dt)
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    # Symmetric swing-like pattern to make previous_action and joint velocity slices non-zero.
    actions[:, 4:8] = args_cli.amplitude * phase
    actions[:, 8:12] = -args_cli.amplitude * phase
    actions[:, 0] = 0.25 * args_cli.amplitude * phase
    actions[:, 1] = -0.25 * args_cli.amplitude * phase
    actions[:, 2] = 0.25 * args_cli.amplitude * phase
    actions[:, 3] = -0.25 * args_cli.amplitude * phase
    return actions


def expected_policy_obs(env: ParkourRslRlVecEnvWrapper) -> torch.Tensor:
    asset = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    command = env.unwrapped.command_manager.get_command("base_velocity")
    return torch.cat(
        (
            asset.data.root_ang_vel_b * 0.25,
            asset.data.projected_gravity_b,
            command,
            asset.data.joint_pos - asset.data.default_joint_pos,
            asset.data.joint_vel * 0.05,
            action_term.action_history_buf[:, -1],
        ),
        dim=-1,
    )


def print_summary(
    sample_count: int,
    sums: dict[str, torch.Tensor],
    maxes: dict[str, torch.Tensor],
    checks: dict[str, torch.Tensor],
    max_checks: dict[str, torch.Tensor],
) -> None:
    denom = max(sample_count, 1)
    print("\n=== PROPRIO SLICE CHECKS ===")
    print(f"samples={sample_count}")
    print("slice | rms | max_abs")
    for name in SLICES:
        print(f"{name:<24} | {torch.sqrt(sums[name] / denom).item():.8f} | {maxes[name].item():.8f}")

    print("\n=== GROUP CONSISTENCY CHECKS ===")
    print("check | rms | max_abs")
    for key in checks:
        print(f"{key:<32} | {torch.sqrt(checks[key] / denom).item():.8f} | {max_checks[key].item():.8f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
