"""Diagnose why a checkpoint resets immediately under deterministic play.

Runs a short deterministic rollout (noise=0, like play) and, on the first
resets, reports which termination cutoff fired (roll / pitch / height / goal)
and the actual roll/pitch/height values, so we can tell whether the policy is
falling, pitching, or being killed by the height floor.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import sys

from isaaclab.app import AppLauncher

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != _SCRIPT_DIR]
sys.path.insert(0, str(_REPO_ROOT))

_CLI = _SCRIPT_DIR / "rsl_rl" / "cli_args.py"
_spec = importlib.util.spec_from_file_location("parkour_rsl_cli_args", _CLI)
cli_args = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(cli_args)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
import isaaclab_tasks  # noqa
import parkour_tasks  # noqa
import parkour_tasks.extreme_parkour_task.config.go2  # noqa


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_pie_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    obs, extras = env.get_observations()
    print("=== initial state (right after reset, before any action) ===")
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_state_w[:, 3:7])
    z = robot.data.root_state_w[:, 2]
    print(f"  roll  abs mean/max: {wrap_to_pi(roll).abs().mean():.3f} / {wrap_to_pi(roll).abs().max():.3f}")
    print(f"  pitch abs mean/max: {wrap_to_pi(pitch).abs().mean():.3f} / {wrap_to_pi(pitch).abs().max():.3f}")
    print(f"  root_z mean/min:    {z.mean():.3f} / {z.min():.3f}")

    ep_len = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    for step in range(args_cli.steps):
        with torch.no_grad():
            actions = policy(extras["observations"], hist_encoding=True)
        obs, _, dones, extras = env.step(actions)
        ep_len += 1
        done_idx = dones.nonzero(as_tuple=False).flatten()
        if done_idx.numel() > 0:
            roll, pitch, _ = euler_xyz_from_quat(robot.data.root_state_w[:, 3:7])
            z = robot.data.root_state_w[:, 2]
            for i in done_idx.tolist():
                r = wrap_to_pi(roll[i]).abs().item()
                p = wrap_to_pi(pitch[i]).abs().item()
                zz = z[i].item()
                cause = []
                if r > 0.7: cause.append(f"ROLL({r:.2f})")
                if p > 0.7: cause.append(f"PITCH({p:.2f})")
                if zz < 0.22: cause.append(f"HEIGHT({zz:.2f})")
                if not cause: cause.append("goal/timeout/other")
                print(f"  env{i} reset @step{int(ep_len[i])}: {' '.join(cause)} (roll={r:.2f} pitch={p:.2f} z={zz:.2f})")
            ep_len[done_idx] = 0
        if step > 30 and (ep_len < 1).all():
            pass
    print("\n=== summary ===")
    print(f"mean steps before first reset (lower=worse): see per-env above")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
