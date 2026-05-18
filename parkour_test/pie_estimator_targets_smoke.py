from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test PIE estimator target observations.")
parser.add_argument("--task", type=str, default="Isaac-PIE-Parkour-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _assert_target_shapes(targets: dict, num_envs: int, device: str):
    expected_shapes = {
        "base_velocity": (num_envs, 3),
        "foot_clearance": (num_envs, 4),
        "height_scan": (num_envs, 132),
        "next_proprioception": (num_envs, 45),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in targets:
            raise AssertionError(f"Missing estimator target: {key}")
        target = targets[key]
        actual_shape = tuple(target.shape)
        if actual_shape != expected_shape:
            raise AssertionError(f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}")
        if target.device.type not in device:
            raise AssertionError(f"{key} device mismatch: expected env device {device}, got {target.device}")
        if not target.isfinite().all():
            raise AssertionError(f"{key} contains non-finite values")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "parkour_tasks"))

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    try:
        import parkour_tasks.extreme_parkour_task.config.go2  # noqa: F401
    except ModuleNotFoundError:
        import parkour_tasks.parkour_tasks.extreme_parkour_task.config.go2  # noqa: F401

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        obs_dict, _ = env.reset()
        _assert_target_shapes(obs_dict["estimator_targets"], env.unwrapped.num_envs, env.unwrapped.device)

        actions = torch.zeros(
            env.unwrapped.num_envs,
            env.unwrapped.action_manager.total_action_dim,
            device=env.unwrapped.device,
        )
        obs_dict, _, _, _, _ = env.step(actions)
        _assert_target_shapes(obs_dict["estimator_targets"], env.unwrapped.num_envs, env.unwrapped.device)

        print("PIE estimator target smoke test passed.")
        for key, target in obs_dict["estimator_targets"].items():
            print(f"{key} shape: {tuple(target.shape)} device: {target.device}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        simulation_app.close()
