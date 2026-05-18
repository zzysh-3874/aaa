from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test PIEEstimator forward with PIE environment observations.")
parser.add_argument("--task", type=str, default="Isaac-PIE-Parkour-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _load_pie_estimator_class():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts/rsl_rl/modules/feature_extractors/pie_estimator.py"
    spec = importlib.util.spec_from_file_location("pie_estimator", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PIEEstimator


def _assert_shapes(outputs, num_envs: int, gru_hidden_dim: int):
    expected_shapes = {
        "v_hat": (num_envs, 3),
        "h_f_hat": (num_envs, 4),
        "z_m": (num_envs, 32),
        "z": (num_envs, 32),
        "z_t": (num_envs, 32),
        "z_mu": (num_envs, 32),
        "z_logvar": (num_envs, 32),
        "height_hat": (num_envs, 132),
        "next_proprio_hat": (num_envs, 45),
        "rnn_hidden": (1, num_envs, gru_hidden_dim),
    }
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(outputs[key].shape)
        if actual_shape != expected_shape:
            raise AssertionError(f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}")


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
        critic = obs_dict["critic"]
        depth = obs_dict["depth_camera"]
        proprioception_history = obs_dict["proprioception_history"]
        if tuple(critic.shape) != (env.unwrapped.num_envs, 180):
            raise AssertionError(f"critic shape mismatch: expected {(env.unwrapped.num_envs, 180)}, got {tuple(critic.shape)}")

        pie_estimator_cls = _load_pie_estimator_class()
        estimator = pie_estimator_cls().to(env.unwrapped.device).eval()
        with torch.no_grad():
            outputs = estimator(depth, proprioception_history)
        _assert_shapes(outputs, env.unwrapped.num_envs, estimator.gru_hidden_dim)

        print("PIEEstimator env forward smoke test passed.")
        print(f"critic shape: {tuple(critic.shape)}")
        print(f"depth_camera shape: {tuple(depth.shape)}")
        print(f"proprioception_history shape: {tuple(proprioception_history.shape)}")
        for key in ("v_hat", "h_f_hat", "z_m", "z", "z_t", "z_mu", "z_logvar", "height_hat", "next_proprio_hat"):
            print(f"{key} shape: {tuple(outputs[key].shape)}")
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
