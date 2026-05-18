from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test PIEEstimator forward + target loss backward in PIE env.")
parser.add_argument("--task", type=str, default="Isaac-PIE-Parkour-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_losses(losses: dict):
    expected_keys = {"loss", "loss_v", "loss_hf", "loss_height", "loss_next_proprio", "loss_kl"}
    if set(losses.keys()) != expected_keys:
        raise AssertionError(f"Loss keys mismatch: expected {expected_keys}, got {set(losses.keys())}")
    for key, value in losses.items():
        if value.ndim != 0:
            raise AssertionError(f"{key} must be scalar, got shape {tuple(value.shape)}")
        if not value.isfinite():
            raise AssertionError(f"{key} is not finite: {value}")


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
        pie_estimator_module = _load_module(
            "pie_estimator",
            "scripts/rsl_rl/modules/feature_extractors/pie_estimator.py",
        )
        loss_module = _load_module(
            "pie_estimator_loss",
            "scripts/rsl_rl/modules/feature_extractors/pie_estimator_loss.py",
        )
        estimator = pie_estimator_module.PIEEstimator().to(env.unwrapped.device).train()
        loss_fn = loss_module.PIEEstimatorLoss(weights={"kl": 0.01})

        current_targets = loss_module.cache_pie_estimator_targets(obs_dict["estimator_targets"])
        outputs = estimator(obs_dict["depth_camera"], obs_dict["proprioception_history"])
        actions = torch.zeros(
            env.unwrapped.num_envs,
            env.unwrapped.action_manager.total_action_dim,
            device=env.unwrapped.device,
        )
        next_obs_dict, _, terminated, truncated, _ = env.step(actions)
        transition_targets = loss_module.build_pie_transition_targets(
            current_targets,
            next_obs_dict,
            terminated=terminated,
            truncated=truncated,
        )

        losses = loss_fn(outputs, transition_targets)
        _assert_losses(losses)
        losses["loss"].backward()

        grad_norm = 0.0
        for param in estimator.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.detach().norm().cpu())
        if grad_norm <= 0.0:
            raise AssertionError("Estimator backward produced zero gradient norm")

        print("PIEEstimator env loss smoke test passed.")
        for key, value in losses.items():
            print(f"{key}: {float(value.detach().cpu()):.6f}")
        print(f"grad_norm: {grad_norm:.6f}")
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
