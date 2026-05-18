from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test PIE estimator rollout storage in PIE env.")
parser.add_argument("--task", type=str, default="Isaac-PIE-Parkour-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--num_steps", type=int, default=4)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _assert_storage_shapes(storage, num_steps: int, num_envs: int):
    depth, proprioception_history, targets, dones = storage.get()
    expected_shapes = {
        "depth": (num_steps, num_envs, 2, 58, 87),
        "proprioception_history": (num_steps, num_envs, 10, 45),
        "base_velocity": (num_steps, num_envs, 3),
        "foot_clearance": (num_steps, num_envs, 4),
        "height_scan": (num_steps, num_envs, 132),
        "next_proprioception": (num_steps, num_envs, 45),
        "next_proprioception_mask": (num_steps, num_envs, 1),
        "dones": (num_steps, num_envs, 1),
    }
    actual_shapes = {
        "depth": tuple(depth.shape),
        "proprioception_history": tuple(proprioception_history.shape),
        "dones": tuple(dones.shape),
    }
    actual_shapes.update({key: tuple(value.shape) for key, value in targets.items()})
    for key, expected_shape in expected_shapes.items():
        actual_shape = actual_shapes[key]
        if actual_shape != expected_shape:
            raise AssertionError(f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}")

    for key, value in targets.items():
        if not value.isfinite().all():
            raise AssertionError(f"{key} contains non-finite values")
    expected_next_mask = (~dones).to(dtype=targets["next_proprioception_mask"].dtype)
    if not targets["next_proprioception_mask"].eq(expected_next_mask).all():
        raise AssertionError("next_proprioception_mask must equal ~dones")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "parkour_tasks"))

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from scripts.rsl_rl.modules.feature_extractors.pie_estimator import PIEEstimator
    from scripts.rsl_rl.modules.feature_extractors.pie_estimator_loss import PIEEstimatorLoss
    from scripts.rsl_rl.modules.feature_extractors.pie_estimator_rollout_storage import PIEEstimatorRolloutStorage

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
        storage = PIEEstimatorRolloutStorage(
            num_envs=env.unwrapped.num_envs,
            num_transitions_per_env=args_cli.num_steps,
            device=env.unwrapped.device,
        )

        for _ in range(args_cli.num_steps):
            cached_step = storage.cache_step_input(obs_dict)
            actions = torch.zeros(
                env.unwrapped.num_envs,
                env.unwrapped.action_manager.total_action_dim,
                device=env.unwrapped.device,
            )
            next_obs_dict, _, terminated, truncated, _ = env.step(actions)
            storage.add_transition(cached_step, next_obs_dict, terminated=terminated, truncated=truncated)
            obs_dict = next_obs_dict

        _assert_storage_shapes(storage, args_cli.num_steps, env.unwrapped.num_envs)

        depth, proprioception_history, targets, _ = storage.flatten()
        estimator = PIEEstimator().to(env.unwrapped.device).train()
        loss_fn = PIEEstimatorLoss(weights={"kl": 0.0})
        outputs = estimator(depth, proprioception_history)
        losses = loss_fn(outputs, targets)
        losses["loss"].backward()

        grad_norm = 0.0
        for param in estimator.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.detach().norm().cpu())
        if grad_norm <= 0.0:
            raise AssertionError("Estimator backward produced zero gradient norm from rollout storage batch")

        print("PIE estimator rollout storage smoke test passed.")
        print(f"stored_steps: {storage.step}")
        print(f"flat_depth shape: {tuple(depth.shape)}")
        print(f"flat_proprioception_history shape: {tuple(proprioception_history.shape)}")
        for key, value in targets.items():
            print(f"{key} shape: {tuple(value.shape)}")
        print(f"loss: {float(losses['loss'].detach().cpu()):.6f}")
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
