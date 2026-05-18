from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test PIE RSL-RL runner wiring.")
parser.add_argument("--task", type=str, default="Isaac-PIE-Parkour-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=4)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "parkour_tasks"))
    sys.path.insert(0, str(repo_root / "scripts" / "rsl_rl"))

    import gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
    from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

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
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.num_steps_per_env = args_cli.num_steps
    agent_cfg.algorithm.num_learning_epochs = 1
    agent_cfg.algorithm.num_mini_batches = 1
    agent_cfg.estimator.pie_num_learning_epochs = 1
    agent_cfg.estimator.pie_num_mini_batches = 1
    agent_cfg.max_iterations = 1
    agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped_env = ParkourRslRlVecEnvWrapper(env)
    try:
        runner = OnPolicyRunnerWithExtractor(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        if runner.privileged_obs_type != "critic":
            raise AssertionError(f"Expected PIE runner to use critic observations, got {runner.privileged_obs_type}")
        if wrapped_env.num_privileged_obs != 180:
            raise AssertionError(f"Expected 180 privileged critic observations, got {wrapped_env.num_privileged_obs}")
        if not runner.alg.uses_pie_estimator:
            raise AssertionError("PIE runner did not enable PIE estimator auxiliary path")
        if runner.alg.pie_estimator_storage is None:
            raise AssertionError("PIE estimator storage was not initialized")

        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

        print("PIE RSL-RL runner smoke test passed.")
        print(f"policy class: {runner.alg.policy.__class__.__name__}")
        print(f"estimator class: {runner.alg.estimator.__class__.__name__}")
        print(f"uses_pie_estimator: {runner.alg.uses_pie_estimator}")
        print(f"privileged_obs_type: {runner.privileged_obs_type}")
        print(f"num_privileged_obs: {wrapped_env.num_privileged_obs}")
    finally:
        wrapped_env.close()


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
