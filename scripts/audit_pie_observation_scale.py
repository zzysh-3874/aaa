"""Audit PIE proprioception/history and estimator target scales."""

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

parser = argparse.ArgumentParser(description="Audit PIE observation scale.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--steps", type=int, default=400, help="Steps to summarize after warmup.")
parser.add_argument("--warmup_steps", type=int, default=80, help="Warmup steps before recording.")
parser.add_argument("--policy_action_limit", type=float, default=None, help="Override actor action_limit before loading.")
parser.add_argument("--clip_actions_override", type=float, default=None, help="Override wrapper clip_actions.")
parser.add_argument("--mode", choices=("zero", "random", "policy"), default="zero", help="Action source.")
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
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import parkour_tasks  # noqa: F401
import parkour_tasks.extreme_parkour_task.config.go2  # noqa: F401


GROUPS = (
    ("ang_vel", slice(0, 3)),
    ("gravity", slice(3, 6)),
    ("command", slice(6, 9)),
    ("joint_pos_delta", slice(9, 21)),
    ("joint_vel", slice(21, 33)),
    ("previous_action", slice(33, 45)),
)


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum: torch.Tensor | None = None
        self.sum_sq: torch.Tensor | None = None
        self.max_abs: torch.Tensor | None = None

    def add(self, value: torch.Tensor) -> None:
        value = value.detach().float().reshape(-1, value.shape[-1]).cpu()
        if self.sum is None:
            self.sum = torch.zeros(value.shape[-1])
            self.sum_sq = torch.zeros(value.shape[-1])
            self.max_abs = torch.zeros(value.shape[-1])
        self.count += value.shape[0]
        self.sum += value.sum(dim=0)
        self.sum_sq += torch.square(value).sum(dim=0)
        self.max_abs = torch.maximum(self.max_abs, value.abs().max(dim=0).values)

    def mean(self) -> torch.Tensor:
        return self.sum / max(self.count, 1)

    def rms(self) -> torch.Tensor:
        return torch.sqrt(self.sum_sq / max(self.count, 1))

    def max_abs_values(self) -> torch.Tensor:
        return self.max_abs


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.policy_action_limit is not None:
        agent_cfg.policy.action_limit = args_cli.policy_action_limit
    if args_cli.clip_actions_override is not None:
        agent_cfg.clip_actions = args_cli.clip_actions_override

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = None
    if args_cli.mode == "policy":
        if args_cli.checkpoint is None:
            raise RuntimeError("--checkpoint is required for --mode policy")
        runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint, load_optimizer=False)
        runner.alg.policy.eval()
        runner.alg.estimator.eval()
        runner.alg.reset_pie_actor_hidden()

    policy_stats = RunningStats()
    history_latest_stats = RunningStats()
    history_flat_stats = RunningStats()
    target_stats = {
        "base_velocity": RunningStats(),
        "foot_clearance": RunningStats(),
        "height_scan": RunningStats(),
        "next_proprioception": RunningStats(),
    }
    depth_stats = RunningStats()

    obs, extras = env.reset()
    for step in range(args_cli.steps + args_cli.warmup_steps):
        obs_dict = extras["observations"]
        if args_cli.mode == "zero":
            action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        elif args_cli.mode == "random":
            action = torch.empty(env.num_envs, env.num_actions, device=env.device).uniform_(-0.5, 0.5)
        else:
            actor_obs = runner.alg.build_pie_actor_observations(obs, obs_dict)
            action = runner.alg.policy.act_inference(actor_obs, hist_encoding=True)
        next_obs, _, dones, next_extras = env.step(action)
        if runner is not None:
            runner.alg.reset_pie_actor_hidden(dones)

        if step >= args_cli.warmup_steps:
            policy = obs_dict["policy"]
            history = unwrap_single(obs_dict["proprioception_history"])
            depth = unwrap_single(obs_dict["depth_camera"])
            targets = obs_dict["estimator_targets"]
            policy_stats.add(policy)
            history_latest_stats.add(history[:, -1])
            history_flat_stats.add(history.reshape(-1, history.shape[-1]))
            depth_stats.add(depth.reshape(depth.shape[0], -1))
            for key, stats in target_stats.items():
                stats.add(targets[key])

        obs = next_obs
        extras = next_extras

    print(f"task={args_cli.task}")
    print(f"mode={args_cli.mode}, checkpoint={args_cli.checkpoint}")
    print(f"num_envs={env.num_envs}, samples={policy_stats.count}")
    print(f"clip_actions={agent_cfg.clip_actions}, policy_action_limit={getattr(agent_cfg.policy, 'action_limit', None)}")
    print_group_table("POLICY PROPRIO", policy_stats)
    print_group_table("HISTORY LATEST", history_latest_stats)
    print_group_table("HISTORY ALL FRAMES", history_flat_stats)
    print_stats("DEPTH FLAT", depth_stats)
    print("\n=== ESTIMATOR TARGET SCALE ===")
    for key, stats in target_stats.items():
        print_stats(key, stats)
    env.close()


def unwrap_single(value):
    if isinstance(value, torch.Tensor):
        return value
    if len(value) != 1:
        raise ValueError(f"Expected single-term mapping, got keys {tuple(value.keys())}")
    return next(iter(value.values()))


def print_group_table(title: str, stats: RunningStats) -> None:
    print(f"\n=== {title} GROUP SCALE ===")
    print("group           | dims | mean_abs | rms_mean | rms_min | rms_max | max_abs")
    for name, group_slice in GROUPS:
        mean = stats.mean()[group_slice]
        rms = stats.rms()[group_slice]
        max_abs = stats.max_abs_values()[group_slice]
        print(
            f"{name:<15} | "
            f"{rms.numel():>4} | "
            f"{mean.abs().mean().item():8.5f} | "
            f"{rms.mean().item():8.5f} | "
            f"{rms.min().item():7.5f} | "
            f"{rms.max().item():7.5f} | "
            f"{max_abs.max().item():7.5f}"
        )


def print_stats(name: str, stats: RunningStats) -> None:
    mean = stats.mean()
    rms = stats.rms()
    max_abs = stats.max_abs_values()
    print(
        f"{name:<22} dims={rms.numel():>5} "
        f"mean_abs={mean.abs().mean().item():.5f} "
        f"rms_mean={rms.mean().item():.5f} "
        f"rms_min={rms.min().item():.5f} "
        f"rms_max={rms.max().item():.5f} "
        f"max_abs={max_abs.max().item():.5f}"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
