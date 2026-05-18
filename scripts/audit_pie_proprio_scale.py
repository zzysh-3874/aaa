"""Audit PIE proprioception/history scale under a checkpoint rollout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


parser = argparse.ArgumentParser(description="Audit PIE proprioception/history scale.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--steps", type=int, default=240, help="Steps to summarize after warmup.")
parser.add_argument("--warmup_steps", type=int, default=80, help="Warmup steps before recording.")
parser.add_argument("--policy_action_limit", type=float, default=None, help="Override actor action_limit before loading.")
parser.add_argument("--clip_actions_override", type=float, default=None, help="Override wrapper clip_actions.")
parser.add_argument("--top_dims", type=int, default=12, help="Number of largest-RMS proprio dims to print.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")
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


SLICES = {
    "ang_vel_scaled": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "command": slice(6, 9),
    "joint_pos_offset": slice(9, 21),
    "joint_vel_scaled": slice(21, 33),
    "previous_action": slice(33, 45),
}


@dataclass
class RunningStats:
    count: int = 0
    sum: float = 0.0
    sq_sum: float = 0.0
    abs_sum: float = 0.0
    max_abs: float = 0.0

    def update(self, value: torch.Tensor) -> None:
        detached = value.detach().float()
        self.count += detached.numel()
        self.sum += detached.sum().item()
        self.sq_sum += detached.square().sum().item()
        self.abs_sum += detached.abs().sum().item()
        self.max_abs = max(self.max_abs, detached.abs().max().item())

    @property
    def mean(self) -> float:
        return self.sum / max(self.count, 1)

    @property
    def rms(self) -> float:
        return (self.sq_sum / max(self.count, 1)) ** 0.5

    @property
    def mean_abs(self) -> float:
        return self.abs_sum / max(self.count, 1)

    @property
    def std(self) -> float:
        variance = self.sq_sum / max(self.count, 1) - self.mean**2
        return max(variance, 0.0) ** 0.5


class ProprioScaleStats:
    def __init__(self, device: torch.device, dim_names: list[str]):
        self.device = device
        self.dim_names = dim_names
        self.slice_stats = {
            source: {name: RunningStats() for name in SLICES}
            for source in ("policy", "history_all", "history_latest", "next_proprio_target")
        }
        self.source_total_sq = {source: 0.0 for source in self.slice_stats}
        self.dim_sq_sum = {source: torch.zeros(45, device=device) for source in self.slice_stats}
        self.dim_count = {source: 0 for source in self.slice_stats}
        self.target_stats = {
            "base_velocity": RunningStats(),
            "foot_clearance": RunningStats(),
            "height_scan": RunningStats(),
        }

    def update_45(self, source: str, value: torch.Tensor) -> None:
        flat = value.reshape(-1, 45).detach().float()
        self.source_total_sq[source] += flat.square().sum().item()
        self.dim_sq_sum[source] += flat.square().sum(dim=0)
        self.dim_count[source] += flat.shape[0]
        for name, slc in SLICES.items():
            self.slice_stats[source][name].update(flat[:, slc])

    def update_target(self, name: str, value: torch.Tensor) -> None:
        self.target_stats[name].update(value)

    def print_summary(self, top_dims: int) -> None:
        print("\n=== 45-DIM PROPRIO SCALE BY SOURCE ===")
        print("source              | slice              | dims | mean_abs | rms      | std      | max_abs  | l2_share")
        for source, source_stats in self.slice_stats.items():
            total_sq = max(self.source_total_sq[source], 1.0e-12)
            for name, slc in SLICES.items():
                stat = source_stats[name]
                l2_share = stat.sq_sum / total_sq
                print(
                    f"{source:<19} | {name:<18} | "
                    f"{slc.stop - slc.start:>4} | {stat.mean_abs:8.5f} | {stat.rms:8.5f} | "
                    f"{stat.std:8.5f} | {stat.max_abs:8.5f} | {l2_share:8.5f}"
                )

        print("\n=== TARGET SCALE ===")
        print("target          | mean_abs | rms      | std      | max_abs")
        for name, stat in self.target_stats.items():
            print(f"{name:<15} | {stat.mean_abs:8.5f} | {stat.rms:8.5f} | {stat.std:8.5f} | {stat.max_abs:8.5f}")

        print("\n=== TOP PROPRIO DIMS BY RMS ===")
        for source in ("policy", "history_all", "next_proprio_target"):
            dim_rms = torch.sqrt(self.dim_sq_sum[source] / max(self.dim_count[source], 1))
            top = torch.topk(dim_rms, k=min(top_dims, dim_rms.numel()))
            print(f"\n{source}:")
            for value, index in zip(top.values.detach().cpu().tolist(), top.indices.detach().cpu().tolist(), strict=True):
                print(f"  {index:02d} {self.dim_names[index]:<32} rms={value:.5f}")


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

    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    runner.alg.policy.eval()
    runner.alg.estimator.eval()

    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    joint_names = list(getattr(action_term, "_joint_names", ()))
    dim_names = build_dim_names(joint_names)
    stats = ProprioScaleStats(env.device, dim_names)

    print("=== PROPRIO SCALE CONFIG ===")
    print(f"task={args_cli.task}")
    print(f"checkpoint={args_cli.checkpoint}")
    print(f"num_envs={env.num_envs}, steps={args_cli.steps}, warmup_steps={args_cli.warmup_steps}")
    print(f"policy_action_limit={getattr(runner.alg.policy, 'action_limit', None)}")
    print(f"wrapper_clip_actions={agent_cfg.clip_actions}")
    print(f"feature_keys={runner.alg.pie_actor_feature_keys}")
    print(f"joint_names={joint_names}")

    obs, extras = env.reset()
    runner.alg.reset_pie_actor_hidden()
    for step in range(args_cli.steps + args_cli.warmup_steps):
        obs_dict = extras["observations"]
        actor_obs = runner.alg.build_pie_actor_observations(obs, obs_dict)
        action = runner.alg.policy.act_inference(actor_obs, hist_encoding=True)
        next_obs, _, dones, next_extras = env.step(action)
        runner.alg.reset_pie_actor_hidden(dones)

        if step >= args_cli.warmup_steps:
            update_stats(stats, obs_dict)

        obs = next_obs
        extras = next_extras

    stats.print_summary(args_cli.top_dims)
    env.close()


def update_stats(stats: ProprioScaleStats, obs_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> None:
    policy = obs_dict["policy"]
    history = obs_dict["proprioception_history"]
    targets = obs_dict["estimator_targets"]
    stats.update_45("policy", policy)
    stats.update_45("history_all", history)
    stats.update_45("history_latest", history[:, -1])
    stats.update_45("next_proprio_target", targets["next_proprioception"])
    stats.update_target("base_velocity", targets["base_velocity"])
    stats.update_target("foot_clearance", targets["foot_clearance"])
    stats.update_target("height_scan", targets["height_scan"])


def build_dim_names(joint_names: list[str]) -> list[str]:
    names = [
        "ang_vel_x_scaled",
        "ang_vel_y_scaled",
        "ang_vel_z_scaled",
        "gravity_x",
        "gravity_y",
        "gravity_z",
        "command_vx",
        "command_vy",
        "command_yaw",
    ]
    if len(joint_names) == 12:
        names.extend(f"joint_pos_offset:{joint}" for joint in joint_names)
        names.extend(f"joint_vel_scaled:{joint}" for joint in joint_names)
        names.extend(f"previous_action:{joint}" for joint in joint_names)
    else:
        names.extend(f"joint_pos_offset_{index}" for index in range(12))
        names.extend(f"joint_vel_scaled_{index}" for index in range(12))
        names.extend(f"previous_action_{index}" for index in range(12))
    return names


if __name__ == "__main__":
    main()
    simulation_app.close()
