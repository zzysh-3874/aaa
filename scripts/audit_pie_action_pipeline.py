"""Audit PIE policy action processing from actor output to joint targets."""

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


parser = argparse.ArgumentParser(description="Audit PIE action processing pipeline for a checkpoint.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--steps", type=int, default=300, help="Steps to summarize after warmup.")
parser.add_argument("--warmup_steps", type=int, default=80, help="Warmup steps before recording.")
parser.add_argument("--policy_action_limit", type=float, default=None, help="Override actor action_limit before loading.")
parser.add_argument("--clip_actions_override", type=float, default=None, help="Override wrapper clip_actions.")
parser.add_argument("--contact_threshold", type=float, default=2.0, help="Unused; kept for CLI consistency.")
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


PREVIOUS_ACTION_SLICE = slice(33, 45)


@dataclass
class PipelineStats:
    samples: int
    policy_sum: torch.Tensor
    policy_abs_sum: torch.Tensor
    clipped_sum: torch.Tensor
    clipped_abs_sum: torch.Tensor
    raw_sum: torch.Tensor
    raw_abs_sum: torch.Tensor
    processed_delta_sum: torch.Tensor
    joint_offset_sum: torch.Tensor
    target_error_abs_sum: torch.Tensor
    policy_max_abs: torch.Tensor
    clip_count_sum: torch.Tensor
    obs_prev_vs_hist_sq_sum: torch.Tensor
    hist_last_vs_clipped_sq_sum: torch.Tensor
    raw_vs_current_sq_sum: torch.Tensor
    raw_vs_delayed_expected_sq_sum: torch.Tensor
    processed_vs_raw_sq_sum: torch.Tensor
    target_error_sq_sum: torch.Tensor

    @classmethod
    def create(cls, device: torch.device | str, num_actions: int) -> "PipelineStats":
        zeros = torch.zeros(num_actions, device=device)
        return cls(
            samples=0,
            policy_sum=zeros.clone(),
            policy_abs_sum=zeros.clone(),
            clipped_sum=zeros.clone(),
            clipped_abs_sum=zeros.clone(),
            raw_sum=zeros.clone(),
            raw_abs_sum=zeros.clone(),
            processed_delta_sum=zeros.clone(),
            joint_offset_sum=zeros.clone(),
            target_error_abs_sum=zeros.clone(),
            policy_max_abs=zeros.clone(),
            clip_count_sum=zeros.clone(),
            obs_prev_vs_hist_sq_sum=zeros.clone(),
            hist_last_vs_clipped_sq_sum=zeros.clone(),
            raw_vs_current_sq_sum=zeros.clone(),
            raw_vs_delayed_expected_sq_sum=zeros.clone(),
            processed_vs_raw_sq_sum=zeros.clone(),
            target_error_sq_sum=zeros.clone(),
        )


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
    if not getattr(runner.alg, "use_pie_actor_features", False):
        raise RuntimeError("This diagnostic expects PIE actor features.")

    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    asset = env.unwrapped.scene["robot"]
    action_joint_names = list(getattr(action_term, "_joint_names", ()))

    print_config(env, agent_cfg, runner, action_term, action_joint_names)

    obs, extras = env.reset()
    runner.alg.reset_pie_actor_hidden()
    stats = PipelineStats.create(env.device, env.num_actions)
    for step in range(args_cli.steps + args_cli.warmup_steps):
        with torch.inference_mode():
            obs_dict = extras["observations"]
            actor_obs = runner.alg.build_pie_actor_observations(obs, obs_dict)
            policy_action = runner.alg.policy.act_inference(actor_obs, hist_encoding=True)
            clipped_action = (
                torch.clamp(policy_action, -agent_cfg.clip_actions, agent_cfg.clip_actions)
                if agent_cfg.clip_actions is not None
                else policy_action
            )

        hist_before = action_term.action_history_buf[:, -1].clone()
        obs_prev_action = obs[:, PREVIOUS_ACTION_SLICE].clone()
        obs, _, dones, extras = env.step(policy_action)
        runner.alg.reset_pie_actor_hidden(dones)

        if step >= args_cli.warmup_steps:
            accumulate_stats(
                env=env,
                stats=stats,
                asset=asset,
                action_term=action_term,
                policy_action=policy_action,
                clipped_action=clipped_action,
                hist_before=hist_before,
                obs_prev_action=obs_prev_action,
            )

    print_summary(stats, action_joint_names)
    env.close()


def print_config(env, agent_cfg, runner, action_term, action_joint_names: list[str]) -> None:
    scale = getattr(action_term, "_scale", None)
    offset = getattr(action_term, "_offset", None)
    raw_clip = getattr(action_term, "_clip", None)
    print("=== ACTION PIPELINE CONFIG ===")
    print(f"task={args_cli.task}")
    print(f"checkpoint={args_cli.checkpoint}")
    print(f"num_envs={env.num_envs}, num_actions={env.num_actions}")
    print(f"policy_action_limit={getattr(runner.alg.policy, 'action_limit', None)}")
    print(f"wrapper_clip_actions={agent_cfg.clip_actions}")
    print(f"action_use_delay={getattr(action_term, '_use_delay', None)}")
    print(f"action_delay={getattr(action_term, 'delay', None)}")
    print("idx | joint | scale | offset | env_raw_clip")
    for idx, joint_name in enumerate(action_joint_names):
        clip_text = "None"
        if raw_clip is not None:
            clip_text = f"[{raw_clip[0, idx, 0].item():.3f}, {raw_clip[0, idx, 1].item():.3f}]"
        print(
            f"{idx:>3} | {joint_name:<16} | "
            f"{_index_action_value(scale, idx):6.3f} | "
            f"{_index_action_value(offset, idx):7.3f} | {clip_text}"
        )


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


def accumulate_stats(
    env: ParkourRslRlVecEnvWrapper,
    stats: PipelineStats,
    asset,
    action_term,
    policy_action: torch.Tensor,
    clipped_action: torch.Tensor,
    hist_before: torch.Tensor,
    obs_prev_action: torch.Tensor,
) -> None:
    raw_action = action_term.raw_actions.clone()
    hist_after = action_term.action_history_buf.clone()
    processed_delta = action_term.processed_actions - asset.data.default_joint_pos
    joint_offset = asset.data.joint_pos - asset.data.default_joint_pos
    target_error = asset.data.joint_pos - action_term.processed_actions

    delay = getattr(action_term, "delay", torch.tensor(0, device=env.device))
    delay_idx = int(delay.item()) if hasattr(delay, "item") else int(delay)
    delayed_expected = hist_after[:, -1 - delay_idx]

    stats.policy_sum += policy_action.mean(dim=0)
    stats.policy_abs_sum += policy_action.abs().mean(dim=0)
    stats.clipped_sum += clipped_action.mean(dim=0)
    stats.clipped_abs_sum += clipped_action.abs().mean(dim=0)
    stats.raw_sum += raw_action.mean(dim=0)
    stats.raw_abs_sum += raw_action.abs().mean(dim=0)
    stats.processed_delta_sum += processed_delta.mean(dim=0)
    stats.joint_offset_sum += joint_offset.mean(dim=0)
    stats.target_error_abs_sum += target_error.abs().mean(dim=0)
    stats.policy_max_abs = torch.maximum(stats.policy_max_abs, policy_action.abs().max(dim=0).values)
    stats.clip_count_sum += (policy_action != clipped_action).float().mean(dim=0)
    stats.obs_prev_vs_hist_sq_sum += torch.square(obs_prev_action - hist_before).mean(dim=0)
    stats.hist_last_vs_clipped_sq_sum += torch.square(hist_after[:, -1] - clipped_action).mean(dim=0)
    stats.raw_vs_current_sq_sum += torch.square(raw_action - clipped_action).mean(dim=0)
    stats.raw_vs_delayed_expected_sq_sum += torch.square(raw_action - delayed_expected).mean(dim=0)
    stats.processed_vs_raw_sq_sum += torch.square(processed_delta - raw_action).mean(dim=0)
    stats.target_error_sq_sum += torch.square(target_error).mean(dim=0)
    stats.samples += 1


def print_summary(stats: PipelineStats, joint_names: list[str]) -> None:
    denom = max(stats.samples, 1)
    print("\n=== GLOBAL CHECKS ===")
    print(f"samples={stats.samples}")
    print(f"obs_previous_action_vs_history_last_rms={torch.sqrt(stats.obs_prev_vs_hist_sq_sum / denom).mean().item():.6f}")
    print(f"history_last_vs_wrapper_clipped_rms={torch.sqrt(stats.hist_last_vs_clipped_sq_sum / denom).mean().item():.6f}")
    print(f"raw_vs_current_clipped_rms={torch.sqrt(stats.raw_vs_current_sq_sum / denom).mean().item():.6f}")
    print(f"raw_vs_delayed_expected_rms={torch.sqrt(stats.raw_vs_delayed_expected_sq_sum / denom).mean().item():.6f}")
    print(f"processed_delta_vs_raw_rms={torch.sqrt(stats.processed_vs_raw_sq_sum / denom).mean().item():.6f}")
    print(f"joint_pos_vs_processed_target_rms={torch.sqrt(stats.target_error_sq_sum / denom).mean().item():.6f}")

    print("\n=== PER-JOINT ACTION PIPELINE ===")
    print(
        "joint | policy_mean | policy_abs | policy_max | clip_frac | clipped_mean | "
        "raw_mean | target_delta | joint_offset | target_err_abs"
    )
    for idx, joint_name in enumerate(joint_names):
        print(
            f"{joint_name:<16} | "
            f"{(stats.policy_sum[idx] / denom).item():11.4f} | "
            f"{(stats.policy_abs_sum[idx] / denom).item():10.4f} | "
            f"{stats.policy_max_abs[idx].item():10.4f} | "
            f"{(stats.clip_count_sum[idx] / denom).item():9.4f} | "
            f"{(stats.clipped_sum[idx] / denom).item():12.4f} | "
            f"{(stats.raw_sum[idx] / denom).item():8.4f} | "
            f"{(stats.processed_delta_sum[idx] / denom).item():12.4f} | "
            f"{(stats.joint_offset_sum[idx] / denom).item():12.4f} | "
            f"{(stats.target_error_abs_sum[idx] / denom).item():14.4f}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
