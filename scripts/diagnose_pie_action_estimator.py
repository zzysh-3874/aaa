"""Diagnose PIE previous-action alignment and per-foot estimator error."""

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


parser = argparse.ArgumentParser(description="Diagnose PIE action-observation alignment and estimator outputs.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--steps", type=int, default=500, help="Policy rollout steps to summarize.")
parser.add_argument("--warmup_steps", type=int, default=80, help="Steps to skip before recording.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
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


LEG_NAMES = ("FL", "FR", "RL", "RR")
JOINT_NAMES = tuple(
    tuple(f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf")) for leg in LEG_NAMES
)
FOOT_TARGET_NAMES = ("FL", "FR", "RL", "RR")
PREVIOUS_ACTION_SLICE = slice(33, 45)


@dataclass
class AlignmentStats:
    samples: int
    obs_prev_minus_executed_sum: torch.Tensor
    obs_prev_minus_executed_sq_sum: torch.Tensor
    obs_prev_minus_executed_abs_sum: torch.Tensor
    policy_minus_executed_sum: torch.Tensor
    policy_minus_executed_sq_sum: torch.Tensor
    policy_minus_executed_abs_sum: torch.Tensor
    hf_error_sum: torch.Tensor
    hf_error_sq_sum: torch.Tensor
    hf_error_abs_sum: torch.Tensor
    hf_pred_sum: torch.Tensor
    hf_target_sum: torch.Tensor

    @classmethod
    def create(cls, device: torch.device | str) -> "AlignmentStats":
        return cls(
            samples=0,
            obs_prev_minus_executed_sum=torch.zeros(12, device=device),
            obs_prev_minus_executed_sq_sum=torch.zeros(12, device=device),
            obs_prev_minus_executed_abs_sum=torch.zeros(12, device=device),
            policy_minus_executed_sum=torch.zeros(12, device=device),
            policy_minus_executed_sq_sum=torch.zeros(12, device=device),
            policy_minus_executed_abs_sum=torch.zeros(12, device=device),
            hf_error_sum=torch.zeros(4, device=device),
            hf_error_sq_sum=torch.zeros(4, device=device),
            hf_error_abs_sum=torch.zeros(4, device=device),
            hf_pred_sum=torch.zeros(4, device=device),
            hf_target_sum=torch.zeros(4, device=device),
        )


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    if not getattr(runner.alg, "use_pie_actor_features", False):
        raise RuntimeError("This diagnostic expects a PIE actor inference wrapper.")

    policy = runner.get_pie_inference_policy(device=env.unwrapped.device)
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    action_joint_names = list(getattr(action_term, "_joint_names", ()))
    joint_indices_by_leg = [
        torch.tensor([action_joint_names.index(name) for name in names], device=env.device, dtype=torch.long)
        for names in JOINT_NAMES
    ]
    print("Action joint names:", action_joint_names)
    print("Previous action slice in PIE policy observation:", PREVIOUS_ACTION_SLICE)
    print("Estimator foot target order:", FOOT_TARGET_NAMES)

    stats = run_rollout(env, runner, policy, action_term)
    print_summary(stats, joint_indices_by_leg)
    env.close()


def run_rollout(
    env: ParkourRslRlVecEnvWrapper,
    runner: OnPolicyRunnerWithExtractor,
    policy,
    action_term,
) -> AlignmentStats:
    _, extras = env.reset()
    policy.reset()
    estimator_hidden = runner.alg.estimator.initial_hidden(env.num_envs, device=env.device)
    stats = AlignmentStats.create(env.device)

    for step in range(args_cli.steps + args_cli.warmup_steps):
        obs_dict = extras["observations"]
        with torch.inference_mode():
            predictions = runner.alg.estimator.forward_obs_dict(obs_dict, hidden_state=estimator_hidden)
            estimator_hidden = predictions["rnn_hidden"].detach()
            actions = policy(obs_dict, hist_encoding=True)

        if step >= args_cli.warmup_steps:
            accumulate_stats(stats, obs_dict, predictions, actions, action_term)

        _, _, dones, extras = env.step(actions)
        policy.reset(dones)
        done_mask = dones.to(device=estimator_hidden.device).reshape(-1).bool()
        if done_mask.any():
            estimator_hidden = estimator_hidden.clone()
            estimator_hidden[:, done_mask, :] = 0.0
    return stats


def accumulate_stats(
    stats: AlignmentStats,
    obs_dict,
    predictions,
    actions: torch.Tensor,
    action_term,
) -> None:
    policy_obs = obs_dict["policy"]
    obs_previous_action = policy_obs[:, PREVIOUS_ACTION_SLICE]
    executed_action = action_term.raw_actions
    obs_prev_minus_executed = obs_previous_action - executed_action
    policy_minus_executed = actions - executed_action

    stats.obs_prev_minus_executed_sum += obs_prev_minus_executed.mean(dim=0)
    stats.obs_prev_minus_executed_sq_sum += torch.square(obs_prev_minus_executed).mean(dim=0)
    stats.obs_prev_minus_executed_abs_sum += torch.abs(obs_prev_minus_executed).mean(dim=0)
    stats.policy_minus_executed_sum += policy_minus_executed.mean(dim=0)
    stats.policy_minus_executed_sq_sum += torch.square(policy_minus_executed).mean(dim=0)
    stats.policy_minus_executed_abs_sum += torch.abs(policy_minus_executed).mean(dim=0)

    hf_pred = predictions["h_f_hat"]
    hf_target = obs_dict["estimator_targets"]["foot_clearance"]
    hf_error = hf_pred - hf_target
    stats.hf_error_sum += hf_error.mean(dim=0)
    stats.hf_error_sq_sum += torch.square(hf_error).mean(dim=0)
    stats.hf_error_abs_sum += torch.abs(hf_error).mean(dim=0)
    stats.hf_pred_sum += hf_pred.mean(dim=0)
    stats.hf_target_sum += hf_target.mean(dim=0)
    stats.samples += 1


def print_summary(stats: AlignmentStats, joint_indices_by_leg: list[torch.Tensor]) -> None:
    denom = max(stats.samples, 1)
    obs_prev_mean = stats.obs_prev_minus_executed_sum / denom
    obs_prev_rms = torch.sqrt(stats.obs_prev_minus_executed_sq_sum / denom)
    obs_prev_abs = stats.obs_prev_minus_executed_abs_sum / denom
    policy_mean = stats.policy_minus_executed_sum / denom
    policy_rms = torch.sqrt(stats.policy_minus_executed_sq_sum / denom)
    policy_abs = stats.policy_minus_executed_abs_sum / denom

    hf_error_mean = stats.hf_error_sum / denom
    hf_rmse = torch.sqrt(stats.hf_error_sq_sum / denom)
    hf_abs = stats.hf_error_abs_sum / denom
    hf_pred_mean = stats.hf_pred_sum / denom
    hf_target_mean = stats.hf_target_sum / denom

    print("\n=== ACTION ALIGNMENT BY LEG ===")
    print("leg | obs_prev-exec mean | obs_prev-exec rms | obs_prev-exec abs | policy-exec mean | policy-exec rms | policy-exec abs")
    for leg, joint_ids in zip(LEG_NAMES, joint_indices_by_leg):
        print(
            f"{leg:>2}  | "
            f"{obs_prev_mean[joint_ids].mean().item():18.3f} | "
            f"{obs_prev_rms[joint_ids].mean().item():17.3f} | "
            f"{obs_prev_abs[joint_ids].mean().item():17.3f} | "
            f"{policy_mean[joint_ids].mean().item():16.3f} | "
            f"{policy_rms[joint_ids].mean().item():15.3f} | "
            f"{policy_abs[joint_ids].mean().item():15.3f}"
        )

    print("\n=== ACTION ALIGNMENT BY JOINT ===")
    print("joint | obs_prev-exec mean | obs_prev-exec rms | obs_prev-exec abs | policy-exec mean | policy-exec rms | policy-exec abs")
    for leg, joint_ids in zip(LEG_NAMES, joint_indices_by_leg):
        for joint_name, joint_idx in zip(("hip", "thigh", "calf"), joint_ids):
            print(
                f"{leg}_{joint_name:<5} | "
                f"{obs_prev_mean[joint_idx].item():18.3f} | "
                f"{obs_prev_rms[joint_idx].item():17.3f} | "
                f"{obs_prev_abs[joint_idx].item():17.3f} | "
                f"{policy_mean[joint_idx].item():16.3f} | "
                f"{policy_rms[joint_idx].item():15.3f} | "
                f"{policy_abs[joint_idx].item():15.3f}"
            )

    print("\n=== FOOT CLEARANCE ESTIMATOR ===")
    print("foot | target_mean | pred_mean | error_mean | rmse | abs_error")
    for idx, foot in enumerate(FOOT_TARGET_NAMES):
        print(
            f"{foot:>2}   | "
            f"{hf_target_mean[idx].item():11.3f} | "
            f"{hf_pred_mean[idx].item():9.3f} | "
            f"{hf_error_mean[idx].item():10.3f} | "
            f"{hf_rmse[idx].item():5.3f} | "
            f"{hf_abs[idx].item():9.3f}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
