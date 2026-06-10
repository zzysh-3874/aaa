"""Occlusion test for the PIE estimator's GRU temporal memory.

Idea: the GRU is supposed to remember terrain that the forward depth camera
saw a few steps ago but can no longer see (under the body / behind the robot).
To test whether it actually does, we run a normal rollout to build up the GRU
hidden state, then at each step run the estimator TWICE on the SAME hidden
state:

  (a) with the real current depth image
  (b) with the current depth image ZEROED OUT (occluded)

and compare each prediction (h_f = per-foot terrain heightmap, height = body
heightmap) against the privileged ground-truth target.

  * If occluded error  ~=  real error  -> the estimate barely uses the current
    frame; it is being carried by GRU memory + proprio -> GRU REMEMBERS.
  * If occluded error  >>  real error  -> the estimate collapses without the
    current frame -> the estimator is "living in the present", GRU memory is
    weak / not relied upon.

We also report the baseline "real" RMSE so the absolute accuracy is visible.

Run locally (needs a checkpoint scp'd from the server), headless:
  python scripts/diagnose_gru_memory.py --task <task> --checkpoint <pt> \
      --num_envs 16 --warmup_steps 80 --steps 240 --headless --enable_cameras
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import sys

from isaaclab.app import AppLauncher

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _SCRIPT_DIR]
sys.path.insert(0, str(_REPO_ROOT))

_CLI = _SCRIPT_DIR / "rsl_rl" / "cli_args.py"
_spec = importlib.util.spec_from_file_location("parkour_rsl_cli_args", _CLI)
cli_args = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(cli_args)

parser = argparse.ArgumentParser(description="Occlusion test for PIE GRU memory.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--warmup_steps", type=int, default=80, help="Steps to build GRU memory before measuring.")
parser.add_argument("--steps", type=int, default=240, help="Measurement steps.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
import isaaclab_tasks  # noqa
import parkour_tasks  # noqa
import parkour_tasks.extreme_parkour_task.config.go2  # noqa


def _unwrap(x):
    """obs terms may be a tensor or a single-key dict."""
    if isinstance(x, dict):
        return next(iter(x.values()))
    return x


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    runner.alg.estimator.eval()
    runner.alg.reset_pie_actor_hidden()

    obs, extras = env.get_observations()
    dev = env.unwrapped.device

    # accumulators
    n_meas = 0
    hf_real_sq = 0.0
    hf_occ_sq = 0.0
    ht_real_sq = 0.0
    ht_occ_sq = 0.0
    # also measure how much the prediction itself MOVES when we occlude
    # (independent of GT): large move = relies on current frame.
    hf_delta_sq = 0.0
    ht_delta_sq = 0.0

    depth_key = "depth_camera"
    ph_key = "proprioception_history"

    for step in range(args_cli.warmup_steps + args_cli.steps):
        obs_dict = extras["observations"]
        hidden = runner.alg.pie_actor_rnn_hidden
        if hidden is None or hidden.shape[1] != obs.shape[0]:
            hidden = runner.alg.estimator.initial_hidden(obs.shape[0], device=dev)
        hidden = hidden.detach().clone()

        depth = _unwrap(obs_dict[depth_key]).to(dev)
        ph = _unwrap(obs_dict[ph_key]).to(dev)

        with torch.no_grad():
            # (a) real depth — this also advances the "official" hidden state
            pred_real = runner.alg.estimator(depth, ph, hidden_state=hidden)
            # (b) occluded depth — SAME hidden state, current frame zeroed
            depth_zero = torch.zeros_like(depth)
            pred_occ = runner.alg.estimator(depth_zero, ph, hidden_state=hidden)

        if step >= args_cli.warmup_steps:
            tgt = obs_dict["estimator_targets"]
            hf_gt = _unwrap(tgt["foot_clearance"]).to(dev)
            ht_gt = _unwrap(tgt["height_scan"]).to(dev)

            hf_real_sq += torch.mean((pred_real["h_f_hat"] - hf_gt) ** 2).item()
            hf_occ_sq += torch.mean((pred_occ["h_f_hat"] - hf_gt) ** 2).item()
            ht_real_sq += torch.mean((pred_real["height_hat"] - ht_gt) ** 2).item()
            ht_occ_sq += torch.mean((pred_occ["height_hat"] - ht_gt) ** 2).item()
            hf_delta_sq += torch.mean((pred_occ["h_f_hat"] - pred_real["h_f_hat"]) ** 2).item()
            ht_delta_sq += torch.mean((pred_occ["height_hat"] - pred_real["height_hat"]) ** 2).item()
            n_meas += 1

        # advance the real policy + env (use real-depth hidden for continuity)
        runner.alg.pie_actor_rnn_hidden = pred_real["rnn_hidden"].detach()
        builtin_actor_obs = runner.alg.build_pie_actor_observations(obs, obs_dict)
        action = runner.alg.policy.act_inference(builtin_actor_obs, hist_encoding=True)
        obs, _, dones, extras = env.step(action)
        runner.alg.reset_pie_actor_hidden(dones)

    n = max(n_meas, 1)
    import math
    hf_real = math.sqrt(hf_real_sq / n)
    hf_occ = math.sqrt(hf_occ_sq / n)
    ht_real = math.sqrt(ht_real_sq / n)
    ht_occ = math.sqrt(ht_occ_sq / n)
    hf_delta = math.sqrt(hf_delta_sq / n)
    ht_delta = math.sqrt(ht_delta_sq / n)

    print("\n" + "=" * 72)
    print("GRU MEMORY OCCLUSION TEST")
    print(f"task={args_cli.task}")
    print(f"checkpoint={args_cli.checkpoint}")
    print(f"measure_steps={n_meas} (warmup={args_cli.warmup_steps})")
    print("=" * 72)
    print(f"{'target':12s} {'RMSE_real':>11s} {'RMSE_occluded':>14s} {'occ/real':>9s} {'pred_move':>10s}")
    print("-" * 72)
    print(f"{'h_f (foot)':12s} {hf_real:>11.4f} {hf_occ:>14.4f} {hf_occ/max(hf_real,1e-6):>9.2f} {hf_delta:>10.4f}")
    print(f"{'height(body)':12s} {ht_real:>11.4f} {ht_occ:>14.4f} {ht_occ/max(ht_real,1e-6):>9.2f} {ht_delta:>10.4f}")
    print("=" * 72)
    print("occ/real ~1.0  -> estimate carried by GRU memory + proprio (REMEMBERS)")
    print("occ/real >>1   -> estimate collapses without current frame (weak memory)")
    print("pred_move      -> how much the prediction shifts when current frame is zeroed")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
