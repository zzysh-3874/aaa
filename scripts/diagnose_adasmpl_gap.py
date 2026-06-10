"""Diagnose the AdaSmpl train/play z_m distribution gap.

Hypothesis: during training the actor's terrain code z_m is encoded from the
GROUND-TRUTH heightmap with probability `adasmpl_prob` (high, ~0.65-0.8 in the
current run), but at play time adasmpl_prob=0 so z_m is always encoded from the
network's RECONSTRUCTED heightmap. If the two z_m differ a lot, the actor sees
an out-of-distribution input at play and produces erratic actions.

For each step (after a warmup so the GRU memory is populated) we run the
estimator twice from the SAME hidden state:
  (A) z_m from GT heightmap  (what the actor mostly saw in training)
  (B) z_m from reconstruction (what the actor sees at play)
and measure:
  - RMSE between z_m_gt and z_m_recon
  - RMSE between the resulting actor actions
  - action RMS scale for reference

A large action RMSE (relative to the action RMS) confirms the gap is the
root cause of the erratic play behaviour.
"""

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

parser = argparse.ArgumentParser(description="Diagnose AdaSmpl GT-vs-recon z_m gap.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--warmup_steps", type=int, default=60)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")

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


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    runner.alg.policy.eval()
    runner.alg.estimator.eval()

    alg = runner.alg
    if not getattr(alg, "use_pie_actor_features", False):
        raise RuntimeError("expects use_pie_actor_features=True")
    if getattr(alg.estimator, "heightmap_encoder", None) is None:
        raise RuntimeError("estimator has no heightmap_encoder; this run does not use the AdaSmpl z_m path")

    feature_keys = alg.pie_actor_feature_keys
    zm_index = feature_keys.index("z_m") if "z_m" in feature_keys else None
    print(f"[DIAG] pie_actor_feature_keys = {feature_keys}")
    print(f"[DIAG] pie_adasmpl_max_prob = {getattr(alg, 'pie_adasmpl_max_prob', None)}")

    def build_actor_obs(policy_obs, predictions):
        feats = alg._prepare_pie_actor_features(predictions)
        if alg.detach_pie_actor_features:
            feats = [f.detach() for f in feats]
        return torch.cat((policy_obs, *feats), dim=-1)

    obs, extras = env.reset()
    alg.reset_pie_actor_hidden()

    n = 0
    zm_sq = 0.0
    zm_gt_norm_sq = 0.0
    act_sq = 0.0
    act_gt_norm_sq = 0.0
    act_recon_norm_sq = 0.0

    total = args_cli.steps + args_cli.warmup_steps
    for step in range(total):
        obs_dict = extras["observations"]
        batch = obs.shape[0]
        if alg.pie_actor_rnn_hidden is None or alg.pie_actor_rnn_hidden.shape[1] != batch:
            alg.pie_actor_rnn_hidden = alg.estimator.initial_hidden(batch, device=obs.device)
        hidden = alg.pie_actor_rnn_hidden

        gt_hm = alg._extract_gt_heightmap(obs_dict)

        with torch.no_grad():
            # (B) reconstruction path (what play uses): adasmpl_prob=0
            pred_recon = alg.estimator.forward_obs_dict(
                obs_dict, hidden_state=hidden, gt_heightmap=None, adasmpl_prob=0.0
            )
            actor_obs_recon = build_actor_obs(obs, pred_recon)
            act_recon = alg.policy.act_inference(actor_obs_recon, hist_encoding=True)

            measured = False
            if gt_hm is not None:
                # (A) GT path (what training mostly used): force all-GT z_m
                pred_gt = alg.estimator.forward_obs_dict(
                    obs_dict, hidden_state=hidden, gt_heightmap=gt_hm, adasmpl_prob=1.0
                )
                actor_obs_gt = build_actor_obs(obs, pred_gt)
                act_gt = alg.policy.act_inference(actor_obs_gt, hist_encoding=True)

                if step >= args_cli.warmup_steps:
                    zm_sq += torch.square(pred_gt["z_m"] - pred_recon["z_m"]).mean().item()
                    zm_gt_norm_sq += torch.square(pred_gt["z_m"]).mean().item()
                    act_sq += torch.square(act_gt - act_recon).mean().item()
                    act_gt_norm_sq += torch.square(act_gt).mean().item()
                    act_recon_norm_sq += torch.square(act_recon).mean().item()
                    n += 1
                    measured = True

            # advance the env using the recon action (deterministic play-like)
            # and keep the actor hidden in sync via the standard builder.
            alg.pie_actor_rnn_hidden = hidden  # ensure builder starts from same hidden
            _ = alg.build_pie_actor_observations(obs, obs_dict)  # updates pie_actor_rnn_hidden
        obs, _, dones, extras = env.step(act_recon)
        alg.reset_pie_actor_hidden(dones)

    print("\n" + "=" * 70)
    print("ADASMPL GT-vs-RECON z_m GAP DIAGNOSIS")
    print("=" * 70)
    if n == 0:
        print("No GT heightmap available in obs_dict['estimator_targets']['height_scan'].")
        print("Cannot run the comparison (the env did not expose GT targets).")
    else:
        import math

        zm_rmse = math.sqrt(zm_sq / n)
        zm_gt_rms = math.sqrt(zm_gt_norm_sq / n)
        act_rmse = math.sqrt(act_sq / n)
        act_gt_rms = math.sqrt(act_gt_norm_sq / n)
        act_recon_rms = math.sqrt(act_recon_norm_sq / n)
        print(f"samples                : {n}")
        print(f"z_m  RMSE (gt vs recon): {zm_rmse:.5f}")
        print(f"z_m  RMS  (gt)         : {zm_gt_rms:.5f}   -> relative gap {zm_rmse / max(zm_gt_rms,1e-9):.2%}")
        print(f"action RMSE (gt vs recon): {act_rmse:.5f}")
        print(f"action RMS (gt path)   : {act_gt_rms:.5f}")
        print(f"action RMS (recon path): {act_recon_rms:.5f}")
        print(f"action relative divergence: {act_rmse / max(act_gt_rms,1e-9):.2%}")
        print("-" * 70)
        print("INTERPRETATION:")
        print("  action relative divergence > ~30% => the actor was trained on GT z_m")
        print("  and behaves very differently on the reconstructed z_m it sees at play.")
        print("  That is the train/play mismatch causing erratic play behaviour.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
