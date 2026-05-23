# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate ParkourEvent.is_in_gap_zone against ground truth.

Plays a checkpoint on ``Isaac-PIE-GapOnly-Unitree-Go2-v0`` and, on every
step, recomputes the expected gap-zone membership from the per-env gap
intervals using a separate code path. Any disagreement is logged and counted.

Run::

    python scripts/rsl_rl/validate_gap_zone.py \
        --task Isaac-PIE-GapOnly-Unitree-Go2-v0 \
        --load_run 2026-05-17_17-00-14 \
        --checkpoint /abs/path/to/model_30500.pt \
        --num_envs 8 --max_steps 600 --headless

Pass criteria: ``mismatches == 0`` AND we observed:
  * at least one env with ``in_zone=True`` (so a positive case was hit);
  * at least one parkour_flat env that stayed False every step.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports (resolve via cwd-injected scripts/rsl_rl)
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Validate is_in_gap_zone end-to-end.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-PIE-GapOnly-Unitree-Go2-v0", help="Task id.")
parser.add_argument("--max_steps", type=int, default=600, help="Validate for this many env steps then exit.")
parser.add_argument(
    "--print_every", type=int, default=50, help="Print a summary line every N steps."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def expected_in_zone(
    env_gap_intervals: torch.Tensor,
    gap_zone_pre: float,
    gap_zone_post: float,
    x_local: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation for cross-checking is_in_gap_zone.

    Loops per-env so the logic stays obviously correct even if the
    vectorised version has a bug.
    """
    num_envs = env_gap_intervals.shape[0]
    out = torch.zeros(num_envs, dtype=torch.bool, device=env_gap_intervals.device)
    for i in range(num_envs):
        for s, e in env_gap_intervals[i]:
            s_f = float(s.item())
            e_f = float(e.item())
            if s_f != s_f or e_f != e_f:  # NaN check
                continue
            if (s_f - gap_zone_pre) <= float(x_local[i].item()) <= (e_f + gap_zone_post):
                out[i] = True
                break
    return out


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[VAL] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    use_pie_inference = (
        agent_cfg.algorithm.class_name != "DistillationWithExtractor"
        and getattr(runner.alg, "use_pie_actor_features", False)
    )
    if use_pie_inference:
        policy = runner.get_pie_inference_policy(device=env.unwrapped.device)
    else:
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        estimator_paras = agent_cfg.to_dict()["estimator"]
        num_prop = estimator_paras["num_prop"]
        num_scan = estimator_paras["num_scan"]
        num_priv_explicit = estimator_paras["num_priv_explicit"]
        estimator = runner.get_estimator_inference_policy(device=env.device)

    parkour_event = env.unwrapped.parkour_manager.get_term("base_parkour")

    # Sanity-check the loaded gap intervals up front.
    print("\n[VAL] === Static check (terrain_gap_intervals) ===")
    print(f"      shape = {tuple(parkour_event.terrain_gap_intervals.shape)}")
    print(f"      gap_zone_pre  = {parkour_event.gap_zone_pre}")
    print(f"      gap_zone_post = {parkour_event.gap_zone_post}")
    flat = parkour_event.terrain_gap_intervals.reshape(-1, 2)
    n_total = flat.shape[0]
    n_nan_rows = torch.isnan(flat[:, 0]).sum().item()
    n_real = n_total - n_nan_rows
    print(f"      tile×slot rows: {n_total} total, {n_real} valid, {n_nan_rows} NaN-padded")
    if n_real > 0:
        valid = flat[~torch.isnan(flat[:, 0])]
        print(f"      valid x_start range: [{valid[:, 0].min():.3f}, {valid[:, 0].max():.3f}]")
        print(f"      valid x_end   range: [{valid[:, 1].min():.3f}, {valid[:, 1].max():.3f}]")
        widths = valid[:, 1] - valid[:, 0]
        print(f"      valid widths : min={widths.min():.3f}, max={widths.max():.3f}, mean={widths.mean():.3f}")

    # Check tile sub_terrain names → expect parkour_flat tiles to have all-NaN rows.
    flat_count = 0
    flat_nonzero = 0
    gap_count = 0
    gap_zero = 0
    for r in range(parkour_event.terrain_gap_intervals.shape[0]):
        for c in range(parkour_event.terrain_gap_intervals.shape[1]):
            tile_name = parkour_event.total_terrain_names[r, c, 0]
            tile_intervals = parkour_event.terrain_gap_intervals[r, c]
            n_valid = (~torch.isnan(tile_intervals[:, 0])).sum().item()
            if tile_name == "parkour_flat":
                flat_count += 1
                if n_valid > 0:
                    flat_nonzero += 1
                    print(f"      ! flat tile (r={r}, c={c}) has {n_valid} gap intervals — should be 0")
            elif tile_name == "gap_corridor":
                gap_count += 1
                if n_valid == 0:
                    gap_zero += 1
                    print(f"      ! gap tile (r={r}, c={c}) has 0 gap intervals — should be > 0")
    print(f"      flat tiles: {flat_count}, with non-empty gap rows: {flat_nonzero}")
    print(f"      gap  tiles: {gap_count}, with 0 gap rows: {gap_zero}")

    # Walk loop with per-step validation.
    obs, extras = env.get_observations()
    mismatches = 0
    flat_env_in_zone = 0
    any_env_in_zone = 0
    timestep = 0

    print("\n[VAL] === Per-step validation ===")
    while simulation_app.is_running() and timestep < args_cli.max_steps:
        with torch.inference_mode():
            if use_pie_inference:
                actions = policy(extras["observations"], hist_encoding=True)
            else:
                obs[:, num_prop + num_scan : num_prop + num_scan + num_priv_explicit] = (
                    estimator.inference(obs[:, :num_prop])
                )
                actions = policy(obs, hist_encoding=True)
        obs, _, dones, extras = env.step(actions)
        if use_pie_inference:
            policy.reset(dones)

        # Compute reference vs API answer.
        positions_w = parkour_event.robot.data.root_pos_w[:, :2]
        x_local = positions_w[:, 0] - parkour_event.env_origins[:, 0]
        api_zone = parkour_event.is_in_gap_zone(positions_w)
        ref_zone = expected_in_zone(
            parkour_event.env_gap_intervals,
            parkour_event.gap_zone_pre,
            parkour_event.gap_zone_post,
            x_local,
        )
        mismatch_mask = api_zone != ref_zone
        if mismatch_mask.any():
            for idx in mismatch_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                mismatches += 1
                tile_name = parkour_event.env_per_terrain_name[idx, 0]
                intervals = parkour_event.env_gap_intervals[idx].cpu().tolist()
                print(
                    f"      MISMATCH step={timestep} env={idx} tile={tile_name} "
                    f"x_local={float(x_local[idx]):.3f} api={bool(api_zone[idx])} "
                    f"ref={bool(ref_zone[idx])} intervals={intervals}"
                )

        if api_zone.any():
            any_env_in_zone += 1

        # Track whether any flat-tile env ever reports in-zone (it should not).
        for idx in range(args_cli.num_envs):
            if (
                parkour_event.env_per_terrain_name[idx, 0] == "parkour_flat"
                and bool(api_zone[idx])
            ):
                flat_env_in_zone += 1
                print(
                    f"      FLAT_IN_ZONE step={timestep} env={idx} "
                    f"x_local={float(x_local[idx]):.3f}"
                )

        if timestep % args_cli.print_every == 0:
            sample_idx = 0
            tile_name = parkour_event.env_per_terrain_name[sample_idx, 0]
            all_intervals = parkour_event.env_gap_intervals[sample_idx].cpu().tolist()
            print(
                f"      step={timestep:4d} "
                f"env0(tile={tile_name}, x_local={float(x_local[sample_idx]):.3f}, "
                f"api_zone={bool(api_zone[sample_idx])}, "
                f"all_gaps={all_intervals}) "
                f"any_in_zone={int(api_zone.sum())}/{args_cli.num_envs}"
            )

        timestep += 1

    print("\n[VAL] === Summary ===")
    print(f"      total steps validated   : {timestep}")
    print(f"      mismatch count          : {mismatches}")
    print(f"      steps with any env in zone : {any_env_in_zone}")
    print(f"      flat_env reports in-zone: {flat_env_in_zone}")

    if mismatches == 0 and flat_env_in_zone == 0 and any_env_in_zone > 0:
        print("[VAL] PASS — is_in_gap_zone matches reference, never fires on flat, "
              "and triggered at least once on gap tiles.")
    else:
        print("[VAL] FAIL — see counters above.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
