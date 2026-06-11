"""Per-sub-terrain rollout diagnosis.

Runs a deterministic (noise=0) rollout and groups per-env results by the
sub-terrain type each env is on (parkour_gap / parkour_hurdle / parkour_flat /
parkour_step / stepping_stones / balance_beam). For each terrain type it reports:
  - mean how_far (distance from start before reset/timeout)
  - mean goals reached (cur_goal_idx) / total goals
  - termination cause breakdown (HEIGHT / ROLL / PITCH / FELL_OFF / REACH_GOAL / TIMEOUT)
  - completion rate (fraction of episodes that reached all goals)

This isolates WHICH terrain type is dragging terrain_levels down: a type with
low how_far + high fall/height termination is where the policy is failing.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import sys
from collections import defaultdict

from isaaclab.app import AppLauncher

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != _SCRIPT_DIR]
sys.path.insert(0, str(_REPO_ROOT))

_CLI = _SCRIPT_DIR / "rsl_rl" / "cli_args.py"
_spec = importlib.util.spec_from_file_location("parkour_rsl_cli_args", _CLI)
cli_args = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(cli_args)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=240)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--play_init_level", type=int, default=None,
                    help="Spawn envs across terrain rows 0..this (default: leave cfg as-is).")
parser.add_argument("--inference_noise_std", type=float, default=None,
                    help="Inject a fixed action noise std (e.g. 0.29 to match training). "
                         "Default None = deterministic (noise=0) like normal play.")
parser.add_argument("--play_min_height", type=float, default=None,
                    help="OBSERVE-ONLY: relax the minimum-height termination cutoff so a "
                         "shaky cold-start does not instantly reset (e.g. 0.10).")
parser.add_argument("--play_max_tilt", type=float, default=None,
                    help="OBSERVE-ONLY: relax max_roll/max_pitch termination cutoffs (radians).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
import isaaclab_tasks  # noqa
import parkour_tasks  # noqa
import parkour_tasks.extreme_parkour_task.config.go2  # noqa


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    if args_cli.play_init_level is not None:
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.play_init_level
        print(f"[DIAG] max_init_terrain_level -> {args_cli.play_init_level}")
    if args_cli.play_min_height is not None:
        env_cfg.terminations.total_terminates.params["minimum_height"] = args_cli.play_min_height
        print(f"[DIAG] minimum_height -> {args_cli.play_min_height}")
    if args_cli.play_max_tilt is not None:
        env_cfg.terminations.total_terminates.params["max_roll"] = args_cli.play_max_tilt
        env_cfg.terminations.total_terminates.params["max_pitch"] = args_cli.play_max_tilt
        print(f"[DIAG] max_roll/max_pitch -> {args_cli.play_max_tilt}")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_pie_inference_policy(device=env.unwrapped.device,
                                             inference_noise_std=args_cli.inference_noise_std)
    print(f"[DIAG] inference_noise_std = {args_cli.inference_noise_std}")

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    pe = unwrapped.parkour_manager.get_term("base_parkour")
    num_goals = unwrapped.scene.terrain.cfg.terrain_generator.num_goals
    maxep = unwrapped.max_episode_length
    n = unwrapped.num_envs
    dev = unwrapped.device

    obs, extras = env.get_observations()

    # Read the ACTUAL termination cutoffs in effect so cause attribution is
    # correct regardless of play-relaxed vs training params.
    _tp = unwrapped.termination_manager.get_term_cfg("total_terminates").params
    TH_ROLL = float(_tp.get("max_roll", 1.5))
    TH_PITCH = float(_tp.get("max_pitch", 1.5))
    TH_HEIGHT = _tp.get("minimum_height", None)
    TH_HEIGHT = float(TH_HEIGHT) if TH_HEIGHT is not None else -1e9
    print(f"[DIAG] termination cutoffs: roll>{TH_ROLL} pitch>{TH_PITCH} height<{TH_HEIGHT}")

    # Per-episode accumulators, reset on done.
    start_xy = robot.data.root_state_w[:, :2].clone()
    ep_len = torch.zeros(n, device=dev)

    # Per-terrain aggregation across all completed episodes.
    agg = defaultdict(lambda: {
        "episodes": 0, "how_far_sum": 0.0, "goals_sum": 0.0,
        "complete": 0,
        "HEIGHT": 0, "ROLL": 0, "PITCH": 0, "FELL_OFF": 0, "REACH_GOAL": 0, "TIMEOUT": 0, "UNKNOWN": 0,
    })

    # Per-terrain terrain_level distribution: every time an env resets, the
    # env's own curriculum has just updated its terrain_level (move_up/down by
    # whether it walked far enough). We record (terrain_name, new_level) at each
    # reset so we can see whether gap/hurdle envs are stuck at a low level while
    # flat/beam envs climb high -- i.e. whether the mean terrain_level is a
    # misleading average. ``terrain.terrain_levels`` is per-env (post-reset).
    terrain = unwrapped.scene.terrain
    level_by_terrain = defaultdict(list)

    for step in range(args_cli.steps):
        with torch.no_grad():
            actions = policy(extras["observations"], hist_encoding=True)

        # snapshot pre-step state (IsaacLab resets terminated envs inside step)
        roll_b, pitch_b, _ = euler_xyz_from_quat(robot.data.root_state_w[:, 3:7])
        roll_b = wrap_to_pi(roll_b).abs().clone()
        pitch_b = wrap_to_pi(pitch_b).abs().clone()
        z_b = robot.data.root_state_w[:, 2].clone()
        xy_b = robot.data.root_state_w[:, :2].clone()
        gidx_b = pe.cur_goal_idx.clone()
        names_b = pe.env_per_terrain_name[:, -1].copy()  # (n,) str

        obs, _, dones, extras = env.step(actions)
        # CRITICAL: reset the PIE actor GRU hidden state for envs that just
        # terminated, exactly like play.py does. Without this the hidden state
        # from the previous (possibly fallen) episode leaks into the freshly
        # respawned env, producing garbage actions -> instant re-termination
        # (the all-zero how_far / all-UNK artefact this script showed before).
        if hasattr(policy, "reset"):
            policy.reset(dones)
        ep_len += 1

        done_idx = dones.nonzero(as_tuple=False).flatten()
        for i in done_idx.tolist():
            name = str(names_b[i])
            how_far = float(torch.norm(xy_b[i] - start_xy[i]).item())
            gi = int(gidx_b[i].item())
            a = agg[name]
            a["episodes"] += 1
            a["how_far_sum"] += how_far
            a["goals_sum"] += gi
            r, p, zz = roll_b[i].item(), pitch_b[i].item(), z_b[i].item()
            if gi >= num_goals:
                a["REACH_GOAL"] += 1; a["complete"] += 1
            elif int(ep_len[i].item()) >= maxep:
                a["TIMEOUT"] += 1
            elif zz < -0.25:
                a["FELL_OFF"] += 1
            elif zz < TH_HEIGHT:
                a["HEIGHT"] += 1
            elif r > TH_ROLL:
                a["ROLL"] += 1
            elif p > TH_PITCH:
                a["PITCH"] += 1
            else:
                a["UNKNOWN"] += 1

        # reset accumulators for envs that ended (start_xy refreshes to new spawn)
        if done_idx.numel() > 0:
            # Record the updated terrain_level for each ended env, keyed by the
            # terrain TYPE it was just traversing (names_b snapshot pre-step).
            lvls = terrain.terrain_levels.detach().cpu().numpy()
            for i in done_idx.tolist():
                level_by_terrain[str(names_b[i])].append(float(lvls[i]))
            start_xy[done_idx] = robot.data.root_state_w[done_idx, :2].clone()
            ep_len[done_idx] = 0

    print("\n" + "=" * 100)
    print(f"PER-TERRAIN DIAGNOSIS  (task={args_cli.task}, num_envs={n}, steps={args_cli.steps}, goals/tile={num_goals})")
    print("=" * 100)
    header = f"{'terrain':16s} {'eps':>5s} {'how_far':>8s} {'goals':>6s} {'compl%':>7s} | {'HEIGHT':>6s} {'ROLL':>5s} {'PITCH':>5s} {'FELL':>5s} {'GOAL':>5s} {'TIME':>5s} {'UNK':>4s}"
    print(header)
    print("-" * 100)
    for name in sorted(agg.keys()):
        a = agg[name]
        e = max(a["episodes"], 1)
        print(f"{name:16s} {a['episodes']:>5d} {a['how_far_sum']/e:>8.2f} {a['goals_sum']/e:>6.2f} "
              f"{100*a['complete']/e:>6.1f}% | {a['HEIGHT']:>6d} {a['ROLL']:>5d} {a['PITCH']:>5d} "
              f"{a['FELL_OFF']:>5d} {a['REACH_GOAL']:>5d} {a['TIMEOUT']:>5d} {a['UNKNOWN']:>4d}")
    print("=" * 100)
    print("how_far: mean meters before reset | goals: mean cur_goal_idx reached | compl%: reached all goals")
    print("termination columns are episode COUNTS per cause")

    # Per-terrain terrain_level distribution (proves/refutes "mean level is
    # inflated by easy terrains while gap/hurdle envs stay stuck low").
    print("\n" + "=" * 70)
    print("PER-TERRAIN terrain_level (post-reset, curriculum-updated)")
    print("=" * 70)
    print(f"{'terrain':16s} {'n':>5s} {'mean_lvl':>9s} {'min':>5s} {'max':>5s}")
    print("-" * 70)
    import numpy as _np
    for name in sorted(level_by_terrain.keys()):
        arr = _np.array(level_by_terrain[name], dtype=float)
        if arr.size == 0:
            continue
        print(f"{name:16s} {arr.size:>5d} {arr.mean():>9.2f} {arr.min():>5.0f} {arr.max():>5.0f}")
    print("=" * 70)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
