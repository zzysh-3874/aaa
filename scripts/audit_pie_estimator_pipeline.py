"""Audit PIE estimator-to-actor feature pipeline for a checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
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


parser = argparse.ArgumentParser(description="Audit PIE estimator feature pipeline.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--steps", type=int, default=240, help="Steps to summarize after warmup.")
parser.add_argument("--warmup_steps", type=int, default=80, help="Warmup steps before recording.")
parser.add_argument("--policy_action_limit", type=float, default=None, help="Override actor action_limit before loading.")
parser.add_argument("--clip_actions_override", type=float, default=None, help="Override wrapper clip_actions.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--summary_out", type=str, default=None, help="Optional file path to also write the summary text to.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")
# PIE depth observations use RayCasterCameraCfg, which works with the
# non-rendering headless kit. Do not force RTX rendering here; callers can still
# pass --enable_cameras explicitly when they need real camera rendering.

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


FEATURE_DIMS = {
    "z_m": 32,
    "z": 32,
    "z_mu": 32,
    "v_hat": 3,
    "h_f_hat": 4,
}
FOOT_NAMES = ("FL", "FR", "RL", "RR")


@dataclass
class EstimatorStats:
    samples: int = 0
    actor_obs_manual_sq: float = 0.0
    actor_obs_prefix_sq: float = 0.0
    z_vs_mu_sq: float = 0.0
    v_sq: torch.Tensor | None = None
    h_f_sq: torch.Tensor | None = None
    height_sq: float = 0.0
    next_proprio_sq: float = 0.0
    next_proprio_count: int = 0
    hidden_reset_checks: int = 0
    hidden_reset_max: float = 0.0
    finite_failures: int = 0
    feature_abs_sum: dict[str, float] | None = None
    feature_max_abs: dict[str, float] | None = None
    feature_clip_frac_sum: dict[str, float] | None = None
    z_mu_abs_sum: float = 0.0
    z_mu_max_abs: float = 0.0
    z_mu_batch_std_sum: float = 0.0
    z_logvar_mean_sum: float = 0.0
    z_logvar_abs_sum: float = 0.0
    posterior_std_mean_sum: float = 0.0
    posterior_std_min: float = float("inf")
    posterior_std_max: float = 0.0
    expected_sampled_z_abs_sum: float = 0.0
    kl_sum_mean_sum: float = 0.0
    kl_per_dim_mean_sum: float = 0.0
    sensitivity_sq: dict[str, float] | None = None
    actor_ablation_sq: dict[str, float] | None = None
    internal_actor_ablation_sq: dict[str, float] | None = None
    internal_head_ablation_sq: dict[str, float] | None = None
    attn_entropy_norm_sum: float = 0.0
    attn_max_sum: float = 0.0
    attn_std_sum: float = 0.0
    attn_out_norm_sum: float = 0.0
    grf_gate_mean_sum: float = 0.0
    grf_gate_std_sum: float = 0.0
    grf_residual_ratio_sum: float = 0.0
    highway_beta_mean_sum: float = 0.0
    highway_beta_std_sum: float = 0.0
    highway_beta_low_frac_sum: float = 0.0
    highway_beta_high_frac_sum: float = 0.0
    highway_gru_contrib_norm_sum: float = 0.0
    highway_fused_contrib_norm_sum: float = 0.0
    hidden_norm_sum: float = 0.0


def emit(msg: str = "") -> None:
    """Print to stdout and, if --summary_out is provided, also append to the file."""
    print(msg)
    out_path = getattr(args_cli, "summary_out", None)
    if out_path:
        try:
            with open(out_path, "a") as f:
                f.write(msg + "\n")
        except Exception as exc:  # pragma: no cover
            print(f"[emit] failed to write summary to {out_path}: {exc}")


def main() -> None:
    # Truncate any existing summary file at the start so each run is fresh.
    out_path = getattr(args_cli, "summary_out", None)
    if out_path:
        try:
            open(out_path, "w").close()
        except Exception as exc:  # pragma: no cover
            print(f"[emit] failed to truncate {out_path}: {exc}")
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

    if not getattr(runner.alg, "use_pie_actor_features", False):
        raise RuntimeError("This diagnostic expects use_pie_actor_features=True")

    print_config(env, agent_cfg, runner)
    stats = EstimatorStats(
        v_sq=torch.zeros(3, device=env.device),
        h_f_sq=torch.zeros(4, device=env.device),
        feature_abs_sum={key: 0.0 for key in runner.alg.pie_actor_feature_keys},
        feature_max_abs={key: 0.0 for key in runner.alg.pie_actor_feature_keys},
        feature_clip_frac_sum={key: 0.0 for key in runner.alg.pie_actor_feature_keys},
        sensitivity_sq={
            f"{source}_{key}": 0.0
            for source in ("depth_shuffle", "proprio_shuffle")
            for key in ("z_mu", "z_m", "v_hat", "h_f_hat")
        },
        actor_ablation_sq={},
        internal_actor_ablation_sq={},
        internal_head_ablation_sq={},
    )

    obs, extras = env.reset()
    runner.alg.reset_pie_actor_hidden()
    for step in range(args_cli.steps + args_cli.warmup_steps):
        obs_dict = extras["observations"]
        hidden = current_or_initial_hidden(runner, obs)

        with torch.no_grad():
            predictions = runner.alg.estimator.forward_obs_dict(obs_dict, hidden_state=hidden)
            prepared_features = runner.alg._prepare_pie_actor_features(predictions)
            manual_actor_obs = torch.cat((obs, *prepared_features), dim=-1)

        runner.alg.pie_actor_rnn_hidden = hidden.clone()
        builtin_actor_obs = runner.alg.build_pie_actor_observations(obs, obs_dict)
        action = runner.alg.policy.act_inference(builtin_actor_obs, hist_encoding=True)
        next_obs, _, dones, next_extras = env.step(action)

        if step >= args_cli.warmup_steps:
            accumulate_stats(
                stats,
                runner,
                obs,
                obs_dict,
                next_obs,
                dones,
                hidden,
                predictions,
                prepared_features,
                manual_actor_obs,
                builtin_actor_obs,
            )

        runner.alg.reset_pie_actor_hidden(dones)
        if step >= args_cli.warmup_steps and dones.any():
            done_hidden = runner.alg.pie_actor_rnn_hidden[:, dones.bool().reshape(-1)]
            stats.hidden_reset_checks += 1
            stats.hidden_reset_max = max(stats.hidden_reset_max, done_hidden.abs().max().item())

        obs = next_obs
        extras = next_extras

    # Synthetic reset check in case rollout did not terminate any env.
    if runner.alg.pie_actor_rnn_hidden is not None:
        mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        mask[0] = 1
        runner.alg.reset_pie_actor_hidden(mask)
        stats.hidden_reset_checks += 1
        stats.hidden_reset_max = max(stats.hidden_reset_max, runner.alg.pie_actor_rnn_hidden[:, 0].abs().max().item())

    print_summary(stats, runner)
    env.close()


def current_or_initial_hidden(runner: OnPolicyRunnerWithExtractor, obs: torch.Tensor) -> torch.Tensor:
    hidden = runner.alg.pie_actor_rnn_hidden
    if hidden is None or hidden.shape[1] != obs.shape[0]:
        return runner.alg.estimator.initial_hidden(obs.shape[0], device=obs.device)
    return hidden.detach().clone()


def print_config(env, agent_cfg, runner: OnPolicyRunnerWithExtractor) -> None:
    estimator = runner.alg.estimator
    emit("=== ESTIMATOR PIPELINE CONFIG ===")
    emit(f"task={args_cli.task}")
    emit(f"checkpoint={args_cli.checkpoint}")
    actor_obs_dim = getattr(runner.alg.policy, "num_actor_obs", None)
    if actor_obs_dim is None:
        actor_obs_dim = runner.alg.policy.actor[0].in_features
    emit(f"num_envs={env.num_envs}, policy_dim={env.num_obs}, actor_obs_dim={actor_obs_dim}")
    emit(f"policy_action_limit={getattr(runner.alg.policy, 'action_limit', None)}")
    emit(f"wrapper_clip_actions={agent_cfg.clip_actions}")
    emit(f"use_pie_actor_features={runner.alg.use_pie_actor_features}")
    emit(f"detach_pie_actor_features={runner.alg.detach_pie_actor_features}")
    emit(f"pie_joint_actor_estimator={runner.alg.pie_joint_actor_estimator}")
    emit(f"feature_keys={runner.alg.pie_actor_feature_keys}")
    emit(f"feature_clip={runner.alg.pie_actor_feature_clip}")
    emit(f"estimator_training_mode={estimator.training}")
    emit(f"sample_latent_in_training={getattr(estimator, 'sample_latent_in_training', None)}")
    emit(f"gru_hidden_dim={getattr(estimator, 'gru_hidden_dim', None)}")
    emit(f"expected_feature_dims={FEATURE_DIMS}")


def accumulate_stats(
    stats: EstimatorStats,
    runner: OnPolicyRunnerWithExtractor,
    policy_obs: torch.Tensor,
    obs_dict,
    next_policy_obs: torch.Tensor,
    dones: torch.Tensor,
    hidden: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    prepared_features: list[torch.Tensor],
    manual_actor_obs: torch.Tensor,
    builtin_actor_obs: torch.Tensor,
) -> None:
    stats.samples += 1
    stats.actor_obs_manual_sq += torch.square(builtin_actor_obs - manual_actor_obs).mean().item()
    # The actor obs is laid out as [policy_obs (proprio), *estimator_features].
    # Older PIE configs used proprio_dim=45; FullParkour uses 47. Derive the
    # prefix length from the actual policy_obs tensor so this audit works for
    # any PIE proprio dimension without a hardcoded constant.
    policy_dim = policy_obs.shape[-1]
    stats.actor_obs_prefix_sq += torch.square(builtin_actor_obs[:, :policy_dim] - policy_obs).mean().item()
    stats.z_vs_mu_sq += torch.square(predictions["z"] - predictions["z_mu"]).mean().item()
    if not all(torch.isfinite(value).all().item() for value in predictions.values()):
        stats.finite_failures += 1
    accumulate_latent_stats(stats, predictions)
    accumulate_sensitivity_stats(stats, runner, obs_dict, hidden, predictions)
    accumulate_actor_ablation_stats(stats, runner, policy_obs, predictions, builtin_actor_obs)
    accumulate_internal_path_stats(stats, runner, policy_obs, obs_dict, hidden, predictions, builtin_actor_obs)

    targets = obs_dict["estimator_targets"]
    stats.v_sq += torch.square(predictions["v_hat"] - targets["base_velocity"]).mean(dim=0)
    stats.h_f_sq += torch.square(predictions["h_f_hat"] - targets["foot_clearance"]).mean(dim=0)
    stats.height_sq += torch.square(predictions["height_hat"] - targets["height_scan"]).mean().item()
    mask = ~dones.bool().reshape(-1)
    if mask.any():
        stats.next_proprio_sq += torch.square(predictions["next_proprio_hat"][mask] - next_policy_obs[mask]).mean().item()
        stats.next_proprio_count += 1

    feature_clip = runner.alg.pie_actor_feature_clip
    for key, feature in zip(runner.alg.pie_actor_feature_keys, prepared_features, strict=True):
        stats.feature_abs_sum[key] += feature.abs().mean().item()
        stats.feature_max_abs[key] = max(stats.feature_max_abs[key], feature.abs().max().item())
        if feature_clip is None:
            clip_frac = 0.0
        else:
            clip = float(feature_clip)
            clip_frac = (feature.abs() >= clip - 1.0e-6).float().mean().item()
        stats.feature_clip_frac_sum[key] += clip_frac


def accumulate_latent_stats(stats: EstimatorStats, predictions: dict[str, torch.Tensor]) -> None:
    z_mu = predictions["z_mu"]
    z_logvar = predictions["z_logvar"]
    posterior_std = torch.exp(0.5 * z_logvar)
    kl_per_dim = -0.5 * (1.0 + z_logvar - torch.square(z_mu) - torch.exp(z_logvar))

    stats.z_mu_abs_sum += z_mu.abs().mean().item()
    stats.z_mu_max_abs = max(stats.z_mu_max_abs, z_mu.abs().max().item())
    stats.z_mu_batch_std_sum += z_mu.std(dim=0, unbiased=False).mean().item()
    stats.z_logvar_mean_sum += z_logvar.mean().item()
    stats.z_logvar_abs_sum += z_logvar.abs().mean().item()
    stats.posterior_std_mean_sum += posterior_std.mean().item()
    stats.posterior_std_min = min(stats.posterior_std_min, posterior_std.min().item())
    stats.posterior_std_max = max(stats.posterior_std_max, posterior_std.max().item())
    stats.expected_sampled_z_abs_sum += (posterior_std * (2.0 / torch.pi) ** 0.5).mean().item()
    stats.kl_sum_mean_sum += kl_per_dim.sum(dim=-1).mean().item()
    stats.kl_per_dim_mean_sum += kl_per_dim.mean().item()


def accumulate_sensitivity_stats(
    stats: EstimatorStats,
    runner: OnPolicyRunnerWithExtractor,
    obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    hidden: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    if hidden.shape[1] <= 1:
        return
    with torch.no_grad():
        depth_shuffled = replace_obs_dict_group(obs_dict, "depth_camera", roll_group(obs_dict["depth_camera"]))
        proprio_shuffled = replace_obs_dict_group(
            obs_dict, "proprioception_history", roll_group(obs_dict["proprioception_history"])
        )
        depth_predictions = runner.alg.estimator.forward_obs_dict(depth_shuffled, hidden_state=hidden)
        proprio_predictions = runner.alg.estimator.forward_obs_dict(proprio_shuffled, hidden_state=hidden)

    for key in ("z_mu", "z_m", "v_hat", "h_f_hat"):
        stats.sensitivity_sq[f"depth_shuffle_{key}"] += torch.square(
            depth_predictions[key] - predictions[key]
        ).mean().item()
        stats.sensitivity_sq[f"proprio_shuffle_{key}"] += torch.square(
            proprio_predictions[key] - predictions[key]
        ).mean().item()


def accumulate_actor_ablation_stats(
    stats: EstimatorStats,
    runner: OnPolicyRunnerWithExtractor,
    policy_obs: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    baseline_actor_obs: torch.Tensor,
) -> None:
    with torch.no_grad():
        baseline_action = runner.alg.policy.act_inference(baseline_actor_obs, hist_encoding=True)
        baseline_features = {key: predictions[key] for key in runner.alg.pie_actor_feature_keys}

        for ablation_name, feature_overrides in build_actor_feature_ablations(baseline_features, predictions).items():
            features = []
            for key in runner.alg.pie_actor_feature_keys:
                feature = feature_overrides.get(key, baseline_features[key])
                if runner.alg.pie_actor_feature_clip is not None:
                    clip = float(runner.alg.pie_actor_feature_clip)
                    feature = torch.clamp(feature, -clip, clip)
                features.append(feature)
            ablated_actor_obs = torch.cat((policy_obs, *features), dim=-1)
            ablated_action = runner.alg.policy.act_inference(ablated_actor_obs, hist_encoding=True)
            stats.actor_ablation_sq.setdefault(ablation_name, 0.0)
            stats.actor_ablation_sq[ablation_name] += torch.square(ablated_action - baseline_action).mean().item()


def build_actor_feature_ablations(
    baseline_features: Mapping[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    zero_ablations = {
        f"zero_{key}": {key: torch.zeros_like(feature)}
        for key, feature in baseline_features.items()
        if key in ("z", "z_mu", "z_m", "v_hat", "h_f_hat")
    }
    posterior_std = torch.exp(0.5 * predictions["z_logvar"])
    sampled_z = predictions["z_mu"] + torch.randn_like(posterior_std) * posterior_std
    zero_ablations["zero_all_estimator"] = {
        key: torch.zeros_like(feature) for key, feature in baseline_features.items()
    }
    if "z" in baseline_features:
        zero_ablations["sample_actor_latent_from_posterior"] = {"z": sampled_z}
    elif "z_mu" in baseline_features:
        zero_ablations["sample_actor_latent_from_posterior"] = {"z_mu": sampled_z}
    return zero_ablations


def accumulate_internal_path_stats(
    stats: EstimatorStats,
    runner: OnPolicyRunnerWithExtractor,
    policy_obs: torch.Tensor,
    obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    hidden: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    baseline_actor_obs: torch.Tensor,
) -> None:
    with torch.no_grad():
        internal = forward_estimator_internal(runner.alg.estimator, obs_dict, hidden)
        baseline_action = runner.alg.policy.act_inference(baseline_actor_obs, hist_encoding=True)
        stats.attn_entropy_norm_sum += internal["attn_entropy_norm"].item()
        stats.attn_max_sum += internal["attn_max"].item()
        stats.attn_std_sum += internal["attn_std"].item()
        stats.attn_out_norm_sum += internal["attn_out_norm"].item()
        stats.grf_gate_mean_sum += internal["grf_gate_mean"].item()
        stats.grf_gate_std_sum += internal["grf_gate_std"].item()
        stats.grf_residual_ratio_sum += internal["grf_residual_ratio"].item()
        stats.highway_beta_mean_sum += internal["highway_beta_mean"].item()
        stats.highway_beta_std_sum += internal["highway_beta_std"].item()
        stats.highway_beta_low_frac_sum += internal["highway_beta_low_frac"].item()
        stats.highway_beta_high_frac_sum += internal["highway_beta_high_frac"].item()
        stats.highway_gru_contrib_norm_sum += internal["highway_gru_contrib_norm"].item()
        stats.highway_fused_contrib_norm_sum += internal["highway_fused_contrib_norm"].item()
        stats.hidden_norm_sum += hidden.norm(dim=-1).mean().item()

        for mode in ("zero_cross_attention", "zero_gru_hidden", "highway_f_only", "highway_gru_only"):
            ablated = forward_estimator_internal(runner.alg.estimator, obs_dict, hidden, mode=mode)["predictions"]
            actor_obs = build_actor_obs_from_predictions(runner, policy_obs, ablated)
            action = runner.alg.policy.act_inference(actor_obs, hist_encoding=True)
            stats.internal_actor_ablation_sq.setdefault(mode, 0.0)
            stats.internal_actor_ablation_sq[mode] += torch.square(action - baseline_action).mean().item()
            for key in ("z_mu", "z_m", "v_hat", "h_f_hat"):
                stat_key = f"{mode}_{key}"
                stats.internal_head_ablation_sq.setdefault(stat_key, 0.0)
                stats.internal_head_ablation_sq[stat_key] += torch.square(ablated[key] - predictions[key]).mean().item()


def build_actor_obs_from_predictions(
    runner: OnPolicyRunnerWithExtractor,
    policy_obs: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    features = []
    for key in runner.alg.pie_actor_feature_keys:
        feature = predictions[key]
        if runner.alg.pie_actor_feature_clip is not None:
            clip = float(runner.alg.pie_actor_feature_clip)
            feature = torch.clamp(feature, -clip, clip)
        features.append(feature)
    return torch.cat((policy_obs, *features), dim=-1)


def forward_estimator_internal(
    estimator,
    obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    hidden: torch.Tensor,
    mode: str | None = None,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    depth = estimator._prepare_depth(obs_dict["depth_camera"])
    proprioception_history = estimator._prepare_proprioception_history(obs_dict["proprioception_history"], depth.device)
    batch_size = depth.shape[0]
    hidden = hidden.to(device=depth.device, dtype=depth.dtype)

    visual_tokens = estimator._tokenize_depth(depth)
    proprio_token = estimator._tokenize_proprioception(proprioception_history, batch_size)
    visual_tokens_norm = estimator.depth_ln(visual_tokens)
    attn_out, attn_weights = estimator.cross_attention(
        query=estimator.proprio_ln(proprio_token),
        key=visual_tokens_norm,
        value=visual_tokens_norm,
        need_weights=True,
        average_attn_weights=False,
    )
    if mode == "zero_cross_attention":
        attn_out = torch.zeros_like(attn_out)
    x = torch.cat((proprio_token.squeeze(1), attn_out.squeeze(1)), dim=-1)
    h = estimator.grf_activation(estimator.grf_fc1(estimator.grf_ln(x)))
    content, grf_gate_logits = estimator.grf_fc2(h).chunk(2, dim=-1)
    grf_gate = torch.sigmoid(grf_gate_logits)
    grf_residual = content * grf_gate
    f = x + grf_residual

    gru_hidden = estimator.initial_hidden(batch_size, device=depth.device) if mode == "zero_gru_hidden" else hidden
    gru_output, next_hidden_state = estimator.gru(f.unsqueeze(1), gru_hidden)
    z_rec = estimator.recurrent_projector(gru_output[:, -1])
    beta = torch.sigmoid(estimator.highway_gate(torch.cat((z_rec, f), dim=-1)))
    if mode == "highway_f_only":
        y = f
    elif mode == "highway_gru_only":
        y = z_rec
    else:
        y = beta * z_rec + (1.0 - beta) * f

    v_hat = estimator.v_head(y)
    h_f_hat = estimator.h_f_head(y)
    z_m = estimator.z_m_head(y)
    z_mu = estimator.z_mu_head(y)
    z_logvar = estimator.z_logvar_head(y)
    z = estimator._latent_sample(z_mu, z_logvar)
    height_hat = estimator.height_decoder(z_m)
    next_proprio_hat = estimator.next_proprio_decoder(torch.cat((z, v_hat, h_f_hat), dim=-1))
    predictions = {
        "v_hat": v_hat,
        "h_f_hat": h_f_hat,
        "z_m": z_m,
        "z": z,
        "z_t": z,
        "z_mu": z_mu,
        "z_logvar": z_logvar,
        "height_hat": height_hat,
        "next_proprio_hat": next_proprio_hat,
        "rnn_hidden": next_hidden_state,
    }

    attn = attn_weights.clamp_min(1.0e-12)
    attn_entropy = -(attn * attn.log()).sum(dim=-1)
    attn_entropy_norm = attn_entropy / torch.log(torch.tensor(attn.shape[-1], device=attn.device, dtype=attn.dtype))
    return {
        "predictions": predictions,
        "attn_entropy_norm": attn_entropy_norm.mean(),
        "attn_max": attn_weights.max(dim=-1).values.mean(),
        "attn_std": attn_weights.std(dim=-1, unbiased=False).mean(),
        "attn_out_norm": attn_out.norm(dim=-1).mean(),
        "grf_gate_mean": grf_gate.mean(),
        "grf_gate_std": grf_gate.std(unbiased=False),
        "grf_residual_ratio": grf_residual.norm(dim=-1).mean() / (x.norm(dim=-1).mean() + 1.0e-8),
        "highway_beta_mean": beta.mean(),
        "highway_beta_std": beta.std(unbiased=False),
        "highway_beta_low_frac": (beta < 0.2).float().mean(),
        "highway_beta_high_frac": (beta > 0.8).float().mean(),
        "highway_gru_contrib_norm": (beta * z_rec).norm(dim=-1).mean(),
        "highway_fused_contrib_norm": ((1.0 - beta) * f).norm(dim=-1).mean(),
    }


def replace_obs_dict_group(
    obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    key: str,
    value: torch.Tensor | Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor | Mapping[str, torch.Tensor]]:
    replaced = dict(obs_dict)
    replaced[key] = value
    return replaced


def roll_group(group: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(group, torch.Tensor):
        return torch.roll(group, shifts=1, dims=0)
    return {key: roll_group(value) for key, value in group.items()}


def print_summary(stats: EstimatorStats, runner: OnPolicyRunnerWithExtractor) -> None:
    denom = max(stats.samples, 1)
    next_denom = max(stats.next_proprio_count, 1)
    emit("\n=== ESTIMATOR PIPELINE CHECKS ===")
    emit(f"samples={stats.samples}")
    emit(f"actor_obs_builtin_vs_manual_rms={(stats.actor_obs_manual_sq / denom) ** 0.5:.8f}")
    emit(f"actor_obs_policy_prefix_rms={(stats.actor_obs_prefix_sq / denom) ** 0.5:.8f}")
    emit(f"z_vs_z_mu_rms={(stats.z_vs_mu_sq / denom) ** 0.5:.8f}")
    emit(f"finite_prediction_failures={stats.finite_failures}")
    emit(f"hidden_reset_checks={stats.hidden_reset_checks}, hidden_reset_max_abs={stats.hidden_reset_max:.8f}")

    emit("\n=== LATENT POSTERIOR STATS ===")
    emit(f"z_mu_mean_abs={stats.z_mu_abs_sum / denom:.8f}")
    emit(f"z_mu_max_abs={stats.z_mu_max_abs:.8f}")
    emit(f"z_mu_batch_std_mean={stats.z_mu_batch_std_sum / denom:.8f}")
    emit(f"z_logvar_mean={stats.z_logvar_mean_sum / denom:.8f}")
    emit(f"z_logvar_mean_abs={stats.z_logvar_abs_sum / denom:.8f}")
    emit(f"posterior_std_mean={stats.posterior_std_mean_sum / denom:.8f}")
    emit(f"posterior_std_min={stats.posterior_std_min:.8f}")
    emit(f"posterior_std_max={stats.posterior_std_max:.8f}")
    emit(f"expected_sampled_z_abs_mean={stats.expected_sampled_z_abs_sum / denom:.8f}")
    emit(f"kl_sum_mean={stats.kl_sum_mean_sum / denom:.8f}")
    emit(f"kl_per_dim_mean={stats.kl_per_dim_mean_sum / denom:.8f}")

    emit("\n=== ESTIMATOR TARGET RMSE ===")
    v_rmse = torch.sqrt(stats.v_sq / denom)
    h_f_rmse = torch.sqrt(stats.h_f_sq / denom)
    emit(f"v_hat_rmse=[{', '.join(f'{x:.5f}' for x in v_rmse.detach().cpu().tolist())}]")
    emit(
        "h_f_hat_rmse="
        + ", ".join(f"{name}:{value:.5f}" for name, value in zip(FOOT_NAMES, h_f_rmse.detach().cpu().tolist(), strict=True))
    )
    emit(f"height_hat_rmse={(stats.height_sq / denom) ** 0.5:.5f}")
    emit(f"next_proprio_hat_rmse={(stats.next_proprio_sq / next_denom) ** 0.5:.5f}")

    emit("\n=== ACTOR FEATURE STATS ===")
    emit("feature | mean_abs | max_abs | clip_frac")
    for key in runner.alg.pie_actor_feature_keys:
        emit(
            f"{key:<8} | "
            f"{stats.feature_abs_sum[key] / denom:8.5f} | "
            f"{stats.feature_max_abs[key]:7.5f} | "
            f"{stats.feature_clip_frac_sum[key] / denom:9.6f}"
        )

    emit("\n=== ONLINE INPUT SENSITIVITY RMS ===")
    emit("source          | z_mu     | z_m      | v_hat    | h_f_hat")
    for source in ("depth_shuffle", "proprio_shuffle"):
        emit(
            f"{source:<15} | "
            f"{(stats.sensitivity_sq[f'{source}_z_mu'] / denom) ** 0.5:8.5f} | "
            f"{(stats.sensitivity_sq[f'{source}_z_m'] / denom) ** 0.5:8.5f} | "
            f"{(stats.sensitivity_sq[f'{source}_v_hat'] / denom) ** 0.5:8.5f} | "
            f"{(stats.sensitivity_sq[f'{source}_h_f_hat'] / denom) ** 0.5:8.5f}"
        )

    emit("\n=== ACTOR FEATURE ABLATION ACTION RMS ===")
    for key, value in stats.actor_ablation_sq.items():
        emit(f"{key:<24} {(value / denom) ** 0.5:.8f}")

    emit("\n=== INTERNAL CROSS-ATTENTION / GATE / GRU STATS ===")
    emit(f"attn_entropy_norm={stats.attn_entropy_norm_sum / denom:.6f}")
    emit(f"attn_max={stats.attn_max_sum / denom:.6f}")
    emit(f"attn_std={stats.attn_std_sum / denom:.6f}")
    emit(f"attn_out_norm={stats.attn_out_norm_sum / denom:.6f}")
    emit(f"grf_gate_mean={stats.grf_gate_mean_sum / denom:.6f}")
    emit(f"grf_gate_std={stats.grf_gate_std_sum / denom:.6f}")
    emit(f"grf_residual_ratio={stats.grf_residual_ratio_sum / denom:.6f}")
    emit(f"highway_beta_mean={stats.highway_beta_mean_sum / denom:.6f}")
    emit(f"highway_beta_std={stats.highway_beta_std_sum / denom:.6f}")
    emit(f"highway_beta_low_frac={stats.highway_beta_low_frac_sum / denom:.6f}")
    emit(f"highway_beta_high_frac={stats.highway_beta_high_frac_sum / denom:.6f}")
    emit(f"highway_gru_contrib_norm={stats.highway_gru_contrib_norm_sum / denom:.6f}")
    emit(f"highway_fused_contrib_norm={stats.highway_fused_contrib_norm_sum / denom:.6f}")
    emit(f"actor_hidden_norm={stats.hidden_norm_sum / denom:.6f}")

    emit("\n=== INTERNAL ABLATION ACTION RMS ===")
    for key, value in stats.internal_actor_ablation_sq.items():
        emit(f"{key:<24} {(value / denom) ** 0.5:.8f}")

    emit("\n=== INTERNAL ABLATION HEAD RMS ===")
    emit("mode                 | z_mu    | z_m     | v_hat   | h_f_hat")
    for mode in ("zero_cross_attention", "zero_gru_hidden", "highway_f_only", "highway_gru_only"):
        emit(
            f"{mode:<20} | "
            f"{(stats.internal_head_ablation_sq[f'{mode}_z_mu'] / denom) ** 0.5:7.5f} | "
            f"{(stats.internal_head_ablation_sq[f'{mode}_z_m'] / denom) ** 0.5:7.5f} | "
            f"{(stats.internal_head_ablation_sq[f'{mode}_v_hat'] / denom) ** 0.5:7.5f} | "
            f"{(stats.internal_head_ablation_sq[f'{mode}_h_f_hat'] / denom) ** 0.5:7.5f}"
        )

    print_actor_input_weight_norms(runner)


def print_actor_input_weight_norms(runner: OnPolicyRunnerWithExtractor) -> None:
    first_layer = runner.alg.policy.actor[0]
    weight = first_layer.weight.detach()
    # The actor input is [policy_obs (proprio), *estimator_features]. Derive the
    # proprio prefix length from the actor's input width minus the estimator
    # feature dims, so this works for any proprio_dim (45 older, 47 FullParkour).
    feature_dim_total = sum(FEATURE_DIMS[key] for key in runner.alg.pie_actor_feature_keys)
    offset = weight.shape[1] - feature_dim_total
    emit("\n=== ACTOR FIRST-LAYER INPUT WEIGHT NORMS ===")
    emit(f"{'policy':<8} mean_col_l2={weight[:, :offset].norm(dim=0).mean().item():.6f}")
    for key in runner.alg.pie_actor_feature_keys:
        dim = FEATURE_DIMS[key]
        segment = weight[:, offset : offset + dim]
        emit(f"{key:<8} mean_col_l2={segment.norm(dim=0).mean().item():.6f}")
        offset += dim


if __name__ == "__main__":
    main()
    simulation_app.close()
