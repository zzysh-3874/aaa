from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PIE_ESTIMATOR_LOSS_WEIGHTS = {
    "v": 1.0,
    "h_f": 1.0,
    "height": 1.0,
    "next_proprio": 1.0,
    "kl": 0.0,
    # Terrain-adaptive weighting strength for the height / foot-clearance
    # losses. 0.0 disables it (uniform MSE, original behaviour). When > 0,
    # per-sample loss weights are derived from the spatial roughness of the
    # ground-truth height scan so that complex terrain (steps / slopes / gaps)
    # contributes more gradient than flat ground. See
    # ``_terrain_difficulty_weight``. A value of 1.0 is a moderate emphasis.
    "terrain_adaptive": 0.0,
}

# PIE proprioception is [ang_vel(3), gravity(3), command(3), joint_pos(12),
# joint_vel(12), previous_action(12)]. For obs_{t+1}, previous_action is the
# sampled action a_t, so it is not a clean prediction target for the estimator.
NEXT_PROPRIO_STATE_DIM = 33


def cache_pie_estimator_targets(
    targets: Mapping[str, torch.Tensor],
    clone: bool = True,
) -> dict[str, torch.Tensor]:
    """Detach current-step PIE targets before the env advances to the next transition."""
    cached_targets: dict[str, torch.Tensor] = {}
    for key, value in targets.items():
        cached_value = value.detach()
        if clone:
            cached_value = cached_value.clone()
        cached_targets[key] = cached_value
    return cached_targets


def build_pie_transition_targets(
    current_targets: Mapping[str, torch.Tensor],
    next_obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    policy_key: str = "policy",
    next_proprio_dim: int = 47,
    clone: bool = True,
    dones: torch.Tensor | None = None,
    terminated: torch.Tensor | None = None,
    truncated: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Align obs_t estimator targets with obs_{t+1} proprioception.

    The estimator input comes from obs_t. Current-step targets such as velocity,
    foot clearance, and height scan stay from obs_t, while next_proprioception is
    replaced by the proprioceptive policy observation from obs_{t+1}.
    """
    transition_targets = cache_pie_estimator_targets(current_targets, clone=clone)
    next_proprioception = _unwrap_single_term(next_obs_dict[policy_key])
    if next_proprioception.ndim != 2:
        raise ValueError(
            "Next proprioception target must be shaped (B, D), "
            f"got {tuple(next_proprioception.shape)}"
        )
    if next_proprioception.shape[-1] != next_proprio_dim:
        raise ValueError(
            f"Next proprioception target dim must be {next_proprio_dim}, "
            f"got {next_proprioception.shape[-1]}"
        )
    next_proprioception = next_proprioception.detach()
    if clone:
        next_proprioception = next_proprioception.clone()
    transition_targets["next_proprioception"] = next_proprioception

    done_tensor = _combine_dones(dones, terminated, truncated)
    if done_tensor is not None:
        transition_targets["next_proprioception_mask"] = _next_proprioception_mask_from_dones(
            done_tensor, next_proprioception
        )
    return transition_targets


def _terrain_difficulty_weight(height_target: torch.Tensor, strength: float) -> torch.Tensor:
    """Per-sample loss weight derived from height-scan spatial roughness.

    Flat ground has a near-constant height scan (low spatial std); steps,
    slopes and gaps have a high spatial std. We map the per-sample spatial std
    of the ground-truth height scan to a multiplicative loss weight:

        w_i = 1 + strength * (std_i / mean_std)

    then normalise so the batch-mean weight is 1.0. Normalisation keeps the
    overall loss magnitude unchanged (so the existing learning rate / loss
    balance still holds) while redistributing gradient toward rough terrain.

    Args:
        height_target: (B, height_dim) ground-truth height scan.
        strength: emphasis factor (>= 0). 0 returns uniform weights.

    Returns:
        (B, 1) per-sample weights with batch mean ~= 1.
    """
    if strength <= 0.0:
        return torch.ones(height_target.shape[0], 1, device=height_target.device, dtype=height_target.dtype)
    # Spatial std across the height-scan dimension, per sample.
    std_i = height_target.std(dim=-1, keepdim=True)  # (B, 1)
    mean_std = std_i.mean().clamp_min(1e-6)
    raw = 1.0 + strength * (std_i / mean_std)
    # Normalise so batch-mean weight == 1 (preserve overall loss scale).
    return raw / raw.mean().clamp_min(1e-6)


def _weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """MSE with per-sample (B,1) weights; reduces to plain mean when weight==1."""
    squared_error = (prediction - target).pow(2)
    # weight is (B, 1); broadcast over feature dim. Because weight is
    # batch-mean-normalised to 1, this matches F.mse_loss when weights are
    # uniform.
    return (squared_error * weight).mean()


def compute_pie_estimator_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute PIE estimator supervision losses without touching PPO or rollout storage."""
    loss_weights = DEFAULT_PIE_ESTIMATOR_LOSS_WEIGHTS.copy()
    if weights is not None:
        loss_weights.update(weights)

    v_target = _target_like(targets["base_velocity"], predictions["v_hat"])
    h_f_target = _target_like(targets["foot_clearance"], predictions["h_f_hat"])
    height_target = _target_like(targets["height_scan"], predictions["height_hat"])
    next_proprio_target = _target_like(targets["next_proprioception"], predictions["next_proprio_hat"])
    next_proprio_mask = targets.get("next_proprioception_mask")

    # Terrain-adaptive per-sample weighting for the perception losses (height
    # scan + foot clearance). Derived from the ground-truth height scan, so no
    # extra terrain metadata is needed. Disabled (uniform) when strength == 0.
    terrain_strength = float(loss_weights.get("terrain_adaptive", 0.0))
    terrain_weight = _terrain_difficulty_weight(height_target, terrain_strength)

    loss_v = F.mse_loss(predictions["v_hat"], v_target)
    loss_hf = _weighted_mse(predictions["h_f_hat"], h_f_target, terrain_weight)
    loss_height = _weighted_mse(predictions["height_hat"], height_target, terrain_weight)
    loss_next_proprio = _masked_mse(
        predictions["next_proprio_hat"][..., :NEXT_PROPRIO_STATE_DIM],
        next_proprio_target[..., :NEXT_PROPRIO_STATE_DIM],
        next_proprio_mask,
    )
    loss_kl = _gaussian_kl(predictions["z_mu"], predictions["z_logvar"])

    loss = (
        loss_weights["v"] * loss_v
        + loss_weights["h_f"] * loss_hf
        + loss_weights["height"] * loss_height
        + loss_weights["next_proprio"] * loss_next_proprio
        + loss_weights["kl"] * loss_kl
    )

    return {
        "loss": loss,
        "loss_v": loss_v,
        "loss_hf": loss_hf,
        "loss_height": loss_height,
        "loss_next_proprio": loss_next_proprio,
        "loss_kl": loss_kl,
    }


class PIEEstimatorLoss(nn.Module):
    """Thin module wrapper around :func:`compute_pie_estimator_loss`."""

    def __init__(self, weights: Mapping[str, float] | None = None):
        super().__init__()
        self.weights = DEFAULT_PIE_ESTIMATOR_LOSS_WEIGHTS.copy()
        if weights is not None:
            self.weights.update(weights)

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return compute_pie_estimator_loss(predictions, targets, self.weights)


def _target_like(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return target.to(device=reference.device, dtype=reference.dtype)


def _gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    per_dim_kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return per_dim_kl.mean()


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    squared_error = (prediction - target).pow(2)
    if mask is None:
        return squared_error.mean()

    mask = mask.to(device=prediction.device, dtype=prediction.dtype)
    while mask.ndim < squared_error.ndim:
        mask = mask.unsqueeze(-1)
    try:
        mask = mask.expand_as(squared_error)
    except RuntimeError as exc:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} cannot broadcast to prediction shape {tuple(prediction.shape)}"
        ) from exc

    denominator = mask.sum().clamp_min(1.0)
    return (squared_error * mask).sum() / denominator


def _combine_dones(
    dones: torch.Tensor | None,
    terminated: torch.Tensor | None,
    truncated: torch.Tensor | None,
) -> torch.Tensor | None:
    if dones is not None:
        return dones
    if terminated is None:
        return truncated
    if truncated is None:
        return terminated
    return torch.logical_or(terminated.to(dtype=torch.bool), truncated.to(dtype=torch.bool))


def _next_proprioception_mask_from_dones(dones: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dones = dones.to(device=reference.device)
    if dones.ndim == 0:
        dones = dones.reshape(1)
    if dones.ndim == 1:
        dones = dones.unsqueeze(-1)
    if dones.ndim != 2 or dones.shape[-1] != 1:
        raise ValueError(f"Dones must be shaped (B,) or (B, 1), got {tuple(dones.shape)}")
    if dones.shape[0] != reference.shape[0]:
        raise ValueError(f"Dones batch size {dones.shape[0]} does not match target batch size {reference.shape[0]}")
    return (~dones.to(dtype=torch.bool)).to(dtype=reference.dtype)


def _unwrap_single_term(value: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(value, Mapping):
        if len(value) != 1:
            raise ValueError(f"Expected a single observation term, got keys: {list(value.keys())}")
        value = next(iter(value.values()))
    return value
