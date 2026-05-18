from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import importlib.util
from pathlib import Path

import torch

try:
    from .pie_estimator_loss import build_pie_transition_targets, cache_pie_estimator_targets
except ImportError:
    module_path = Path(__file__).with_name("pie_estimator_loss.py")
    spec = importlib.util.spec_from_file_location("pie_estimator_loss", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    build_pie_transition_targets = module.build_pie_transition_targets
    cache_pie_estimator_targets = module.cache_pie_estimator_targets


DEFAULT_PIE_ESTIMATOR_TARGET_SHAPES = {
    "base_velocity": (3,),
    "foot_clearance": (4,),
    "height_scan": (132,),
    "next_proprioception": (45,),
    "next_proprioception_mask": (1,),
}


@dataclass
class PIEEstimatorStepInput:
    """Cached obs_t estimator input and current-step targets."""

    depth: torch.Tensor
    proprioception_history: torch.Tensor
    estimator_targets: dict[str, torch.Tensor]


class PIEEstimatorRolloutStorage:
    """Rollout storage for PIE estimator auxiliary supervision.

    This storage is intentionally separate from PPO's policy storage. It caches
    estimator inputs from obs_t and aligns next_proprioception against obs_{t+1}.
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        depth_shape: tuple[int, ...] = (2, 58, 87),
        proprioception_history_shape: tuple[int, ...] = (10, 45),
        target_shapes: Mapping[str, tuple[int, ...]] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.depth_shape = tuple(depth_shape)
        self.proprioception_history_shape = tuple(proprioception_history_shape)
        self.target_shapes = dict(DEFAULT_PIE_ESTIMATOR_TARGET_SHAPES)
        if target_shapes is not None:
            self.target_shapes.update(target_shapes)
        self.device = torch.device(device)
        self.dtype = dtype

        self.depth = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *self.depth_shape,
            device=self.device,
            dtype=dtype,
        )
        self.proprioception_history = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *self.proprioception_history_shape,
            device=self.device,
            dtype=dtype,
        )
        self.estimator_targets = {
            key: torch.zeros(
                num_transitions_per_env,
                num_envs,
                *shape,
                device=self.device,
                dtype=dtype,
            )
            for key, shape in self.target_shapes.items()
        }
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device, dtype=torch.bool)
        self.step = 0

    def cache_step_input(
        self,
        obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
        depth_key: str = "depth_camera",
        proprioception_history_key: str = "proprioception_history",
        estimator_targets_key: str = "estimator_targets",
    ) -> PIEEstimatorStepInput:
        """Clone obs_t estimator tensors before env.step can refresh observation buffers."""
        depth = self._cache_observation(obs_dict[depth_key], self.depth_shape, depth_key)
        proprioception_history = self._cache_observation(
            obs_dict[proprioception_history_key],
            self.proprioception_history_shape,
            proprioception_history_key,
        )
        estimator_targets = cache_pie_estimator_targets(obs_dict[estimator_targets_key])
        return PIEEstimatorStepInput(depth, proprioception_history, estimator_targets)

    def add_transition(
        self,
        step_input: PIEEstimatorStepInput,
        next_obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
        dones: torch.Tensor | None = None,
        terminated: torch.Tensor | None = None,
        truncated: torch.Tensor | None = None,
    ) -> None:
        """Store one obs_t -> obs_{t+1} estimator transition."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("PIE estimator rollout storage overflow. Call clear() before adding transitions.")

        self.depth[self.step].copy_(self._to_storage_tensor(step_input.depth, self.depth_shape, "depth"))
        self.proprioception_history[self.step].copy_(
            self._to_storage_tensor(
                step_input.proprioception_history,
                self.proprioception_history_shape,
                "proprioception_history",
            )
        )

        transition_targets = build_pie_transition_targets(
            step_input.estimator_targets,
            next_obs_dict,
            dones=dones,
            terminated=terminated,
            truncated=truncated,
        )
        if "next_proprioception_mask" not in transition_targets:
            transition_targets["next_proprioception_mask"] = torch.ones(
                self.num_envs,
                1,
                device=self.device,
                dtype=self.dtype,
            )

        for key, storage_tensor in self.estimator_targets.items():
            if key not in transition_targets:
                raise KeyError(f"Missing PIE estimator target: {key}")
            storage_tensor[self.step].copy_(
                self._to_storage_tensor(transition_targets[key], self.target_shapes[key], key)
            )

        done_tensor = self._prepare_dones(dones, terminated, truncated)
        self.dones[self.step].copy_(done_tensor)
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    def is_full(self) -> bool:
        return self.step == self.num_transitions_per_env

    def get(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Return time-major storage up to the current write step."""
        return (
            self.depth[: self.step],
            self.proprioception_history[: self.step],
            {key: value[: self.step] for key, value in self.estimator_targets.items()},
            self.dones[: self.step],
        )

    def flatten(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Flatten time and environment dimensions for estimator training."""
        depth, proprioception_history, targets, dones = self.get()
        return (
            depth.flatten(0, 1),
            proprioception_history.flatten(0, 1),
            {key: value.flatten(0, 1) for key, value in targets.items()},
            dones.flatten(0, 1),
        )

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 1,
        shuffle: bool = True,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]]:
        """Yield flattened PIE estimator minibatches."""
        depth, proprioception_history, targets, dones = self.flatten()
        batch_size = depth.shape[0]
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size == 0:
            raise ValueError(f"num_mini_batches={num_mini_batches} is larger than batch_size={batch_size}")
        sampled_size = mini_batch_size * num_mini_batches

        for _ in range(num_epochs):
            if shuffle:
                indices = torch.randperm(batch_size, device=self.device)[:sampled_size]
            else:
                indices = torch.arange(sampled_size, device=self.device)
            for mini_batch_idx in range(num_mini_batches):
                start = mini_batch_idx * mini_batch_size
                end = start + mini_batch_size
                batch_idx = indices[start:end]
                yield (
                    depth[batch_idx],
                    proprioception_history[batch_idx],
                    {key: value[batch_idx] for key, value in targets.items()},
                    dones[batch_idx],
                )

    def sequence_mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 1,
        shuffle_envs: bool = True,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]]:
        """Yield time-major minibatches that preserve per-env temporal order."""
        depth, proprioception_history, targets, dones = self.get()
        if self.step == 0:
            return
        num_chunks = min(num_mini_batches, self.num_envs)
        if num_chunks <= 0:
            raise ValueError(f"num_mini_batches must be positive, got {num_mini_batches}")

        for _ in range(num_epochs):
            if shuffle_envs:
                env_indices = torch.randperm(self.num_envs, device=self.device)
            else:
                env_indices = torch.arange(self.num_envs, device=self.device)
            for env_batch_idx in torch.chunk(env_indices, num_chunks):
                if env_batch_idx.numel() == 0:
                    continue
                yield (
                    depth[:, env_batch_idx],
                    proprioception_history[:, env_batch_idx],
                    {key: value[:, env_batch_idx] for key, value in targets.items()},
                    dones[:, env_batch_idx],
                )

    def _cache_observation(
        self,
        value: torch.Tensor | Mapping[str, torch.Tensor],
        expected_shape: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        return self._to_storage_tensor(value, expected_shape, name).detach().clone()

    def _to_storage_tensor(
        self,
        value: torch.Tensor | Mapping[str, torch.Tensor],
        expected_shape: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        tensor = _unwrap_single_term(value).to(device=self.device, dtype=self.dtype)
        expected_batch_shape = (self.num_envs, *expected_shape)
        if tuple(tensor.shape) != expected_batch_shape:
            raise ValueError(f"{name} must have shape {expected_batch_shape}, got {tuple(tensor.shape)}")
        return tensor

    def _prepare_dones(
        self,
        dones: torch.Tensor | None,
        terminated: torch.Tensor | None,
        truncated: torch.Tensor | None,
    ) -> torch.Tensor:
        if dones is None:
            if terminated is None and truncated is None:
                return torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
            if terminated is None:
                dones = truncated
            elif truncated is None:
                dones = terminated
            else:
                dones = torch.logical_or(terminated.to(dtype=torch.bool), truncated.to(dtype=torch.bool))

        dones = dones.to(device=self.device)
        if dones.ndim == 1:
            dones = dones.unsqueeze(-1)
        if dones.ndim != 2 or dones.shape != (self.num_envs, 1):
            raise ValueError(f"dones must have shape ({self.num_envs},) or ({self.num_envs}, 1), got {tuple(dones.shape)}")
        return dones.to(dtype=torch.bool)


def _unwrap_single_term(value: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(value, Mapping):
        if len(value) != 1:
            raise ValueError(f"Expected a single observation term, got keys: {list(value.keys())}")
        value = next(iter(value.values()))
    return value
