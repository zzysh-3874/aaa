from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_storage_class():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts/rsl_rl/modules/feature_extractors/pie_estimator_rollout_storage.py"
    spec = importlib.util.spec_from_file_location("pie_estimator_rollout_storage", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PIEEstimatorRolloutStorage


def _make_obs_dict(batch_size: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    return {
        "policy": torch.randn(batch_size, 45),
        "depth_camera": torch.randn(batch_size, 2, 58, 87),
        "proprioception_history": torch.randn(batch_size, 10, 45),
        "estimator_targets": {
            "base_velocity": torch.randn(batch_size, 3),
            "foot_clearance": torch.randn(batch_size, 4),
            "height_scan": torch.randn(batch_size, 132),
            "next_proprioception": torch.randn(batch_size, 45),
        },
    }


def test_pie_estimator_rollout_storage_caches_transition_aligned_targets():
    pie_estimator_rollout_storage = _load_storage_class()
    num_envs = 3
    num_steps = 4
    storage = pie_estimator_rollout_storage(num_envs, num_steps)

    for step in range(num_steps):
        obs_dict = _make_obs_dict(num_envs)
        cached_step = storage.cache_step_input(obs_dict)
        expected_depth = cached_step.depth.clone()
        expected_base_velocity = cached_step.estimator_targets["base_velocity"].clone()
        next_obs_dict = _make_obs_dict(num_envs)
        next_policy = next_obs_dict["policy"].clone()
        dones = torch.tensor([False, step % 2 == 0, False])

        obs_dict["depth_camera"].add_(100.0)
        obs_dict["estimator_targets"]["base_velocity"].add_(100.0)
        storage.add_transition(cached_step, next_obs_dict, dones=dones)

        assert torch.equal(storage.depth[step], expected_depth)
        assert torch.equal(storage.estimator_targets["base_velocity"][step], expected_base_velocity)
        assert torch.equal(storage.estimator_targets["next_proprioception"][step], next_policy)
        assert torch.equal(storage.estimator_targets["next_proprioception_mask"][step], (~dones).float().unsqueeze(-1))

    depth, proprioception_history, targets, dones = storage.get()
    assert tuple(depth.shape) == (num_steps, num_envs, 2, 58, 87)
    assert tuple(proprioception_history.shape) == (num_steps, num_envs, 10, 45)
    assert tuple(targets["base_velocity"].shape) == (num_steps, num_envs, 3)
    assert tuple(targets["foot_clearance"].shape) == (num_steps, num_envs, 4)
    assert tuple(targets["height_scan"].shape) == (num_steps, num_envs, 132)
    assert tuple(targets["next_proprioception"].shape) == (num_steps, num_envs, 45)
    assert tuple(targets["next_proprioception_mask"].shape) == (num_steps, num_envs, 1)
    assert tuple(dones.shape) == (num_steps, num_envs, 1)

    flat_depth, flat_proprioception_history, flat_targets, flat_dones = storage.flatten()
    assert tuple(flat_depth.shape) == (num_steps * num_envs, 2, 58, 87)
    assert tuple(flat_proprioception_history.shape) == (num_steps * num_envs, 10, 45)
    assert tuple(flat_targets["next_proprioception_mask"].shape) == (num_steps * num_envs, 1)
    assert tuple(flat_dones.shape) == (num_steps * num_envs, 1)


def test_pie_estimator_rollout_storage_minibatch_generator_shapes():
    pie_estimator_rollout_storage = _load_storage_class()
    num_envs = 2
    num_steps = 4
    storage = pie_estimator_rollout_storage(num_envs, num_steps)
    for _ in range(num_steps):
        cached_step = storage.cache_step_input(_make_obs_dict(num_envs))
        storage.add_transition(cached_step, _make_obs_dict(num_envs), dones=torch.zeros(num_envs, dtype=torch.bool))

    batches = list(storage.mini_batch_generator(num_mini_batches=4, num_epochs=1, shuffle=False))
    assert len(batches) == 4
    for depth, proprioception_history, targets, dones in batches:
        assert tuple(depth.shape) == (2, 2, 58, 87)
        assert tuple(proprioception_history.shape) == (2, 10, 45)
        assert tuple(targets["base_velocity"].shape) == (2, 3)
        assert tuple(targets["next_proprioception"].shape) == (2, 45)
        assert tuple(targets["next_proprioception_mask"].shape) == (2, 1)
        assert tuple(dones.shape) == (2, 1)


def test_pie_estimator_rollout_storage_sequence_minibatch_generator_preserves_time_order():
    pie_estimator_rollout_storage = _load_storage_class()
    num_envs = 3
    num_steps = 4
    storage = pie_estimator_rollout_storage(num_envs, num_steps)
    for step in range(num_steps):
        cached_step = storage.cache_step_input(_make_obs_dict(num_envs))
        storage.add_transition(
            cached_step,
            _make_obs_dict(num_envs),
            dones=torch.tensor([False, step == 1, False]),
        )

    batches = list(storage.sequence_mini_batch_generator(num_mini_batches=2, num_epochs=1, shuffle_envs=False))

    assert len(batches) == 2
    total_envs = 0
    for depth, proprioception_history, targets, dones in batches:
        total_envs += depth.shape[1]
        assert tuple(depth.shape[0:1] + depth.shape[2:]) == (num_steps, 2, 58, 87)
        assert tuple(proprioception_history.shape[0:1] + proprioception_history.shape[2:]) == (num_steps, 10, 45)
        assert tuple(targets["base_velocity"].shape[0:1] + targets["base_velocity"].shape[2:]) == (num_steps, 3)
        assert tuple(targets["next_proprioception"].shape[0:1] + targets["next_proprioception"].shape[2:]) == (
            num_steps,
            45,
        )
        assert tuple(targets["next_proprioception_mask"].shape[0:1] + targets["next_proprioception_mask"].shape[2:]) == (
            num_steps,
            1,
        )
        assert tuple(dones.shape[0:1] + dones.shape[2:]) == (num_steps, 1)
    assert total_envs == num_envs
