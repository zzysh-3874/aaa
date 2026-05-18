from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_loss_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts/rsl_rl/modules/feature_extractors/pie_estimator_loss.py"
    spec = importlib.util.spec_from_file_location("pie_estimator_loss", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pie_estimator_loss_random_tensor_backward():
    loss_module = _load_loss_module()
    batch_size = 8
    predictions = {
        "v_hat": torch.randn(batch_size, 3, requires_grad=True),
        "h_f_hat": torch.randn(batch_size, 4, requires_grad=True),
        "z_m": torch.randn(batch_size, 32, requires_grad=True),
        "z_mu": torch.randn(batch_size, 32, requires_grad=True),
        "z_logvar": torch.randn(batch_size, 32, requires_grad=True),
        "height_hat": torch.randn(batch_size, 132, requires_grad=True),
        "next_proprio_hat": torch.randn(batch_size, 45, requires_grad=True),
    }
    targets = {
        "base_velocity": torch.randn(batch_size, 3),
        "foot_clearance": torch.randn(batch_size, 4),
        "height_scan": torch.randn(batch_size, 132),
        "next_proprioception": torch.randn(batch_size, 45),
    }

    losses = loss_module.compute_pie_estimator_loss(predictions, targets, weights={"kl": 0.01})

    expected_keys = {"loss", "loss_v", "loss_hf", "loss_height", "loss_next_proprio", "loss_kl"}
    assert set(losses.keys()) == expected_keys
    for value in losses.values():
        assert value.ndim == 0
        assert value.isfinite()

    losses["loss"].backward()
    for key in ("v_hat", "h_f_hat", "z_mu", "z_logvar", "height_hat", "next_proprio_hat"):
        grad = predictions[key].grad
        assert grad is not None
        assert grad.isfinite().all()


def test_build_pie_transition_targets_uses_next_policy_proprioception():
    loss_module = _load_loss_module()
    batch_size = 8
    current_targets = {
        "base_velocity": torch.randn(batch_size, 3),
        "foot_clearance": torch.randn(batch_size, 4),
        "height_scan": torch.randn(batch_size, 132),
        "next_proprioception": torch.zeros(batch_size, 45),
    }
    current_snapshot = {key: value.clone() for key, value in current_targets.items()}
    next_policy = torch.randn(batch_size, 45)

    transition_targets = loss_module.build_pie_transition_targets(
        current_targets,
        {"policy": next_policy},
    )

    assert torch.equal(transition_targets["base_velocity"], current_snapshot["base_velocity"])
    assert torch.equal(transition_targets["foot_clearance"], current_snapshot["foot_clearance"])
    assert torch.equal(transition_targets["height_scan"], current_snapshot["height_scan"])
    assert torch.equal(transition_targets["next_proprioception"], next_policy)

    current_targets["base_velocity"].add_(100.0)
    next_policy.add_(100.0)
    assert torch.equal(transition_targets["base_velocity"], current_snapshot["base_velocity"])
    assert not torch.equal(transition_targets["next_proprioception"], next_policy)


def test_build_pie_transition_targets_adds_next_proprioception_mask():
    loss_module = _load_loss_module()
    batch_size = 3
    current_targets = {
        "base_velocity": torch.randn(batch_size, 3),
        "foot_clearance": torch.randn(batch_size, 4),
        "height_scan": torch.randn(batch_size, 132),
        "next_proprioception": torch.zeros(batch_size, 45),
    }
    next_policy = torch.randn(batch_size, 45)

    transition_targets = loss_module.build_pie_transition_targets(
        current_targets,
        {"policy": next_policy},
        dones=torch.tensor([False, True, False]),
    )

    expected_mask = torch.tensor([[1.0], [0.0], [1.0]])
    assert torch.equal(transition_targets["next_proprioception_mask"], expected_mask)


def test_next_proprioception_loss_uses_done_mask():
    loss_module = _load_loss_module()
    batch_size = 3
    predictions = {
        "v_hat": torch.zeros(batch_size, 3),
        "h_f_hat": torch.zeros(batch_size, 4),
        "z_m": torch.zeros(batch_size, 32),
        "z_mu": torch.zeros(batch_size, 32),
        "z_logvar": torch.zeros(batch_size, 32),
        "height_hat": torch.zeros(batch_size, 132),
        "next_proprio_hat": torch.zeros(batch_size, 45),
    }
    targets = {
        "base_velocity": torch.zeros(batch_size, 3),
        "foot_clearance": torch.zeros(batch_size, 4),
        "height_scan": torch.zeros(batch_size, 132),
        "next_proprioception": torch.zeros(batch_size, 45),
        "next_proprioception_mask": torch.tensor([[1.0], [0.0], [1.0]]),
    }
    targets["next_proprioception"][0].fill_(1.0)
    targets["next_proprioception"][1].fill_(10.0)

    losses = loss_module.compute_pie_estimator_loss(predictions, targets)
    assert torch.isclose(losses["loss_next_proprio"], torch.tensor(0.5))

    targets["next_proprioception_mask"].zero_()
    losses = loss_module.compute_pie_estimator_loss(predictions, targets)
    assert torch.equal(losses["loss_next_proprio"], torch.tensor(0.0))


def test_gaussian_kl_averages_latent_dimensions():
    loss_module = _load_loss_module()
    mu = torch.ones(2, 32)
    logvar = torch.zeros(2, 32)

    loss_kl = loss_module._gaussian_kl(mu, logvar)

    assert torch.isclose(loss_kl, torch.tensor(0.5))


if __name__ == "__main__":
    test_pie_estimator_loss_random_tensor_backward()
    test_build_pie_transition_targets_uses_next_policy_proprioception()
    test_build_pie_transition_targets_adds_next_proprioception_mask()
    test_next_proprioception_loss_uses_done_mask()
    test_gaussian_kl_averages_latent_dimensions()
    print("PIEEstimator loss random tensor backward test passed.")
