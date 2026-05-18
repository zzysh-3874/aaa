from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_pie_estimator_class():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts/rsl_rl/modules/feature_extractors/pie_estimator.py"
    spec = importlib.util.spec_from_file_location("pie_estimator", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PIEEstimator


def test_pie_estimator_random_tensor_shapes():
    pie_estimator_cls = _load_pie_estimator_class()
    estimator = pie_estimator_cls()

    depth = torch.randn(8, 2, 58, 87)
    proprioception_history = torch.randn(8, 10, 45)

    outputs = estimator(depth, proprioception_history)

    expected_shapes = {
        "v_hat": (8, 3),
        "h_f_hat": (8, 4),
        "z_m": (8, 32),
        "z": (8, 32),
        "z_t": (8, 32),
        "z_mu": (8, 32),
        "z_logvar": (8, 32),
        "height_hat": (8, 132),
        "next_proprio_hat": (8, 45),
        "rnn_hidden": (1, 8, 256),
    }
    for key, expected_shape in expected_shapes.items():
        assert key in outputs
        assert tuple(outputs[key].shape) == expected_shape


def test_pie_estimator_cross_modal_sequence_shape():
    pie_estimator_cls = _load_pie_estimator_class()
    estimator = pie_estimator_cls()

    depth = torch.randn(4, 2, 58, 87)
    proprioception_history = torch.randn(4, 10, 45)

    sequence = estimator.encode_cross_modal_sequence(depth, proprioception_history)

    assert estimator.num_visual_tokens == 54
    assert estimator.proprio_history_dim == 450
    assert estimator.proprio_encoder[0].in_features == estimator.proprio_history_dim
    assert estimator.gru.input_size == estimator.fusion_dim * 2
    assert estimator.gru.hidden_size == estimator.fusion_dim * 2
    assert estimator.height_decoder[0].in_features == estimator.z_m_dim
    assert estimator.next_proprio_decoder[0].in_features == estimator.latent_dim + 3 + 4
    assert tuple(sequence.shape) == (4, 1, estimator.gru.input_size)


def test_pie_estimator_accepts_flattened_proprio_history():
    pie_estimator_cls = _load_pie_estimator_class()
    estimator = pie_estimator_cls()

    depth = torch.randn(4, 2, 58, 87)
    proprioception_history = torch.randn(4, 10 * 45)

    outputs = estimator(depth, proprioception_history)

    assert tuple(outputs["v_hat"].shape) == (4, 3)
    assert tuple(outputs["rnn_hidden"].shape) == (1, 4, 256)


if __name__ == "__main__":
    test_pie_estimator_random_tensor_shapes()
    test_pie_estimator_cross_modal_sequence_shape()
    test_pie_estimator_accepts_flattened_proprio_history()
    print("PIEEstimator random tensor shape test passed.")
