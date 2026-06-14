"""Standalone sanity test for the separated TR-Net PIEEstimator path.

Verifies (without IsaacLab):
  - forward shapes match the single-GRU path
  - merged (2,B,H) hidden state round-trips through forward
  - depth actually influences z_m (the whole point: visual feed-forward)
  - gradients flow to the depth encoder from a z_m-based loss
  - done-style hidden masking broadcasts over the 2-layer hidden
"""
import sys
from pathlib import Path
import importlib.util

import torch

_REPO = Path(__file__).resolve().parent.parent
_EST_PATH = _REPO / "scripts" / "rsl_rl" / "modules" / "feature_extractors" / "pie_estimator.py"
_spec = importlib.util.spec_from_file_location("pie_estimator_standalone", _EST_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PIEEstimator = _mod.PIEEstimator


def build(**over):
    kw = dict(
        depth_channels=2,
        depth_image_shape=(58, 87),
        proprio_dim=45,
        proprio_history_len=10,
        z_m_dim=64,
        latent_dim=32,
        foot_height_dim=36,
        height_dim=132,
        use_height_refine=True,
        height_grid_shape=(12, 11),
        use_heightmap_encoder=True,
        use_separated_trnet=True,
    )
    kw.update(over)
    return PIEEstimator(**kw)


def main():
    torch.manual_seed(0)
    B = 8
    est = build()
    est.eval()

    depth = torch.randn(B, 2, 58, 87)
    proprio = torch.randn(B, 10, 45)

    # initial hidden must be (2, B, H)
    h0 = est.initial_hidden(B, device="cpu")
    assert h0.shape == (2, B, est.gru_hidden_dim), h0.shape

    out = est.forward(depth, proprio, hidden_state=h0)
    assert out["v_hat"].shape == (B, 3), out["v_hat"].shape
    assert out["h_f_hat"].shape == (B, 36), out["h_f_hat"].shape
    assert out["z_m"].shape == (B, 64), out["z_m"].shape
    assert out["z_mu"].shape == (B, 32), out["z_mu"].shape
    assert out["height_hat"].shape == (B, 132), out["height_hat"].shape
    assert out["rnn_hidden"].shape == (2, B, est.gru_hidden_dim), out["rnn_hidden"].shape
    print("[OK] shapes correct; hidden round-trips as (2,B,H)")

    # depth must influence z_m (visual feed-forward link)
    with torch.no_grad():
        z_m_a = est.forward(depth, proprio, hidden_state=h0)["z_m"]
        depth_shuf = depth[torch.randperm(B)]
        z_m_b = est.forward(depth_shuf, proprio, hidden_state=h0)["z_m"]
    delta = (z_m_a - z_m_b).abs().mean().item()
    print(f"[INFO] |z_m(depth) - z_m(depth_shuffled)| mean = {delta:.5f}")
    assert delta > 1e-4, "z_m does not depend on depth -- visual link broken!"
    print("[OK] z_m depends on depth (visual feed-forward present)")

    # gradient flows to the depth encoder from a z_m loss
    est.train()
    out = est.forward(depth, proprio, hidden_state=h0)
    loss = out["z_m"].pow(2).mean() + out["height_hat"].pow(2).mean()
    loss.backward()
    g = est.depth_encoder.encoder[0].weight.grad
    assert g is not None and g.abs().sum().item() > 0, "no grad to depth encoder"
    print("[OK] gradient reaches depth encoder from z_m/height loss")

    # done-style masking broadcasts across the 2-layer hidden
    hid = out["rnn_hidden"].detach()
    done = torch.zeros(B, dtype=torch.bool)
    done[0] = True
    masked = hid * (~done).reshape(1, -1, 1).to(hid.dtype)
    assert masked[:, 0, :].abs().sum().item() == 0.0
    assert masked[:, 1, :].abs().sum().item() > 0.0
    print("[OK] (2,B,H) hidden masks correctly per-env on done")

    print("\nALL SEPARATED-TRNET TESTS PASSED")


if __name__ == "__main__":
    main()
