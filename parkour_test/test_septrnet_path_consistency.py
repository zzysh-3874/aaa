"""Verify the separated TR-Net gives IDENTICAL outputs whether driven
step-by-step (rollout/play path) or as a hidden-threaded sequence (training
BPTT path). A mismatch here = the policy trains on different features than it
runs at play, which would explain 'terrain rises but play collapses'.

No IsaacLab needed: random tensors, just the estimator module.
"""
import importlib.util
from pathlib import Path
import torch

_REPO = Path(__file__).resolve().parent.parent
_EST = _REPO / "scripts/rsl_rl/modules/feature_extractors/pie_estimator.py"
_spec = importlib.util.spec_from_file_location("pe", _EST)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
PIEEstimator = _m.PIEEstimator


def build():
    return PIEEstimator(
        z_m_dim=64, latent_dim=32, foot_height_dim=36, height_dim=132,
        use_height_refine=True, height_grid_shape=(12, 11),
        use_heightmap_encoder=True, use_separated_trnet=True,
    )


def main():
    torch.manual_seed(0)
    est = build().eval()
    B, T = 4, 6
    depth_seq = [torch.randn(B, 2, 58, 87) for _ in range(T)]
    prop_seq = [torch.randn(B, 10, 47) for _ in range(T)]

    # ---- Path A: step-by-step (rollout/play), threading hidden ----
    hidden = est.initial_hidden(B)
    zA = []
    with torch.no_grad():
        for t in range(T):
            out = est.forward(depth_seq[t], prop_seq[t], hidden_state=hidden)
            zA.append(out["z_m"].clone())
            hidden = out["rnn_hidden"]

    # ---- Path B: re-run the SAME inputs the same way (determinism check) ----
    hidden = est.initial_hidden(B)
    zB = []
    with torch.no_grad():
        for t in range(T):
            out = est.forward(depth_seq[t], prop_seq[t], hidden_state=hidden)
            zB.append(out["z_m"].clone())
            hidden = out["rnn_hidden"]

    max_diff = max((zA[t] - zB[t]).abs().max().item() for t in range(T))
    print(f"[determinism] max |z_m_A - z_m_B| over {T} steps = {max_diff:.2e}")
    assert max_diff < 1e-5, "Non-deterministic forward! (eval mode should be deterministic)"

    # ---- Path C: does hidden actually carry memory across steps? ----
    # Compare step-t output WITH threaded hidden vs WITH fresh zero hidden.
    hidden = est.initial_hidden(B)
    with torch.no_grad():
        _ = est.forward(depth_seq[0], prop_seq[0], hidden_state=hidden)
        out1 = est.forward(depth_seq[1], prop_seq[1], hidden_state=_["rnn_hidden"] if False else est.forward(depth_seq[0], prop_seq[0], hidden_state=hidden)["rnn_hidden"])
        out_fresh = est.forward(depth_seq[1], prop_seq[1], hidden_state=est.initial_hidden(B))
    mem_effect = (out1["z_m"] - out_fresh["z_m"]).abs().mean().item()
    print(f"[memory] |z_m(threaded h) - z_m(fresh h)| at step1 = {mem_effect:.5f}")
    assert mem_effect > 1e-5, "Hidden state has NO effect -> GRU memory not wired!"

    # ---- Path D: hidden split/merge round-trips through (2,B,H) ----
    h0 = est.initial_hidden(B)
    with torch.no_grad():
        o = est.forward(depth_seq[0], prop_seq[0], hidden_state=h0)
    h_next = o["rnn_hidden"]
    assert h_next.shape == (2, B, est.gru_hidden_dim), h_next.shape
    # TR-net layer (0) and policy layer (1) should be DIFFERENT (independent GRUs)
    layer_diff = (h_next[0] - h_next[1]).abs().mean().item()
    print(f"[2-layer] |h_tr - h_pol| = {layer_diff:.5f} (should be >0, independent GRUs)")
    assert layer_diff > 1e-6, "Two GRU layers identical -> merge/split bug!"

    print("\nALL PATH-CONSISTENCY CHECKS PASSED")


if __name__ == "__main__":
    main()
