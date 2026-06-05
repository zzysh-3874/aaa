from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    """Resolve a small set of activations without pulling in runner dependencies."""
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(_activation(activation))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class DepthCNNEncoder(nn.Module):
    """Small CNN encoder that keeps a 2D feature map for visual tokenization."""

    def __init__(
        self,
        in_channels: int = 2,
        output_dim: int = 128,
        output_shape: tuple[int, int] = (6, 9),
        activation: str = "elu",
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            _activation(activation),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(64, output_dim, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.AdaptiveAvgPool2d(output_shape),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.encoder(depth)


class HeightmapRefineDecoder(nn.Module):
    """START-style two-stage heightmap refinement (rough -> refined).

    The rough heightmap (already produced by the MLP ``height_decoder``) is
    reshaped to a 2D grid and passed through a small conv encoder-decoder
    (U-Net-lite with one skip connection) that cleans up edges and flat
    surfaces. Following START (arXiv 2409.15692, sec. II-B), the rough map is
    supervised with MSE and the refined map with L1; the refinement does not
    add new information but sharpens edges that single-stage reconstruction
    blurs.

    Input/output are (B, H*W) flattened grids so the surrounding code keeps
    treating the heightmap as a flat vector; reshaping to (B,1,H,W) happens
    internally with the configured grid shape.
    """

    def __init__(self, grid_shape: tuple[int, int], hidden_channels: int = 16, activation: str = "elu"):
        super().__init__()
        self.grid_shape = grid_shape
        c = hidden_channels
        self.enc1 = nn.Sequential(nn.Conv2d(1, c, 3, padding=1), _activation(activation))
        self.enc2 = nn.Sequential(nn.Conv2d(c, c * 2, 3, padding=1), _activation(activation))
        # Decoder takes concat(enc2, enc1) as a single-level skip connection.
        self.dec = nn.Sequential(
            nn.Conv2d(c * 2 + c, c, 3, padding=1),
            _activation(activation),
            nn.Conv2d(c, 1, 3, padding=1),
        )

    def forward(self, rough_flat: torch.Tensor) -> torch.Tensor:
        b = rough_flat.shape[0]
        h, w = self.grid_shape
        x = rough_flat.reshape(b, 1, h, w)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        # Residual refinement: predict a correction added to the rough map.
        delta = self.dec(torch.cat((e2, e1), dim=1))
        refined = x + delta
        return refined.reshape(b, h * w)


class PIEEstimator(nn.Module):
    """PIE estimator skeleton with CReF-style cross-modal attention and online-memory GRU.

    Inputs:
        depth: (B, 2, 58, 87)
        proprioception_history: (B, 10, 47)

    The proprio history token queries visual tokens through cross-modal attention.
    The GRU receives one fused step per forward call and carries temporal memory in its hidden state.
    Outputs are returned as a dict with the estimator heads and reconstruction decoders.
    """

    def __init__(
        self,
        depth_channels: int = 2,
        depth_image_shape: tuple[int, int] = (58, 87),
        proprio_dim: int = 47,
        proprio_history_len: int = 10,
        depth_feature_map_shape: tuple[int, int] = (6, 9),
        depth_feature_dim: int = 128,
        proprio_feature_dim: int = 128,
        fusion_dim: int = 128,
        gru_hidden_dim: int = 256,
        latent_dim: int = 32,
        z_m_dim: int = 32,
        height_dim: int = 132,
        foot_height_dim: int = 4,
        next_proprio_dim: int = 47,
        activation: str = "elu",
        transformer_heads: int = 4,
        sample_latent_in_training: bool = False,
        height_decoder_hidden_dims: list[int] | tuple[int, ...] | None = None,
        # START-style heightmap refinement (B). When enabled, the rough height
        # map from ``height_decoder`` is refined by a conv U-Net-lite; both
        # rough and refined maps are returned for dual (MSE + L1) supervision.
        use_height_refine: bool = False,
        height_grid_shape: tuple[int, int] = (12, 11),
        height_refine_hidden_channels: int = 16,
        # START-style AdaSmpl (A). When enabled, the estimator can encode a
        # ground-truth heightmap (passed in at train time) into z_m instead of
        # the depth-reconstructed one, with a schedule controlled externally.
        # This only wires the *capability*; the sampling probability is applied
        # by the training loop, which calls forward(..., gt_heightmap=...).
        use_heightmap_encoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.depth_channels = depth_channels
        self.depth_image_shape = depth_image_shape
        self.proprio_dim = proprio_dim
        self.proprio_history_len = proprio_history_len
        self.proprio_history_dim = proprio_history_len * proprio_dim
        self.depth_feature_map_shape = depth_feature_map_shape
        self.fusion_dim = fusion_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.latent_dim = latent_dim
        self.z_m_dim = z_m_dim
        self.foot_height_dim = foot_height_dim
        self.sample_latent_in_training = sample_latent_in_training
        self.num_visual_tokens = depth_feature_map_shape[0] * depth_feature_map_shape[1]
        self.fused_dim = fusion_dim * 2

        self.depth_encoder = DepthCNNEncoder(depth_channels, depth_feature_dim, depth_feature_map_shape, activation)
        self.visual_token_projection = nn.Linear(depth_feature_dim, fusion_dim)
        self.proprio_encoder = _mlp(self.proprio_history_dim, [proprio_feature_dim], fusion_dim, activation)

        self.visual_position_embedding = nn.Parameter(torch.empty(1, self.num_visual_tokens, fusion_dim))
        self.visual_type_embedding = nn.Parameter(torch.empty(1, 1, fusion_dim))
        self.proprio_type_embedding = nn.Parameter(torch.empty(1, 1, fusion_dim))
        self._reset_token_embeddings()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=transformer_heads,
            batch_first=True,
        )
        self.depth_ln = nn.LayerNorm(fusion_dim)
        self.proprio_ln = nn.LayerNorm(fusion_dim)

        self.grf_ln = nn.LayerNorm(self.fused_dim)
        self.grf_fc1 = nn.Linear(self.fused_dim, self.fused_dim)
        self.grf_activation = nn.ELU()
        self.grf_fc2 = nn.Linear(self.fused_dim, self.fused_dim * 2)

        self.gru = nn.GRU(input_size=self.fused_dim, hidden_size=gru_hidden_dim, batch_first=True)
        self.recurrent_projector = nn.Linear(gru_hidden_dim, self.fused_dim)
        self.highway_gate = nn.Linear(self.fused_dim * 2, self.fused_dim)

        self.v_head = nn.Linear(self.fused_dim, 3)
        self.h_f_head = nn.Linear(self.fused_dim, foot_height_dim)
        self.z_m_head = nn.Linear(self.fused_dim, z_m_dim)
        self.z_mu_head = nn.Linear(self.fused_dim, latent_dim)
        self.z_logvar_head = nn.Linear(self.fused_dim, latent_dim)

        if height_decoder_hidden_dims is None:
            height_decoder_hidden_dims = [128]
        self.height_decoder = _mlp(z_m_dim, list(height_decoder_hidden_dims), height_dim, activation)

        # --- START-style options (both opt-in; default off = current behaviour) ---
        self.use_height_refine = bool(use_height_refine)
        self.use_heightmap_encoder = bool(use_heightmap_encoder)
        self.height_grid_shape = tuple(height_grid_shape)
        self.height_dim = height_dim
        if self.use_height_refine:
            assert self.height_grid_shape[0] * self.height_grid_shape[1] == height_dim, (
                f"height_grid_shape {self.height_grid_shape} does not match height_dim {height_dim}"
            )
            self.height_refine = HeightmapRefineDecoder(
                self.height_grid_shape, height_refine_hidden_channels, activation
            )
        else:
            self.height_refine = None
        if self.use_heightmap_encoder:
            # Encodes a heightmap grid (GT during AdaSmpl, else reconstructed)
            # into the z_m terrain code that the actor consumes. This mirrors
            # START's "policy takes the (reconstructed or GT) heightmap" design.
            self.heightmap_encoder = _mlp(height_dim, [256, 128], z_m_dim, activation)
        else:
            self.heightmap_encoder = None
        # Remove z_m from next_proprio decoder inputs so the VAE latent z must carry
        # the residual information required to reconstruct next proprioception.
        # Keeping z_m in the decoder gave the model a shortcut that caused posterior
        # collapse (z_mu ≈ 0) because z_m is not regularised by a KL prior.
        next_proprio_input_dim = latent_dim + 3 + foot_height_dim
        self.next_proprio_decoder = _mlp(next_proprio_input_dim, [128], next_proprio_dim, activation)

    def forward(
        self,
        depth: torch.Tensor | Mapping[str, torch.Tensor],
        proprioception_history: torch.Tensor | Mapping[str, torch.Tensor],
        hidden_state: torch.Tensor | None = None,
        gt_heightmap: torch.Tensor | None = None,
        adasmpl_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        depth = self._prepare_depth(depth)
        proprioception_history = self._prepare_proprioception_history(proprioception_history, depth.device)
        if hidden_state is not None:
            hidden_state = hidden_state.to(device=depth.device, dtype=depth.dtype)

        fused_step = self.encode_cross_modal_sequence(depth, proprioception_history)
        gru_output, next_hidden_state = self.gru(fused_step, hidden_state)
        z_rec = self.recurrent_projector(gru_output[:, -1])
        f = fused_step.squeeze(1)
        beta = torch.sigmoid(self.highway_gate(torch.cat((z_rec, f), dim=-1)))
        y = beta * z_rec + (1.0 - beta) * f

        v_hat = self.v_head(y)
        h_f_hat = self.h_f_head(y)
        z_b = self.z_m_head(y)
        z_mu = self.z_mu_head(y)
        z_logvar = self.z_logvar_head(y)
        z = self._latent_sample(z_mu, z_logvar)

        # --- Heightmap reconstruction (rough -> optional refine), START-style ---
        height_rough = self.height_decoder(z_b)
        if self.height_refine is not None:
            height_refined = self.height_refine(height_rough)
        else:
            height_refined = height_rough

        # --- Actor terrain code z_m ---
        if self.heightmap_encoder is not None:
            # START AdaSmpl: the actor consumes an ENCODING of the heightmap.
            # During training, with probability adasmpl_prob, encode the
            # ground-truth heightmap (clear features speed up learning); else
            # encode the network's own refined reconstruction. Safety: GT is
            # only ever supplied (gt_heightmap not None) with adasmpl_prob > 0,
            # and adasmpl_prob is only raised by the training loop - at
            # play/deployment it stays 0 (and real robots have no GT), so this
            # path then always encodes the reconstruction. We deliberately do
            # NOT gate on self.training so the rollout (eval-mode) actor can
            # also see GT early on, which is the whole point of AdaSmpl.
            policy_heightmap = height_refined
            if gt_heightmap is not None and adasmpl_prob > 0.0:
                gt_heightmap = gt_heightmap.to(device=height_refined.device, dtype=height_refined.dtype)
                # Per-sample Bernoulli mask so a fraction of the batch sees GT.
                use_gt = (torch.rand(height_refined.shape[0], 1, device=height_refined.device) < adasmpl_prob)
                policy_heightmap = torch.where(use_gt, gt_heightmap, height_refined)
            z_m = self.heightmap_encoder(policy_heightmap)
        else:
            # Legacy behaviour: z_m is the free head output; height decoded from it.
            z_m = z_b

        next_proprio_hat = self.next_proprio_decoder(torch.cat((z, v_hat, h_f_hat), dim=-1))

        return {
            "v_hat": v_hat,
            "h_f_hat": h_f_hat,
            "z_m": z_m,
            "z": z,
            "z_t": z,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "height_hat": height_refined,
            "height_hat_rough": height_rough,
            "next_proprio_hat": next_proprio_hat,
            "rnn_hidden": next_hidden_state,
        }

    def forward_obs_dict(
        self,
        obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
        hidden_state: torch.Tensor | None = None,
        depth_key: str = "depth_camera",
        proprioception_history_key: str = "proprioception_history",
        gt_heightmap: torch.Tensor | None = None,
        adasmpl_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        return self(
            obs_dict[depth_key],
            obs_dict[proprioception_history_key],
            hidden_state,
            gt_heightmap=gt_heightmap,
            adasmpl_prob=adasmpl_prob,
        )

    def initial_hidden(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)

    def encode_cross_modal_sequence(
        self,
        depth: torch.Tensor | Mapping[str, torch.Tensor],
        proprioception_history: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        depth = self._prepare_depth(depth)
        proprioception_history = self._prepare_proprioception_history(proprioception_history, depth.device)

        batch_size, history_len, _ = proprioception_history.shape
        if history_len != self.proprio_history_len:
            raise ValueError(
                f"Proprioception history length must be {self.proprio_history_len}, got {history_len}"
            )

        visual_tokens = self._tokenize_depth(depth)
        proprio_token = self._tokenize_proprioception(proprioception_history, batch_size)
        visual_tokens_norm = self.depth_ln(visual_tokens)
        attn_out, _ = self.cross_attention(
            query=self.proprio_ln(proprio_token),
            key=visual_tokens_norm,
            value=visual_tokens_norm,
        )
        x = torch.cat((proprio_token.squeeze(1), attn_out.squeeze(1)), dim=-1)
        f = self._gated_residual_fusion(x)
        return f.unsqueeze(1)

    def _tokenize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth_feature_map = self.depth_encoder(depth)
        visual_tokens = depth_feature_map.flatten(2).transpose(1, 2)
        visual_tokens = self.visual_token_projection(visual_tokens)
        return visual_tokens + self.visual_position_embedding + self.visual_type_embedding

    def _tokenize_proprioception(self, proprioception_history: torch.Tensor, batch_size: int) -> torch.Tensor:
        proprio_token = self.proprio_encoder(proprioception_history.reshape(batch_size, self.proprio_history_dim))
        return proprio_token.unsqueeze(1) + self.proprio_type_embedding

    def _gated_residual_fusion(self, x: torch.Tensor) -> torch.Tensor:
        h = self.grf_activation(self.grf_fc1(self.grf_ln(x)))
        content, gate = self.grf_fc2(h).chunk(2, dim=-1)
        return x + content * torch.sigmoid(gate)

    def _latent_sample(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.sample_latent_in_training):
            return z_mu
        std = torch.exp(0.5 * z_logvar)
        return z_mu + torch.randn_like(std) * std

    def _prepare_depth(self, depth: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        depth = self._unwrap_single_term(depth)
        device = next(self.parameters()).device
        depth = depth.to(device=device, dtype=torch.float32)

        if depth.ndim == 5 and depth.shape[1] == 1:
            depth = depth.squeeze(1)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.ndim == 2:
            expected_flat_dim = self.depth_channels * self.depth_image_shape[0] * self.depth_image_shape[1]
            if depth.shape[-1] != expected_flat_dim:
                raise ValueError(f"Flattened depth must have last dim {expected_flat_dim}, got {depth.shape[-1]}")
            depth = depth.reshape(depth.shape[0], self.depth_channels, *self.depth_image_shape)
        if depth.ndim != 4:
            raise ValueError(f"Depth must be shaped (B, C, H, W), got {tuple(depth.shape)}")
        if depth.shape[1] != self.depth_channels:
            raise ValueError(f"Depth channel dim must be {self.depth_channels}, got {depth.shape[1]}")
        return depth

    def _prepare_proprioception_history(
        self,
        proprioception_history: torch.Tensor | Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        proprioception_history = self._unwrap_single_term(proprioception_history)
        proprioception_history = proprioception_history.to(device=device, dtype=torch.float32)

        if proprioception_history.ndim == 4 and proprioception_history.shape[1] == 1:
            proprioception_history = proprioception_history.squeeze(1)
        if proprioception_history.ndim == 2:
            if proprioception_history.shape[-1] != self.proprio_history_dim:
                raise ValueError(
                    "Flattened proprioception history must have last dim "
                    f"{self.proprio_history_dim}, "
                    f"got {proprioception_history.shape[-1]}"
                )
            proprioception_history = proprioception_history.reshape(
                proprioception_history.shape[0], self.proprio_history_len, self.proprio_dim
            )
        if proprioception_history.ndim != 3:
            raise ValueError(
                "Proprioception history must be shaped (B, T, D), "
                f"got {tuple(proprioception_history.shape)}"
            )
        if proprioception_history.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"Proprioception feature dim must be {self.proprio_dim}, "
                f"got {proprioception_history.shape[-1]}"
            )
        return proprioception_history

    @staticmethod
    def _unwrap_single_term(value: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(value, Mapping):
            if len(value) != 1:
                raise ValueError(f"Expected a single observation term, got keys: {list(value.keys())}")
            value = next(iter(value.values()))
        return value

    def _reset_token_embeddings(self) -> None:
        nn.init.normal_(self.visual_position_embedding, std=0.02)
        nn.init.normal_(self.visual_type_embedding, std=0.02)
        nn.init.normal_(self.proprio_type_embedding, std=0.02)
