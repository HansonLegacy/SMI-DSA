import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_encoder import ResNetPyramid


def _make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


def _sincos_1d(values, dim: int):
    if dim <= 0:
        raise ValueError("Embedding dimension must be positive.")

    out_dtype = values.dtype
    values = values.float()
    half_dim = dim // 2
    if half_dim == 0:
        return values.unsqueeze(-1).to(dtype=out_dtype)

    denom = max(half_dim - 1, 1)
    freq = torch.exp(
        torch.arange(half_dim, device=values.device, dtype=values.dtype)
        * (-math.log(10000.0) / denom)
    )
    angles = values.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb.to(dtype=out_dtype)


def _sincos_2d_grid(height: int, width: int, dim: int, device, dtype):
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy = y.view(height, 1).expand(height, width).reshape(-1)
    xx = x.view(1, width).expand(height, width).reshape(-1)
    y_dim = dim // 2
    x_dim = dim - y_dim
    return torch.cat([_sincos_1d(yy, y_dim), _sincos_1d(xx, x_dim)], dim=-1)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            _make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            _make_group_norm(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class BiasedCrossAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")

        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key_value,
        key_padding_mask=None,
        attn_bias=None,
        return_weights: bool = False,
    ):
        batch, num_query, _ = query.shape
        num_key = key_value.shape[1]

        q = self.q_proj(query).view(batch, num_query, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch, num_key, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch, num_key, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            scores = scores + attn_bias.unsqueeze(1).to(device=scores.device, dtype=scores.dtype)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :].bool(), -1.0e4)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, num_query, self.channels)
        out = self.out_proj(out)

        if return_weights:
            return out, attn.mean(dim=1)
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_channels = max(channels, int(channels * mlp_ratio))
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        self.attn = BiasedCrossAttention(channels, num_heads=num_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, query, context, key_padding_mask=None, attn_bias=None, return_weights: bool = False):
        attn_result = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            key_padding_mask=key_padding_mask,
            attn_bias=attn_bias,
            return_weights=return_weights,
        )
        if return_weights:
            attn_out, weights = attn_result
        else:
            attn_out = attn_result
            weights = None

        query = query + attn_out
        query = query + self.ffn(self.ffn_norm(query))
        if return_weights:
            return query, weights
        return query


class AnchorQueryBuilder(nn.Module):
    def __init__(
        self,
        image_channels: int,
        channels: int,
        pos_embed_dim: int = 32,
        tau_is_relative: bool = True,
    ):
        super().__init__()
        self.pos_embed_dim = int(pos_embed_dim)
        self.tau_is_relative = bool(tau_is_relative)
        self.image_encoder = nn.Sequential(
            ConvNormAct(image_channels, channels),
            ResidualConvBlock(channels),
        )
        query_in_channels = channels * 2 + self.pos_embed_dim * 3
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(query_in_channels),
            nn.Linear(query_in_channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    @staticmethod
    def _gather_anchors(video, anchor_indices):
        batch = video.shape[0]
        batch_idx = torch.arange(batch, device=video.device)
        return video[batch_idx, anchor_indices[:, 0]], video[batch_idx, anchor_indices[:, 1]]

    def forward(self, video, anchor_indices, tau_list, seq_len: int):
        batch = video.shape[0]
        anchor0, anchor1 = self._gather_anchors(video, anchor_indices)
        desc0 = self.image_encoder(anchor0).mean(dim=(-1, -2))
        desc1 = self.image_encoder(anchor1).mean(dim=(-1, -2))

        denom = max(seq_len - 1, 1)
        anchor0_pos = anchor_indices[:, 0].to(device=video.device, dtype=video.dtype) / denom
        anchor1_pos = anchor_indices[:, 1].to(device=video.device, dtype=video.dtype) / denom
        if self.tau_is_relative:
            tau_pos = anchor0_pos.unsqueeze(1) + tau_list * (
                anchor1_pos - anchor0_pos
            ).unsqueeze(1)
        else:
            tau_pos = tau_list / denom

        num_targets = tau_list.shape[1]
        desc0 = desc0.unsqueeze(1).expand(-1, num_targets, -1)
        desc1 = desc1.unsqueeze(1).expand(-1, num_targets, -1)
        anchor0_emb = _sincos_1d(anchor0_pos, self.pos_embed_dim).unsqueeze(1).expand(
            -1, num_targets, -1
        )
        anchor1_emb = _sincos_1d(anchor1_pos, self.pos_embed_dim).unsqueeze(1).expand(
            -1, num_targets, -1
        )
        tau_emb = _sincos_1d(tau_pos, self.pos_embed_dim)
        return self.query_mlp(torch.cat([desc0, desc1, anchor0_emb, anchor1_emb, tau_emb], dim=-1))


class LocalStreamMixer(nn.Module):
    metadata_channels = 6

    def __init__(
        self,
        image_channels: int,
        channels: int,
        temporal_sigma: float = 2.0,
        anchor_prior_strength: float = 0.25,
        diff_mode: str = "signed",
    ):
        super().__init__()
        self.temporal_sigma = float(temporal_sigma)
        self.anchor_prior_strength = float(anchor_prior_strength)
        self.diff_mode = str(diff_mode)

        self.raw_proj = nn.Sequential(
            ConvNormAct(image_channels, channels),
            ResidualConvBlock(channels),
        )
        self.frame_adapter = nn.Sequential(
            ConvNormAct(channels * 2 + self.metadata_channels, channels),
            ResidualConvBlock(channels),
        )
        self.context_fusion = nn.Sequential(
            ConvNormAct(channels * 4 + 1, channels),
            ResidualConvBlock(channels),
        )

    @staticmethod
    def _metadata_maps(metadata, height: int, width: int):
        return metadata.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, height, width)

    @staticmethod
    def _gather_local_anchors(feat_seq, anchor_local_indices):
        batch = feat_seq.shape[0]
        batch_idx = torch.arange(batch, device=feat_seq.device)
        return feat_seq[batch_idx, anchor_local_indices[:, 0]], feat_seq[batch_idx, anchor_local_indices[:, 1]]

    def _build_frame_metadata(self, near_indices, near_valid, anchor_local_indices, delta, seq_len: int):
        batch, num_near = near_indices.shape
        dtype = delta.dtype
        device = near_indices.device
        denom = max(seq_len - 1, 1)

        near_pos = near_indices.to(device=device, dtype=dtype)
        frame_norm = near_pos / denom
        near_delta = torch.gather(delta, 1, near_indices.clamp(min=0, max=max(seq_len - 1, 0)))
        delta_norm = near_delta.to(dtype=dtype) / denom
        log_delta = torch.log1p(near_delta.to(dtype=dtype))

        anchor0_flag = torch.zeros((batch, num_near), device=device, dtype=dtype)
        anchor1_flag = torch.zeros((batch, num_near), device=device, dtype=dtype)
        anchor0_flag.scatter_(1, anchor_local_indices[:, 0:1], 1.0)
        anchor1_flag.scatter_(1, anchor_local_indices[:, 1:2], 1.0)
        valid = near_valid.to(device=device, dtype=dtype)

        return torch.stack(
            [frame_norm, delta_norm, log_delta, anchor0_flag, anchor1_flag, valid],
            dim=-1,
        ) * valid.unsqueeze(-1)

    def _build_temporal_weights(self, near_indices, near_valid, anchor_local_indices, target_indices):
        batch, num_near = near_indices.shape
        num_targets = target_indices.shape[1]
        dtype = target_indices.dtype
        device = target_indices.device

        near_pos = near_indices.to(device=device, dtype=dtype)
        temporal_delta = near_pos.unsqueeze(1) - target_indices.unsqueeze(-1)
        sigma = max(self.temporal_sigma, 1.0e-6)
        weights = torch.exp(-0.5 * (temporal_delta / sigma) ** 2)
        weights = weights * near_valid.to(device=device, dtype=dtype).unsqueeze(1)

        anchor_boost = torch.zeros((batch, num_near), device=device, dtype=dtype)
        anchor_boost.scatter_(1, anchor_local_indices[:, 0:1], 1.0)
        anchor_boost.scatter_(1, anchor_local_indices[:, 1:2], 1.0)
        weights = weights + self.anchor_prior_strength * anchor_boost.unsqueeze(1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return weights

    def forward(
        self,
        local_clip,
        feat_seq,
        near_indices,
        near_valid,
        anchor_local_indices,
        delta,
        target_indices,
        seq_len: int,
    ):
        batch, num_near, image_channels, image_h, image_w = local_clip.shape
        _, _, channels, level_h, level_w = feat_seq.shape

        raw = F.interpolate(
            local_clip.reshape(batch * num_near, image_channels, image_h, image_w),
            size=(level_h, level_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        raw = self.raw_proj(raw).reshape(batch, num_near, channels, level_h, level_w)

        frame_metadata = self._build_frame_metadata(
            near_indices,
            near_valid,
            anchor_local_indices,
            delta,
            seq_len,
        )
        metadata_maps = self._metadata_maps(frame_metadata, level_h, level_w).to(
            device=feat_seq.device,
            dtype=feat_seq.dtype,
        )
        adapted = torch.cat([feat_seq, raw.to(dtype=feat_seq.dtype), metadata_maps], dim=2)
        adapted = self.frame_adapter(
            adapted.reshape(
                batch * num_near,
                channels * 2 + self.metadata_channels,
                level_h,
                level_w,
            )
        ).reshape(batch, num_near, channels, level_h, level_w)
        adapted = adapted * near_valid.to(device=adapted.device, dtype=adapted.dtype).view(
            batch, num_near, 1, 1, 1
        )

        weights = self._build_temporal_weights(
            near_indices,
            near_valid,
            anchor_local_indices,
            target_indices.to(device=feat_seq.device, dtype=feat_seq.dtype),
        )
        context = torch.sum(
            adapted.unsqueeze(1)
            * weights.to(device=adapted.device, dtype=adapted.dtype).view(batch, -1, num_near, 1, 1, 1),
            dim=2,
        )

        anchor0, anchor1 = self._gather_local_anchors(adapted, anchor_local_indices)
        anchor0 = anchor0.unsqueeze(1).expand(-1, target_indices.shape[1], -1, -1, -1)
        anchor1 = anchor1.unsqueeze(1).expand(-1, target_indices.shape[1], -1, -1, -1)
        pair_delta = anchor1 - anchor0
        if self.diff_mode == "abs":
            pair_delta = torch.abs(pair_delta)

        denom = max(seq_len - 1, 1)
        target_map = (target_indices.to(device=feat_seq.device, dtype=feat_seq.dtype) / denom)
        target_map = target_map.view(batch, target_indices.shape[1], 1, 1, 1).expand(
            -1, -1, 1, level_h, level_w
        )

        fusion_in = torch.cat([anchor0, anchor1, pair_delta, context, target_map], dim=2)
        return self.context_fusion(
            fusion_in.reshape(batch * target_indices.shape[1], channels * 4 + 1, level_h, level_w)
        ).reshape(batch, target_indices.shape[1], channels, level_h, level_w)


class ScaleAwareTokenProjector(nn.Module):
    metadata_channels = 7

    def __init__(self, channels: int, pos_embed_dim: int = 32):
        super().__init__()
        self.channels = int(channels)
        self.pos_embed_dim = int(pos_embed_dim)
        self.token_norm = nn.LayerNorm(channels)
        self.metadata_mlp = nn.Sequential(
            nn.Linear(self.metadata_channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.grid_pos_mlp = nn.Sequential(
            nn.Linear(pos_embed_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, feat, grid_size: int, metadata):
        grid_size = max(1, int(grid_size))
        pooled = F.adaptive_avg_pool2d(feat, (grid_size, grid_size))
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens)

        metadata_tokens = self.metadata_mlp(metadata).unsqueeze(1)
        grid_pos = _sincos_2d_grid(
            grid_size,
            grid_size,
            self.pos_embed_dim,
            feat.device,
            feat.dtype,
        )
        grid_pos = self.grid_pos_mlp(grid_pos).unsqueeze(0)
        return tokens + metadata_tokens + grid_pos


class GlobalTokenMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        lambda_distance: float = 0.75,
        mu_resolution: float = 0.25,
        learnable_bias_strength: bool = False,
    ):
        super().__init__()
        self.read = CrossAttentionBlock(
            channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.lambda_distance = nn.Parameter(
            torch.tensor(float(lambda_distance)),
            requires_grad=bool(learnable_bias_strength),
        )
        self.mu_resolution = nn.Parameter(
            torch.tensor(float(mu_resolution)),
            requires_grad=bool(learnable_bias_strength),
        )

    def _build_bias(self, query, token_delta, token_resolution):
        distance_penalty = self.lambda_distance.clamp_min(0.0) * torch.log1p(token_delta)
        resolution_penalty = self.mu_resolution.clamp_min(0.0) * torch.log(
            token_resolution.clamp_min(1.0)
        )
        bias = -(distance_penalty + resolution_penalty)
        return bias.unsqueeze(1).expand(-1, query.shape[1], -1)

    def forward(
        self,
        query,
        tokens,
        token_valid,
        token_delta,
        token_resolution,
        return_attention: bool = False,
    ):
        if tokens.shape[1] == 0:
            zeros = query.new_zeros(query.shape)
            if return_attention:
                return zeros, query.new_zeros(query.shape[0], query.shape[1], 0)
            return zeros

        key_padding_mask = ~token_valid.bool()
        attn_bias = self._build_bias(
            query,
            token_delta.to(device=query.device, dtype=query.dtype),
            token_resolution.to(device=query.device, dtype=query.dtype),
        )
        has_tokens = token_valid.bool().any(dim=1).to(device=query.device, dtype=query.dtype)
        result = self.read(
            query,
            tokens,
            key_padding_mask=key_padding_mask,
            attn_bias=attn_bias,
            return_weights=return_attention,
        )
        if return_attention:
            out, weights = result
            out = out * has_tokens.view(-1, 1, 1)
            weights = weights * has_tokens.view(-1, 1, 1)
            return out, weights
        return result * has_tokens.view(-1, 1, 1)


class TokenToMapModulator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, local_map, global_token):
        batch, num_targets, channels, _, _ = local_map.shape
        gamma, beta = self.modulation(global_token).chunk(2, dim=-1)
        gamma = gamma.view(batch, num_targets, channels, 1, 1)
        beta = beta.view(batch, num_targets, channels, 1, 1)
        return local_map + gamma * local_map + beta


class ADAMS(nn.Module):
    """
    Anatomical-Dynamic Anchor-aware Multi-scale Sequence Context Encoder.

    ADAMS keeps the original per-frame MFE/CFE modules intact and only changes
    how sequence context is scheduled, buffered, mixed, and exposed:
    - MFE is treated as the Dynamic encoder.
    - CFE is treated as the Anatomical encoder.
    - Near anchor-centered frames keep full multi-scale maps.
    - Mid/far frames are immediately compressed into scale-aware tokens.
    - Global tokens modulate local maps instead of generating spatial detail.

    Forward input:
        video:          [B, T, C, H, W]
        anchor_indices: [B, 2] or [2]
        tau_list:       [B, S], [B], [S], or scalar. By default tau is relative
                        to the anchor interval, so 0.5 is the midpoint.

    Output:
        dynamic_context_pyr:    list of [B, S, C_l, H_l, W_l] tensors
        anatomical_context_pyr: list of [B, S, C_l, H_l, W_l] tensors
        anchor_dynamic_pyr:     list of [B, 2, C_l, H_l, W_l] original-MFE anchor tensors
        anchor_anatomical_pyr:  list of [B, 2, C_l, H_l, W_l] original-CFE anchor tensors

    If squeeze_single_tau=True and S=1, the output pyramid tensors are squeezed
    to [B, C_l, H_l, W_l] for easier downstream prototyping.
    """

    def __init__(
        self,
        feat_channels: int = 32,
        image_channels: int = 3,
        stem_image_channels: Optional[int] = None,
        dynamic_encoder: Optional[nn.Module] = None,
        anatomical_encoder: Optional[nn.Module] = None,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        resolution_schedule: Sequence[Tuple[int, int]] = ((1, 1), (5, 2), (13, 4), (29, 8)),
        far_resolution_factor: int = 8,
        min_stem_size: int = 8,
        near_window: int = 1,
        base_token_grid: int = 8,
        temporal_sigma: float = 2.0,
        anchor_prior_strength: float = 0.25,
        pos_embed_dim: int = 32,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        lambda_dynamic: float = 0.75,
        lambda_anatomical: float = 0.25,
        mu_dynamic: float = 0.25,
        mu_anatomical: float = 0.25,
        learnable_bias_strength: bool = False,
        tau_is_relative: bool = True,
        squeeze_single_tau: bool = False,
    ):
        super().__init__()
        self.feat_channels = int(feat_channels)
        self.image_channels = int(image_channels)
        if stem_image_channels is None:
            stem_image_channels = 3 if dynamic_encoder is None or anatomical_encoder is None else image_channels
        self.stem_image_channels = int(stem_image_channels)
        self.channel_multipliers = tuple(int(multiplier) for multiplier in channel_multipliers)
        self.channel_list = [self.feat_channels * multiplier for multiplier in self.channel_multipliers]
        self.num_levels = len(self.channel_list)
        self.resolution_schedule = tuple((int(delta), int(factor)) for delta, factor in resolution_schedule)
        self.far_resolution_factor = int(far_resolution_factor)
        self.min_stem_size = int(min_stem_size)
        self.near_window = int(near_window)
        self.base_token_grid = int(base_token_grid)
        self.tau_is_relative = bool(tau_is_relative)
        self.squeeze_single_tau = bool(squeeze_single_tau)

        if self.num_levels <= 0:
            raise ValueError("ADAMS requires at least one pyramid level.")
        if self.near_window < 0:
            raise ValueError(f"near_window must be >= 0, got {near_window}.")
        if self.base_token_grid < 1:
            raise ValueError(f"base_token_grid must be >= 1, got {base_token_grid}.")
        if self.min_stem_size < 1:
            raise ValueError(f"min_stem_size must be >= 1, got {min_stem_size}.")

        if dynamic_encoder is None:
            self.dynamic_encoder = ResNetPyramid(feat_channels)
        else:
            object.__setattr__(self, "dynamic_encoder", dynamic_encoder)
        if anatomical_encoder is None:
            self.anatomical_encoder = ResNetPyramid(feat_channels)
        else:
            object.__setattr__(self, "anatomical_encoder", anatomical_encoder)

        self.query_builders = nn.ModuleList(
            [
                AnchorQueryBuilder(
                    image_channels=image_channels,
                    channels=channels,
                    pos_embed_dim=pos_embed_dim,
                    tau_is_relative=tau_is_relative,
                )
                for channels in self.channel_list
            ]
        )
        self.dynamic_local_mixers = nn.ModuleList(
            [
                LocalStreamMixer(
                    image_channels,
                    channels,
                    temporal_sigma=temporal_sigma,
                    anchor_prior_strength=anchor_prior_strength,
                    diff_mode="signed",
                )
                for channels in self.channel_list
            ]
        )
        self.anatomical_local_mixers = nn.ModuleList(
            [
                LocalStreamMixer(
                    image_channels,
                    channels,
                    temporal_sigma=temporal_sigma,
                    anchor_prior_strength=anchor_prior_strength,
                    diff_mode="abs",
                )
                for channels in self.channel_list
            ]
        )
        self.dynamic_token_projectors = nn.ModuleList(
            [ScaleAwareTokenProjector(channels, pos_embed_dim=pos_embed_dim) for channels in self.channel_list]
        )
        self.anatomical_token_projectors = nn.ModuleList(
            [ScaleAwareTokenProjector(channels, pos_embed_dim=pos_embed_dim) for channels in self.channel_list]
        )
        self.dynamic_global_mixers = nn.ModuleList(
            [
                GlobalTokenMixer(
                    channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    lambda_distance=lambda_dynamic,
                    mu_resolution=mu_dynamic,
                    learnable_bias_strength=learnable_bias_strength,
                )
                for channels in self.channel_list
            ]
        )
        self.anatomical_global_mixers = nn.ModuleList(
            [
                GlobalTokenMixer(
                    channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    lambda_distance=lambda_anatomical,
                    mu_resolution=mu_anatomical,
                    learnable_bias_strength=learnable_bias_strength,
                )
                for channels in self.channel_list
            ]
        )
        self.dynamic_modulators = nn.ModuleList(
            [TokenToMapModulator(channels) for channels in self.channel_list]
        )
        self.anatomical_modulators = nn.ModuleList(
            [TokenToMapModulator(channels) for channels in self.channel_list]
        )

    @staticmethod
    def _normalize_anchor_indices(anchor_indices, batch_size: int, seq_len: int, device):
        if anchor_indices is None:
            raise ValueError("ADAMS requires anchor_indices.")
        anchor_indices = anchor_indices.to(device=device, dtype=torch.long)
        if anchor_indices.dim() == 1:
            anchor_indices = anchor_indices.view(1, -1).expand(batch_size, -1)
        if anchor_indices.shape[1] < 2:
            raise ValueError(
                f"anchor_indices must contain at least two indices, got {tuple(anchor_indices.shape)}."
            )
        return anchor_indices[:, :2].clamp(min=0, max=max(seq_len - 1, 0))

    @staticmethod
    def _normalize_tau_list(tau_list, batch_size: int, device, dtype):
        if tau_list is None:
            tau_list = torch.tensor(0.5, device=device, dtype=dtype)
        elif not torch.is_tensor(tau_list):
            tau_list = torch.tensor(tau_list, device=device, dtype=dtype)
        else:
            tau_list = tau_list.to(device=device, dtype=dtype)

        if tau_list.dim() == 0:
            return tau_list.view(1, 1).expand(batch_size, 1)
        if tau_list.dim() == 1:
            if tau_list.numel() == batch_size:
                return tau_list.view(batch_size, 1)
            return tau_list.view(1, -1).expand(batch_size, -1)
        if tau_list.dim() == 2:
            if tau_list.shape[0] == 1 and batch_size > 1:
                return tau_list.expand(batch_size, -1)
            if tau_list.shape[0] != batch_size:
                raise ValueError(f"tau_list batch mismatch: expected {batch_size}, got {tau_list.shape[0]}.")
            return tau_list
        raise ValueError(f"tau_list must be scalar, 1D, or 2D, got shape {tuple(tau_list.shape)}.")

    @staticmethod
    def _default_frame_mask(batch_size: int, seq_len: int, device, dtype):
        return torch.ones((batch_size, seq_len), device=device, dtype=dtype)

    def _build_delta(self, seq_len: int, anchor_indices, dtype):
        device = anchor_indices.device
        positions = torch.arange(seq_len, device=device).view(1, seq_len)
        delta0 = torch.abs(positions - anchor_indices[:, 0:1])
        delta1 = torch.abs(positions - anchor_indices[:, 1:2])
        return torch.minimum(delta0, delta1).to(dtype=dtype)

    def _build_resolution_factors(self, delta):
        factors = torch.full_like(delta, self.far_resolution_factor, dtype=torch.long)
        for max_delta, factor in reversed(self.resolution_schedule):
            factors = torch.where(delta <= max_delta, torch.full_like(factors, int(factor)), factors)
        return factors

    def _clamp_resolution_factors_for_stem(self, resolution_factors, height: int, width: int):
        max_safe_factor = max(1, min(int(height), int(width)) // self.min_stem_size)
        if int(resolution_factors.max().item()) <= max_safe_factor:
            return resolution_factors

        allowed_factors = sorted(
            {1, self.far_resolution_factor}
            | {int(factor) for _, factor in self.resolution_schedule}
        )
        safe_factors = [factor for factor in allowed_factors if factor <= max_safe_factor]
        safe_factor = max(safe_factors) if safe_factors else 1
        return torch.where(
            resolution_factors > max_safe_factor,
            torch.full_like(resolution_factors, int(safe_factor)),
            resolution_factors,
        )

    def _build_target_indices(self, anchor_indices, tau_list, dtype):
        anchor0 = anchor_indices[:, 0].to(dtype=dtype).view(-1, 1)
        anchor1 = anchor_indices[:, 1].to(dtype=dtype).view(-1, 1)
        if self.tau_is_relative:
            return anchor0 + tau_list * (anchor1 - anchor0)
        return tau_list

    def _build_near_indices(self, near_mask, anchor_indices):
        batch, seq_len = near_mask.shape
        device = near_mask.device
        order = torch.arange(seq_len, device=device).view(1, seq_len).expand(batch, -1)
        invalid_order = order + seq_len
        sort_key = torch.where(near_mask, order, invalid_order)
        near_count = int(near_mask.sum(dim=1).max().item())
        near_count = max(near_count, 2)
        near_indices = torch.argsort(sort_key, dim=1)[:, :near_count]
        near_valid = torch.gather(near_mask, 1, near_indices)

        anchor0_match = near_indices == anchor_indices[:, 0:1]
        anchor1_match = near_indices == anchor_indices[:, 1:2]
        anchor0_local = anchor0_match.float().argmax(dim=1)
        anchor1_local = anchor1_match.float().argmax(dim=1)
        anchor_local = torch.stack([anchor0_local, anchor1_local], dim=1).long()
        return near_indices, near_valid, anchor_local

    @staticmethod
    def _gather_video_frames(video, indices):
        batch = video.shape[0]
        batch_idx = torch.arange(batch, device=video.device).view(batch, 1).expand_as(indices)
        return video[batch_idx, indices]

    @staticmethod
    def _downsample_frames(frames, factor: int):
        factor = int(factor)
        if factor == 1:
            return frames
        height, width = frames.shape[-2:]
        target_h = max(1, height // factor)
        target_w = max(1, width // factor)
        return F.interpolate(
            frames,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def _prepare_stem_frames(self, frames):
        if frames.shape[1] == self.stem_image_channels:
            return frames
        if frames.shape[1] == 1 and self.stem_image_channels == 3:
            return frames.repeat(1, 3, 1, 1)
        raise ValueError(
            f"ADAMS stem expects {self.stem_image_channels} channels, "
            f"but received {frames.shape[1]}. Pass compatible MFE/CFE encoders or set stem_image_channels."
        )

    def _token_grid_for_resolution(self, resolution_factor: int):
        return max(1, self.base_token_grid // max(int(resolution_factor), 1))

    def _build_token_metadata(
        self,
        flat_indices,
        batch_ids,
        time_ids,
        delta,
        resolution_factors,
        frame_mask,
        anchor_indices,
        seq_len,
    ):
        dtype = delta.dtype
        device = delta.device
        denom = max(seq_len - 1, 1)
        batch_delta = delta.reshape(-1)[flat_indices].to(dtype=dtype)
        batch_resolution = resolution_factors.reshape(-1)[flat_indices].to(dtype=dtype)
        valid = frame_mask.reshape(-1)[flat_indices].to(dtype=dtype)

        pos_norm = time_ids.to(device=device, dtype=dtype) / denom
        delta_norm = batch_delta / denom
        log_delta = torch.log1p(batch_delta)
        log_resolution = torch.log(batch_resolution.clamp_min(1.0))
        anchor0 = anchor_indices[batch_ids, 0]
        anchor1 = anchor_indices[batch_ids, 1]
        left_anchor = torch.minimum(anchor0, anchor1)
        right_anchor = torch.maximum(anchor0, anchor1)
        left_side = (time_ids <= left_anchor).to(device=device, dtype=dtype)
        right_side = (time_ids >= right_anchor).to(device=device, dtype=dtype)

        metadata = torch.stack(
            [
                pos_norm,
                delta_norm,
                log_delta,
                log_resolution,
                left_side,
                right_side,
                valid,
            ],
            dim=-1,
        )
        return metadata, batch_delta, batch_resolution

    def _pack_token_pieces(self, pieces, batch_size: int, channels: int, device, dtype):
        if not pieces:
            empty_tokens = torch.zeros((batch_size, 0, channels), device=device, dtype=dtype)
            empty_valid = torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
            empty_scalar = torch.zeros((batch_size, 0), device=device, dtype=dtype)
            return empty_tokens, empty_valid, empty_scalar, empty_scalar

        per_batch_tokens: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        per_batch_delta: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        per_batch_resolution: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        for tokens, batch_ids, frame_delta, frame_resolution in pieces:
            num_frame_tokens = tokens.shape[1]
            token_delta = frame_delta.unsqueeze(1).expand(-1, num_frame_tokens).reshape(-1)
            token_resolution = frame_resolution.unsqueeze(1).expand(-1, num_frame_tokens).reshape(-1)
            flat_tokens = tokens.reshape(-1, tokens.shape[-1])
            flat_batch = batch_ids.unsqueeze(1).expand(-1, num_frame_tokens).reshape(-1)
            for batch_idx in range(batch_size):
                keep = flat_batch == batch_idx
                if bool(keep.any().item()):
                    per_batch_tokens[batch_idx].append(flat_tokens[keep])
                    per_batch_delta[batch_idx].append(token_delta[keep])
                    per_batch_resolution[batch_idx].append(token_resolution[keep])

        lengths = [
            sum(piece.shape[0] for piece in batch_pieces)
            for batch_pieces in per_batch_tokens
        ]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            empty_tokens = torch.zeros((batch_size, 0, channels), device=device, dtype=dtype)
            empty_valid = torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
            empty_scalar = torch.zeros((batch_size, 0), device=device, dtype=dtype)
            return empty_tokens, empty_valid, empty_scalar, empty_scalar

        packed_tokens = torch.zeros((batch_size, max_len, channels), device=device, dtype=dtype)
        packed_valid = torch.zeros((batch_size, max_len), device=device, dtype=torch.bool)
        packed_delta = torch.zeros((batch_size, max_len), device=device, dtype=dtype)
        packed_resolution = torch.ones((batch_size, max_len), device=device, dtype=dtype)

        for batch_idx, length in enumerate(lengths):
            if length == 0:
                continue
            tokens_b = torch.cat(per_batch_tokens[batch_idx], dim=0).to(device=device, dtype=dtype)
            delta_b = torch.cat(per_batch_delta[batch_idx], dim=0).to(device=device, dtype=dtype)
            resolution_b = torch.cat(per_batch_resolution[batch_idx], dim=0).to(device=device, dtype=dtype)
            packed_tokens[batch_idx, :length] = tokens_b
            packed_delta[batch_idx, :length] = delta_b
            packed_resolution[batch_idx, :length] = resolution_b
            packed_valid[batch_idx, :length] = True

        return packed_tokens, packed_valid, packed_delta, packed_resolution

    def _run_stem_grouped(
        self,
        video,
        delta,
        resolution_factors,
        frame_mask,
        near_mask,
        near_indices,
        anchor_indices,
    ):
        batch, seq_len, channels, height, width = video.shape
        num_near = near_indices.shape[1]
        flat_video = video.reshape(batch * seq_len, channels, height, width)
        flat_resolution = resolution_factors.reshape(-1)
        flat_valid = frame_mask.reshape(-1) > 0
        flat_near = near_mask.reshape(-1)
        flat_near_indices = (
            torch.arange(batch, device=video.device).view(batch, 1) * seq_len
            + near_indices
        )
        near_lookup = torch.full(
            (batch * seq_len,),
            -1,
            device=video.device,
            dtype=torch.long,
        )
        near_lookup[flat_near_indices.reshape(-1)] = torch.arange(
            batch * num_near,
            device=video.device,
            dtype=torch.long,
        )

        local_dynamic_maps = None
        local_anatomical_maps = None
        dynamic_token_pieces = [[] for _ in range(self.num_levels)]
        anatomical_token_pieces = [[] for _ in range(self.num_levels)]

        unique_factors = sorted({int(factor) for factor in resolution_factors.unique().detach().cpu().tolist()})
        for factor in unique_factors:
            selected = (flat_resolution == factor) & flat_valid
            if not selected.any():
                continue

            flat_indices = torch.nonzero(selected, as_tuple=False).squeeze(1)
            frames = self._prepare_stem_frames(self._downsample_frames(flat_video[flat_indices], factor))
            dynamic_pyr = self.dynamic_encoder(frames)
            anatomical_pyr = self.anatomical_encoder(frames)
            if len(dynamic_pyr) != self.num_levels or len(anatomical_pyr) != self.num_levels:
                raise ValueError(
                    f"ADAMS expected {self.num_levels} levels from MFE/CFE, got "
                    f"{len(dynamic_pyr)} and {len(anatomical_pyr)}."
                )

            selected_near = flat_near[flat_indices]
            if selected_near.any():
                if factor != 1:
                    raise ValueError(
                        "Near frames must use full resolution. Check resolution_schedule and near_window."
                    )
                if local_dynamic_maps is None:
                    local_dynamic_maps = [
                        feat.new_zeros((batch * num_near, *feat.shape[1:]))
                        for feat in dynamic_pyr
                    ]
                    local_anatomical_maps = [
                        feat.new_zeros((batch * num_near, *feat.shape[1:]))
                        for feat in anatomical_pyr
                    ]
                near_flat_indices = flat_indices[selected_near]
                near_slots = near_lookup[near_flat_indices]
                if bool((near_slots < 0).any().item()):
                    raise ValueError("Failed to map near-frame indices into the compact ADAMS buffer.")
                for level, (dynamic_feat, anatomical_feat) in enumerate(zip(dynamic_pyr, anatomical_pyr)):
                    local_dynamic_maps[level][near_slots] = dynamic_feat[selected_near]
                    local_anatomical_maps[level][near_slots] = anatomical_feat[selected_near]

            selected_mid_far = ~selected_near
            if selected_mid_far.any():
                mid_far_flat_indices = flat_indices[selected_mid_far]
                batch_ids = mid_far_flat_indices // seq_len
                time_ids = mid_far_flat_indices % seq_len
                metadata, frame_delta, frame_resolution = self._build_token_metadata(
                    mid_far_flat_indices,
                    batch_ids,
                    time_ids,
                    delta,
                    resolution_factors,
                    frame_mask,
                    anchor_indices,
                    seq_len,
                )
                token_grid = self._token_grid_for_resolution(factor)
                for level, (
                    dynamic_feat,
                    anatomical_feat,
                    dynamic_projector,
                    anatomical_projector,
                ) in enumerate(
                    zip(
                        dynamic_pyr,
                        anatomical_pyr,
                        self.dynamic_token_projectors,
                        self.anatomical_token_projectors,
                    )
                ):
                    dynamic_tokens = dynamic_projector(
                        dynamic_feat[selected_mid_far],
                        token_grid,
                        metadata.to(device=dynamic_feat.device, dtype=dynamic_feat.dtype),
                    )
                    anatomical_tokens = anatomical_projector(
                        anatomical_feat[selected_mid_far],
                        token_grid,
                        metadata.to(device=anatomical_feat.device, dtype=anatomical_feat.dtype),
                    )
                    dynamic_token_pieces[level].append(
                        (
                            dynamic_tokens,
                            batch_ids,
                            frame_delta.to(device=dynamic_tokens.device, dtype=dynamic_tokens.dtype),
                            frame_resolution.to(device=dynamic_tokens.device, dtype=dynamic_tokens.dtype),
                        )
                    )
                    anatomical_token_pieces[level].append(
                        (
                            anatomical_tokens,
                            batch_ids,
                            frame_delta.to(device=anatomical_tokens.device, dtype=anatomical_tokens.dtype),
                            frame_resolution.to(device=anatomical_tokens.device, dtype=anatomical_tokens.dtype),
                        )
                    )

        if local_dynamic_maps is None or local_anatomical_maps is None:
            raise ValueError("ADAMS could not collect any near-frame maps; check anchor_indices/frame_mask.")

        local_dynamic_maps = [
            feat.view(batch, num_near, *feat.shape[1:])
            for feat in local_dynamic_maps
        ]
        local_anatomical_maps = [
            feat.view(batch, num_near, *feat.shape[1:])
            for feat in local_anatomical_maps
        ]

        return local_dynamic_maps, local_anatomical_maps, dynamic_token_pieces, anatomical_token_pieces

    @staticmethod
    def _gather_anchor_feature_pyr(near_pyr, anchor_local_indices):
        batch = anchor_local_indices.shape[0]
        batch_idx = torch.arange(batch, device=anchor_local_indices.device).view(batch, 1)
        anchor_pyr = [feat[batch_idx, anchor_local_indices] for feat in near_pyr]
        anchor0_pyr = [feat[:, 0] for feat in anchor_pyr]
        anchor1_pyr = [feat[:, 1] for feat in anchor_pyr]
        return anchor_pyr, anchor0_pyr, anchor1_pyr

    def forward(
        self,
        video,
        anchor_indices,
        tau_list=None,
        frame_mask=None,
        return_debug: bool = False,
    ):
        if video.dim() != 5:
            raise ValueError(f"ADAMS expects video [B, T, C, H, W], got {tuple(video.shape)}.")
        if video.shape[2] != self.image_channels:
            raise ValueError(
                f"ADAMS was initialized for {self.image_channels} image channels, "
                f"but video has {video.shape[2]}."
            )

        batch, seq_len = video.shape[:2]
        device = video.device
        dtype = video.dtype
        anchor_indices = self._normalize_anchor_indices(anchor_indices, batch, seq_len, device)
        tau_list = self._normalize_tau_list(tau_list, batch, device, dtype)
        target_indices = self._build_target_indices(anchor_indices, tau_list, dtype=dtype)

        if frame_mask is None:
            frame_mask = self._default_frame_mask(batch, seq_len, device, dtype)
        else:
            frame_mask = frame_mask.to(device=device, dtype=dtype).reshape(batch, seq_len)

        delta = self._build_delta(seq_len, anchor_indices, dtype=dtype)
        resolution_factors = self._build_resolution_factors(delta)
        resolution_factors = self._clamp_resolution_factors_for_stem(
            resolution_factors,
            video.shape[-2],
            video.shape[-1],
        )
        near_mask = (delta <= self.near_window) & (frame_mask > 0)
        near_indices, near_valid, anchor_local_indices = self._build_near_indices(
            near_mask,
            anchor_indices,
        )
        near_clip = self._gather_video_frames(video, near_indices)

        (
            local_dynamic_seq_pyr,
            local_anatomical_seq_pyr,
            dynamic_token_pieces,
            anatomical_token_pieces,
        ) = self._run_stem_grouped(
            video,
            delta=delta,
            resolution_factors=resolution_factors,
            frame_mask=frame_mask,
            near_mask=near_mask,
            near_indices=near_indices,
            anchor_indices=anchor_indices,
        )

        (
            anchor_dynamic_pyr,
            anchor0_dynamic_pyr,
            anchor1_dynamic_pyr,
        ) = self._gather_anchor_feature_pyr(
            local_dynamic_seq_pyr,
            anchor_local_indices,
        )
        (
            anchor_anatomical_pyr,
            anchor0_anatomical_pyr,
            anchor1_anatomical_pyr,
        ) = self._gather_anchor_feature_pyr(
            local_anatomical_seq_pyr,
            anchor_local_indices,
        )

        dynamic_context_pyr = []
        anatomical_context_pyr = []
        local_dynamic_pyr = []
        local_anatomical_pyr = []
        global_dynamic_pyr = []
        global_anatomical_pyr = []
        debug = {
            "dynamic_attention_pyr": [],
            "anatomical_attention_pyr": [],
            "dynamic_token_valid_pyr": [],
            "anatomical_token_valid_pyr": [],
        }

        for level, channels in enumerate(self.channel_list):
            local_dynamic = self.dynamic_local_mixers[level](
                near_clip,
                local_dynamic_seq_pyr[level],
                near_indices,
                near_valid,
                anchor_local_indices,
                delta,
                target_indices,
                seq_len,
            )
            local_anatomical = self.anatomical_local_mixers[level](
                near_clip,
                local_anatomical_seq_pyr[level],
                near_indices,
                near_valid,
                anchor_local_indices,
                delta,
                target_indices,
                seq_len,
            )
            query = self.query_builders[level](video, anchor_indices, tau_list, seq_len)

            dynamic_tokens, dynamic_valid, dynamic_delta, dynamic_resolution = self._pack_token_pieces(
                dynamic_token_pieces[level],
                batch,
                channels,
                device,
                dtype,
            )
            anatomical_tokens, anatomical_valid, anatomical_delta, anatomical_resolution = self._pack_token_pieces(
                anatomical_token_pieces[level],
                batch,
                channels,
                device,
                dtype,
            )

            dynamic_global_result = self.dynamic_global_mixers[level](
                query,
                dynamic_tokens,
                dynamic_valid,
                dynamic_delta,
                dynamic_resolution,
                return_attention=return_debug,
            )
            anatomical_global_result = self.anatomical_global_mixers[level](
                query,
                anatomical_tokens,
                anatomical_valid,
                anatomical_delta,
                anatomical_resolution,
                return_attention=return_debug,
            )
            if return_debug:
                dynamic_global, dynamic_attention = dynamic_global_result
                anatomical_global, anatomical_attention = anatomical_global_result
                debug["dynamic_attention_pyr"].append(dynamic_attention)
                debug["anatomical_attention_pyr"].append(anatomical_attention)
                debug["dynamic_token_valid_pyr"].append(dynamic_valid)
                debug["anatomical_token_valid_pyr"].append(anatomical_valid)
            else:
                dynamic_global = dynamic_global_result
                anatomical_global = anatomical_global_result

            dynamic_context = self.dynamic_modulators[level](local_dynamic, dynamic_global)
            anatomical_context = self.anatomical_modulators[level](
                local_anatomical,
                anatomical_global,
            )

            local_dynamic_pyr.append(local_dynamic)
            local_anatomical_pyr.append(local_anatomical)
            global_dynamic_pyr.append(dynamic_global)
            global_anatomical_pyr.append(anatomical_global)
            dynamic_context_pyr.append(dynamic_context)
            anatomical_context_pyr.append(anatomical_context)

        if self.squeeze_single_tau and tau_list.shape[1] == 1:
            dynamic_context_pyr = [feat[:, 0] for feat in dynamic_context_pyr]
            anatomical_context_pyr = [feat[:, 0] for feat in anatomical_context_pyr]
            local_dynamic_pyr = [feat[:, 0] for feat in local_dynamic_pyr]
            local_anatomical_pyr = [feat[:, 0] for feat in local_anatomical_pyr]

        result = {
            "dynamic_context_pyr": dynamic_context_pyr,
            "anatomical_context_pyr": anatomical_context_pyr,
            "F_dynamic_pyr": dynamic_context_pyr,
            "F_anatomical_pyr": anatomical_context_pyr,
            "local_dynamic_pyr": local_dynamic_pyr,
            "local_anatomical_pyr": local_anatomical_pyr,
            "anchor_dynamic_pyr": anchor_dynamic_pyr,
            "anchor_anatomical_pyr": anchor_anatomical_pyr,
            "anchor0_dynamic_pyr": anchor0_dynamic_pyr,
            "anchor1_dynamic_pyr": anchor1_dynamic_pyr,
            "anchor0_anatomical_pyr": anchor0_anatomical_pyr,
            "anchor1_anatomical_pyr": anchor1_anatomical_pyr,
            "mtmc_feat0_pyr": anchor0_dynamic_pyr,
            "mtmc_feat1_pyr": anchor1_dynamic_pyr,
            "global_dynamic_tokens_pyr": global_dynamic_pyr,
            "global_anatomical_tokens_pyr": global_anatomical_pyr,
        }
        if return_debug:
            debug.update(
                {
                    "delta": delta,
                    "resolution_factors": resolution_factors,
                    "near_mask": near_mask,
                    "near_indices": near_indices,
                    "near_valid": near_valid,
                    "anchor_local_indices": anchor_local_indices,
                    "target_indices": target_indices,
                    "tau_list": tau_list,
                }
            )
            result["debug"] = debug
        return result


AnatomicalDynamicAnchorAwareMultiScaleSequenceContextEncoder = ADAMS
