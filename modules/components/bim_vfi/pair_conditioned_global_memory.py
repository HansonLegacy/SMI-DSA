import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sincos_1d(positions, dim: int):
    if dim <= 0:
        raise ValueError("dim must be positive for sinusoidal embedding.")

    out_dtype = positions.dtype
    positions = positions.float()
    half_dim = dim // 2
    if half_dim == 0:
        return positions.unsqueeze(-1).to(dtype=out_dtype)

    denom = max(half_dim - 1, 1)
    freq = torch.exp(
        torch.arange(half_dim, device=positions.device, dtype=positions.dtype)
        * (-math.log(10000.0) / denom)
    )
    angles = positions.unsqueeze(-1) * freq
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


def _valid_lengths(frame_mask, seq_len: int, device):
    if frame_mask is None:
        return torch.full((1,), seq_len, device=device, dtype=torch.long)
    return frame_mask.to(device=device).long().sum(dim=1).clamp_min(1)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class BiasedCrossAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0):
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
    def __init__(self, channels: int, num_heads: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.0):
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


class StagePriorEncoder(nn.Module):
    """
    Encodes lightweight DSA phase statistics into per-frame stage tokens.
    Input stage_raw shape: [B, T, 4].
    Output stage tokens shape: [B, T, C].
    """

    def __init__(
        self,
        channels: int = 256,
        temporal_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(4, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        if temporal_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=num_heads,
                dim_feedforward=max(channels, int(channels * mlp_ratio)),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        else:
            self.temporal = None
        self.norm = nn.LayerNorm(channels)

    def forward(self, stage_raw, frame_mask=None):
        tokens = self.input_mlp(stage_raw)
        if self.temporal is not None:
            key_padding_mask = None
            if frame_mask is not None:
                key_padding_mask = frame_mask.to(device=tokens.device) <= 0
            tokens = self.temporal(tokens, src_key_padding_mask=key_padding_mask)
        tokens = self.norm(tokens)
        if frame_mask is not None:
            tokens = tokens * frame_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        return tokens


class LowResFrameEncoder(nn.Module):
    """
    Encodes concat([I_t, I_t - I_{t-1}, I_{t+1} - I_t]) into spatial tokens.
    """

    def __init__(
        self,
        image_channels: int = 1,
        channels: int = 256,
        downsample: int = 8,
        stem_channels: int = 64,
    ):
        super().__init__()
        if downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {downsample}.")

        in_channels = int(image_channels) * 3
        layers = []
        current_channels = in_channels
        current_downsample = int(downsample)
        hidden = max(stem_channels, channels // 4)

        if current_downsample == 1:
            layers.append(ConvNormAct(current_channels, hidden, stride=1))
        else:
            while current_downsample > 1:
                if current_downsample % 2 != 0:
                    raise ValueError("LowResFrameEncoder currently expects downsample to be a power of 2.")
                out_channels = min(channels, hidden)
                layers.append(ConvNormAct(current_channels, out_channels, stride=2))
                current_channels = out_channels
                hidden = min(channels, hidden * 2)
                current_downsample //= 2

        layers.extend(
            [
                ConvNormAct(current_channels, channels, stride=1),
                nn.Conv2d(channels, channels, 3, padding=1),
            ]
        )
        self.encoder = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"LowResFrameEncoder expects [B, T, C, H, W], got {tuple(x.shape)}")

        batch, seq_len, channels, height, width = x.shape
        x_flat = x.reshape(batch * seq_len, channels, height, width)
        feat = self.encoder(x_flat)
        low_h, low_w = feat.shape[-2:]
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.out_norm(tokens)
        tokens = tokens.reshape(batch, seq_len, low_h * low_w, -1)
        return tokens, (low_h, low_w)


class TemporalMemoryEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        num_slots: int,
        num_heads: int = 8,
        temporal_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        encoder_type: str = "transformer",
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_slots = int(num_slots)
        self.encoder_type = str(encoder_type).lower()
        self.slot_pos = nn.Parameter(torch.zeros(num_slots, channels))

        if temporal_layers <= 0 or self.encoder_type == "identity":
            self.encoder = None
        elif self.encoder_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=num_heads,
                dim_feedforward=max(channels, int(channels * mlp_ratio)),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        elif self.encoder_type == "bigru":
            hidden = max(channels // 2, 1)
            self.encoder = nn.GRU(
                input_size=channels,
                hidden_size=hidden,
                num_layers=temporal_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if temporal_layers > 1 else 0.0,
            )
            self.gru_out = nn.Linear(hidden * 2, channels)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.norm = nn.LayerNorm(channels)

    def forward(self, memory, frame_mask=None, frame_positions=None):
        if memory.dim() != 4:
            raise ValueError(f"TemporalMemoryEncoder expects [B, T, N, C], got {tuple(memory.shape)}")

        batch, seq_len, num_slots, channels = memory.shape
        if num_slots != self.num_slots or channels != self.channels:
            raise ValueError(
                f"Expected memory slots/channels ({self.num_slots}, {self.channels}), "
                f"got ({num_slots}, {channels})."
            )

        if frame_positions is None:
            denom = max(seq_len - 1, 1)
            pos = torch.arange(seq_len, device=memory.device, dtype=memory.dtype).view(1, seq_len) / denom
            pos = pos.expand(batch, -1)
        else:
            pos = frame_positions.to(device=memory.device, dtype=memory.dtype).reshape(batch, seq_len)

        time_pos = _sincos_1d(pos, channels).unsqueeze(2)
        slot_pos = self.slot_pos.to(device=memory.device, dtype=memory.dtype).view(1, 1, num_slots, channels)
        tokens = memory + time_pos + slot_pos
        tokens = tokens.reshape(batch, seq_len * num_slots, channels)

        key_padding_mask = None
        if frame_mask is not None:
            key_padding_mask = (frame_mask.to(device=memory.device) <= 0).unsqueeze(-1)
            key_padding_mask = key_padding_mask.expand(batch, seq_len, num_slots).reshape(batch, seq_len * num_slots)

        if self.encoder is not None:
            if self.encoder_type == "transformer":
                tokens = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
            elif self.encoder_type == "bigru":
                tokens, _ = self.encoder(tokens)
                tokens = self.gru_out(tokens)

        tokens = self.norm(tokens)
        tokens = tokens.reshape(batch, seq_len, num_slots, channels)
        if frame_mask is not None:
            tokens = tokens * frame_mask.to(device=memory.device, dtype=memory.dtype).view(batch, seq_len, 1, 1)
        return tokens


class GlobalMemoryBuilder(nn.Module):
    """
    Builds pair-agnostic global memories from a low-frame-rate DSA sequence.

    Output:
        M_motion:    [B, T, N_motion, C]
        M_structure: [B, T, N_structure, C]
    """

    def __init__(
        self,
        image_channels: int = 1,
        channels: int = 256,
        n_motion: int = 2,
        n_structure: int = 4,
        downsample: int = 8,
        stage_layers: int = 2,
        temporal_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        temporal_encoder_type: str = "transformer",
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_motion = int(n_motion)
        self.n_structure = int(n_structure)

        self.stage_encoder = StagePriorEncoder(
            channels=channels,
            temporal_layers=stage_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.frame_encoder = LowResFrameEncoder(
            image_channels=image_channels,
            channels=channels,
            downsample=downsample,
        )

        self.write_queries_motion = nn.Parameter(torch.randn(n_motion, channels) * 0.02)
        self.write_queries_structure = nn.Parameter(torch.randn(n_structure, channels) * 0.02)
        self.motion_stage_to_query = self._make_stage_query_mlp(n_motion, channels)
        self.structure_stage_to_query = self._make_stage_query_mlp(n_structure, channels)

        self.motion_write = CrossAttentionBlock(
            channels, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.structure_write = CrossAttentionBlock(
            channels, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.motion_temporal = TemporalMemoryEncoder(
            channels,
            n_motion,
            num_heads=num_heads,
            temporal_layers=temporal_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            encoder_type=temporal_encoder_type,
        )
        self.structure_temporal = TemporalMemoryEncoder(
            channels,
            n_structure,
            num_heads=num_heads,
            temporal_layers=temporal_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            encoder_type=temporal_encoder_type,
        )

    @staticmethod
    def _make_stage_query_mlp(num_slots: int, channels: int):
        return nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, num_slots * channels),
        )

    @staticmethod
    def build_change_input(video, frame_mask=None):
        if video.dim() != 5:
            raise ValueError(f"video must be [B, T, C, H, W], got {tuple(video.shape)}")

        d_prev = torch.zeros_like(video)
        d_next = torch.zeros_like(video)
        if video.shape[1] > 1:
            d_prev[:, 1:] = video[:, 1:] - video[:, :-1]
            d_prev[:, 0] = d_prev[:, 1]
            d_next[:, :-1] = video[:, 1:] - video[:, :-1]
            d_next[:, -1] = d_next[:, -2]
            if frame_mask is not None:
                valid_lengths = frame_mask.to(device=video.device).long().sum(dim=1).clamp_min(0)
                for batch_idx, seq_len in enumerate(valid_lengths.tolist()):
                    if seq_len <= 0:
                        d_prev[batch_idx].zero_()
                        d_next[batch_idx].zero_()
                    elif seq_len == 1:
                        d_prev[batch_idx].zero_()
                        d_next[batch_idx].zero_()
                    else:
                        d_prev[batch_idx, 0] = d_prev[batch_idx, 1]
                        d_next[batch_idx, seq_len - 1] = d_next[batch_idx, seq_len - 2]
                        d_prev[batch_idx, seq_len:] = 0
                        d_next[batch_idx, seq_len:] = 0
        return torch.cat([video, d_prev, d_next], dim=2), d_prev, d_next

    @staticmethod
    def build_stage_raw(video, d_prev, d_next, frame_positions=None):
        batch, seq_len = video.shape[:2]
        mean_frame = video.mean(dim=(2, 3, 4))
        mean_prev = d_prev.abs().mean(dim=(2, 3, 4))
        mean_next = d_next.abs().mean(dim=(2, 3, 4))
        if frame_positions is None:
            denom = max(seq_len - 1, 1)
            pos = torch.arange(seq_len, device=video.device, dtype=video.dtype).view(1, seq_len) / denom
            pos = pos.expand(batch, -1)
        else:
            pos = frame_positions.to(device=video.device, dtype=video.dtype).reshape(batch, seq_len)
        return torch.stack([mean_frame, mean_prev, mean_next, pos], dim=-1)

    def forward(self, video, frame_mask=None, frame_positions=None):
        batch, seq_len = video.shape[:2]
        change_input, d_prev, d_next = self.build_change_input(video, frame_mask=frame_mask)
        stage_raw = self.build_stage_raw(video, d_prev, d_next, frame_positions=frame_positions)
        stage_tokens = self.stage_encoder(stage_raw, frame_mask=frame_mask)

        frame_tokens, spatial_hw = self.frame_encoder(change_input)
        if frame_mask is not None:
            frame_tokens = frame_tokens * frame_mask.to(
                device=frame_tokens.device, dtype=frame_tokens.dtype
            ).view(batch, seq_len, 1, 1)

        motion_delta = self.motion_stage_to_query(stage_tokens).reshape(
            batch, seq_len, self.n_motion, self.channels
        )
        structure_delta = self.structure_stage_to_query(stage_tokens).reshape(
            batch, seq_len, self.n_structure, self.channels
        )
        motion_queries = self.write_queries_motion.view(1, 1, self.n_motion, self.channels) + motion_delta
        structure_queries = (
            self.write_queries_structure.view(1, 1, self.n_structure, self.channels) + structure_delta
        )

        flat_frame_tokens = frame_tokens.reshape(batch * seq_len, frame_tokens.shape[2], self.channels)
        motion_raw = self.motion_write(
            motion_queries.reshape(batch * seq_len, self.n_motion, self.channels),
            flat_frame_tokens,
        ).reshape(batch, seq_len, self.n_motion, self.channels)
        structure_raw = self.structure_write(
            structure_queries.reshape(batch * seq_len, self.n_structure, self.channels),
            flat_frame_tokens,
        ).reshape(batch, seq_len, self.n_structure, self.channels)

        if frame_mask is not None:
            valid = frame_mask.to(device=video.device, dtype=video.dtype).view(batch, seq_len, 1, 1)
            motion_raw = motion_raw * valid
            structure_raw = structure_raw * valid

        m_motion = self.motion_temporal(motion_raw, frame_mask=frame_mask, frame_positions=frame_positions)
        m_structure = self.structure_temporal(
            structure_raw, frame_mask=frame_mask, frame_positions=frame_positions
        )
        return {
            "M_motion": m_motion,
            "M_structure": m_structure,
            "stage_tokens": stage_tokens,
            "spatial_hw": spatial_hw,
        }


class PairMemoryReader(nn.Module):
    """
    Reads global motion/structure memories for the current anchor pair and tau.

    Input:
        M_motion:    [B, T, N_motion, C]
        M_structure: [B, T, N_structure, C]
        k:           [B], left anchor index
        tau:         [B], target position inside (k, k + 1)

    Output:
        C_motion:    [B, R_motion, C]
        C_structure: [B, R_structure, C]
    """

    def __init__(
        self,
        channels: int = 256,
        n_motion: int = 2,
        n_structure: int = 4,
        r_motion: int = 4,
        r_structure: int = 4,
        num_heads: int = 8,
        pos_embed_dim: int = 64,
        lambda_motion: float = 0.75,
        lambda_structure: float = 0.2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_motion = int(n_motion)
        self.n_structure = int(n_structure)
        self.r_motion = int(r_motion)
        self.r_structure = int(r_structure)
        self.pos_embed_dim = int(pos_embed_dim)
        self.lambda_motion = float(lambda_motion)
        self.lambda_structure = float(lambda_structure)

        prompt_in = channels * 4 + pos_embed_dim * 3
        self.prompt_mlp = nn.Sequential(
            nn.LayerNorm(prompt_in),
            nn.Linear(prompt_in, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

        self.read_queries_motion = nn.Parameter(torch.randn(r_motion, channels) * 0.02)
        self.read_queries_structure = nn.Parameter(torch.randn(r_structure, channels) * 0.02)
        self.motion_prompt_to_query = self._make_prompt_query_mlp(r_motion, channels)
        self.structure_prompt_to_query = self._make_prompt_query_mlp(r_structure, channels)

        self.motion_read = CrossAttentionBlock(
            channels, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.structure_read = CrossAttentionBlock(
            channels, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )

    @staticmethod
    def _make_prompt_query_mlp(num_slots: int, channels: int):
        return nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, num_slots * channels),
        )

    @staticmethod
    def _gather_pair(memory, k, anchor_indices=None):
        batch, seq_len = memory.shape[:2]
        batch_idx = torch.arange(batch, device=memory.device)
        if anchor_indices is not None:
            anchor_indices = anchor_indices.to(device=memory.device, dtype=torch.long)
            if anchor_indices.dim() == 1:
                anchor_indices = anchor_indices.view(1, -1).expand(batch, -1)
            k = anchor_indices[:, 0].reshape(batch)
            k_next = anchor_indices[:, 1].reshape(batch)
        else:
            k = k.to(device=memory.device, dtype=torch.long).reshape(batch)
            k_next = k + 1
        k = k.clamp(min=0, max=max(seq_len - 1, 0))
        k_next = k_next.clamp(min=0, max=max(seq_len - 1, 0))
        return memory[batch_idx, k], memory[batch_idx, k_next], k, k_next

    @staticmethod
    def _key_padding_mask(frame_mask, num_slots: int, seq_len: int, device):
        if frame_mask is None:
            return None
        mask = frame_mask.to(device=device) <= 0
        return mask.unsqueeze(-1).expand(-1, seq_len, num_slots).reshape(mask.shape[0], seq_len * num_slots)

    @staticmethod
    def _time_distance_bias(center, seq_len: int, num_slots: int, num_read: int, lamb: float, dtype, device):
        center = center.to(device=device, dtype=dtype).reshape(-1, 1)
        time_index = torch.arange(seq_len, device=device, dtype=dtype).view(1, seq_len)
        bias_t = -float(lamb) * torch.log1p(torch.abs(time_index - center))
        bias = bias_t.unsqueeze(-1).expand(-1, seq_len, num_slots).reshape(-1, seq_len * num_slots)
        return bias.unsqueeze(1).expand(-1, num_read, -1)

    def _build_prompt(self, m_motion, m_structure, k, tau, frame_mask=None, anchor_indices=None):
        batch, seq_len = m_motion.shape[:2]
        m_k_motion, m_k1_motion, k, k_next = self._gather_pair(
            m_motion, k, anchor_indices=anchor_indices
        )
        m_k_structure, m_k1_structure, _, _ = self._gather_pair(
            m_structure, k, anchor_indices=anchor_indices
        )

        m_k_motion = m_k_motion.mean(dim=1)
        m_k1_motion = m_k1_motion.mean(dim=1)
        m_k_structure = m_k_structure.mean(dim=1)
        m_k1_structure = m_k1_structure.mean(dim=1)

        tau = tau.to(device=m_motion.device, dtype=m_motion.dtype).reshape(batch)
        lengths = _valid_lengths(frame_mask, seq_len, m_motion.device).to(dtype=m_motion.dtype)
        if lengths.numel() == 1 and batch > 1:
            lengths = lengths.expand(batch)
        denom = (lengths - 1).clamp_min(1.0)
        span = (k_next - k).to(dtype=m_motion.dtype).clamp_min(1.0)
        center_index = k.to(dtype=m_motion.dtype) + tau * span
        k_norm = k.to(dtype=m_motion.dtype) / denom
        center_norm = center_index / denom

        prompt = torch.cat(
            [
                m_k_motion,
                m_k1_motion,
                m_k_structure,
                m_k1_structure,
                _sincos_1d(k_norm, self.pos_embed_dim),
                _sincos_1d(center_norm, self.pos_embed_dim),
                _sincos_1d(tau, self.pos_embed_dim),
            ],
            dim=-1,
        )
        return self.prompt_mlp(prompt), center_index

    def forward(
        self,
        m_motion,
        m_structure,
        k,
        tau,
        frame_mask=None,
        anchor_indices=None,
        return_attention: bool = False,
    ):
        batch, seq_len, n_motion, channels = m_motion.shape
        if n_motion != self.n_motion or channels != self.channels:
            raise ValueError(
                f"Expected M_motion [B, T, {self.n_motion}, {self.channels}], got {tuple(m_motion.shape)}"
            )
        if m_structure.shape[2] != self.n_structure or m_structure.shape[3] != self.channels:
            raise ValueError(
                f"Expected M_structure [B, T, {self.n_structure}, {self.channels}], "
                f"got {tuple(m_structure.shape)}"
            )

        q_pair, center_index = self._build_prompt(
            m_motion,
            m_structure,
            k,
            tau,
            frame_mask=frame_mask,
            anchor_indices=anchor_indices,
        )
        motion_queries = self.read_queries_motion.view(1, self.r_motion, channels)
        motion_queries = motion_queries + self.motion_prompt_to_query(q_pair).reshape(
            batch, self.r_motion, channels
        )
        structure_queries = self.read_queries_structure.view(1, self.r_structure, channels)
        structure_queries = structure_queries + self.structure_prompt_to_query(q_pair).reshape(
            batch, self.r_structure, channels
        )

        motion_flat = m_motion.reshape(batch, seq_len * self.n_motion, channels)
        structure_flat = m_structure.reshape(batch, seq_len * self.n_structure, channels)

        motion_bias = self._time_distance_bias(
            center_index,
            seq_len,
            self.n_motion,
            self.r_motion,
            self.lambda_motion,
            m_motion.dtype,
            m_motion.device,
        )
        structure_bias = self._time_distance_bias(
            center_index,
            seq_len,
            self.n_structure,
            self.r_structure,
            self.lambda_structure,
            m_structure.dtype,
            m_structure.device,
        )

        motion_mask = self._key_padding_mask(frame_mask, self.n_motion, seq_len, m_motion.device)
        structure_mask = self._key_padding_mask(frame_mask, self.n_structure, seq_len, m_structure.device)

        c_motion = self.motion_read(
            motion_queries,
            motion_flat,
            key_padding_mask=motion_mask,
            attn_bias=motion_bias,
            return_weights=return_attention,
        )
        c_structure = self.structure_read(
            structure_queries,
            structure_flat,
            key_padding_mask=structure_mask,
            attn_bias=structure_bias,
            return_weights=return_attention,
        )

        result = {
            "C_motion": c_motion[0] if return_attention else c_motion,
            "C_structure": c_structure[0] if return_attention else c_structure,
            "q_pair": q_pair,
        }
        if return_attention:
            result["motion_attn"] = c_motion[1]
            result["structure_attn"] = c_structure[1]
        return result


class CoordinateMotionPriorHead(nn.Module):
    def __init__(self, channels: int = 256, out_channels: int = 4, hidden_channels: int = None):
        super().__init__()
        hidden_channels = channels if hidden_channels is None else hidden_channels
        self.code_proj = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )
        self.coord_mlp = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, motion_tokens, spatial_hw):
        batch, _, channels = motion_tokens.shape
        height, width = spatial_hw
        code = self.code_proj(motion_tokens.mean(dim=1))
        grid = _sincos_2d_grid(height, width, channels, motion_tokens.device, motion_tokens.dtype)
        x = grid.unsqueeze(0) + code.unsqueeze(1)
        prior = self.coord_mlp(x)
        return prior.transpose(1, 2).reshape(batch, -1, height, width)


class TokenContextPyramidAdapter(nn.Module):
    """
    Adapts pair-conditioned memory tokens to the 3-level context pyramid used by
    the current MTMC/TCAR implementations.

    This is a compatibility adapter: the global memory remains token-based, and
    only the downstream-facing representation is expanded into spatial maps.
    """

    def __init__(
        self,
        token_channels: int = 256,
        feat_channels: int = 32,
        channel_multipliers=(1, 2, 4),
        hidden_channels: int = None,
        use_coordinate_mlp: bool = True,
    ):
        super().__init__()
        self.token_channels = int(token_channels)
        self.feat_channels = int(feat_channels)
        self.channel_multipliers = tuple(int(multiplier) for multiplier in channel_multipliers)
        self.use_coordinate_mlp = bool(use_coordinate_mlp)
        hidden_channels = self.token_channels if hidden_channels is None else int(hidden_channels)

        self.level_mlps = nn.ModuleList()
        for multiplier in self.channel_multipliers:
            out_channels = self.feat_channels * multiplier
            self.level_mlps.append(
                nn.Sequential(
                    nn.LayerNorm(self.token_channels),
                    nn.Linear(self.token_channels, hidden_channels),
                    nn.GELU(),
                    nn.Linear(hidden_channels, out_channels),
                )
            )

    def forward(self, tokens, target_hw_pyr):
        if tokens is None:
            return None
        if tokens.dim() != 3:
            raise ValueError(f"TokenContextPyramidAdapter expects [B, R, C], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.token_channels:
            raise ValueError(
                f"Expected token channels {self.token_channels}, got {tokens.shape[-1]}"
            )
        if len(target_hw_pyr) < len(self.level_mlps):
            raise ValueError(
                f"Expected at least {len(self.level_mlps)} target sizes, got {len(target_hw_pyr)}"
            )

        batch = tokens.shape[0]
        code = tokens.mean(dim=1)
        context_pyr = []
        for level_mlp, target_hw in zip(self.level_mlps, target_hw_pyr):
            height, width = int(target_hw[0]), int(target_hw[1])
            if self.use_coordinate_mlp:
                grid = _sincos_2d_grid(height, width, self.token_channels, tokens.device, tokens.dtype)
                level_input = grid.unsqueeze(0) + code.unsqueeze(1)
                level_context = level_mlp(level_input).transpose(1, 2).reshape(
                    batch, -1, height, width
                )
            else:
                level_context = level_mlp(code).view(batch, -1, 1, 1).expand(
                    batch, -1, height, width
                )
            context_pyr.append(level_context)
        return context_pyr


class PairConditionedGlobalMemory(nn.Module):
    """
    Standalone prototype for pair-conditioned global memory in DSA-VFI.

    This module intentionally does not match the current GTME interface yet.
    It exposes the proposed compact memory tensors and pair-conditioned reads.
    """

    def __init__(
        self,
        image_channels: int = 1,
        channels: int = 256,
        n_motion: int = 2,
        n_structure: int = 4,
        r_motion: int = 4,
        r_structure: int = 4,
        downsample: int = 8,
        num_heads: int = 8,
        stage_layers: int = 2,
        temporal_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        lambda_motion: float = 0.75,
        lambda_structure: float = 0.2,
        motion_prior_channels: int = 4,
        use_motion_prior_head: bool = True,
    ):
        super().__init__()
        self.builder = GlobalMemoryBuilder(
            image_channels=image_channels,
            channels=channels,
            n_motion=n_motion,
            n_structure=n_structure,
            downsample=downsample,
            stage_layers=stage_layers,
            temporal_layers=temporal_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.reader = PairMemoryReader(
            channels=channels,
            n_motion=n_motion,
            n_structure=n_structure,
            r_motion=r_motion,
            r_structure=r_structure,
            num_heads=num_heads,
            lambda_motion=lambda_motion,
            lambda_structure=lambda_structure,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.use_motion_prior_head = bool(use_motion_prior_head)
        if self.use_motion_prior_head:
            self.motion_prior_head = CoordinateMotionPriorHead(
                channels=channels,
                out_channels=motion_prior_channels,
            )
            self.motion_gate = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, 1),
                nn.Sigmoid(),
            )
        else:
            self.motion_prior_head = None
            self.motion_gate = None
        self.structure_norm = nn.LayerNorm(channels)

    def forward(
        self,
        video,
        k,
        tau,
        frame_mask=None,
        frame_positions=None,
        anchor_indices=None,
        return_attention: bool = False,
    ):
        memory_dict = self.builder(video, frame_mask=frame_mask, frame_positions=frame_positions)
        read_dict = self.reader(
            memory_dict["M_motion"],
            memory_dict["M_structure"],
            k,
            tau,
            frame_mask=frame_mask,
            anchor_indices=anchor_indices,
            return_attention=return_attention,
        )

        result = {**memory_dict, **read_dict}
        result["structure_condition"] = self.structure_norm(result["C_structure"])
        if self.use_motion_prior_head:
            result["motion_prior"] = self.motion_prior_head(result["C_motion"], memory_dict["spatial_hw"])
            motion_code = result["C_motion"].mean(dim=1)
            result["motion_gate"] = self.motion_gate(motion_code).view(video.shape[0], 1, 1, 1)
        return result


DSAPairGlobalMemory = PairConditionedGlobalMemory
