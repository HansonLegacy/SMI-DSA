import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


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


class TemporalMixBlock(nn.Module):
    """
    Lightweight temporal mixer for [B, T, C, H, W] tensors.

    This is intentionally local in time. LTCE is expected to handle a short
    target-centered window, while long-range phase/context is handled elsewhere.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise_temporal = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x):
        batch, seq_len, channels, height, width = x.shape
        residual = x
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.pointwise(self.depthwise_temporal(x))
        x = x.reshape(batch, channels, seq_len, height * width)
        x = self.norm(x)
        x = x.reshape(batch, channels, seq_len, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.act(x + residual)


class FrameAdapter(nn.Module):
    """
    Per-frame feature adapter.

    It enriches each MFE/CFE frame feature with raw image evidence and temporal
    metadata, but does not collapse the temporal dimension.
    """

    def __init__(self, channels: int, metadata_channels: int):
        super().__init__()
        self.metadata_channels = int(metadata_channels)
        self.in_proj = ConvNormAct(channels * 2 + metadata_channels, channels)
        self.residual = ResidualConvBlock(channels)

    @staticmethod
    def _metadata_maps(metadata, height: int, width: int):
        return metadata.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, height, width)

    def forward(self, feat_seq, raw_seq, metadata):
        batch, seq_len, channels, height, width = feat_seq.shape
        metadata_maps = self._metadata_maps(metadata, height, width).to(
            device=feat_seq.device,
            dtype=feat_seq.dtype,
        )
        x = torch.cat([feat_seq, raw_seq, metadata_maps], dim=2)
        x = x.reshape(batch * seq_len, channels * 2 + self.metadata_channels, height, width)
        x = self.residual(self.in_proj(x))
        return x.reshape(batch, seq_len, channels, height, width)


class MotionEvidenceBlock(nn.Module):
    """
    Builds local motion evidence before target-centered fusion.

    The block keeps signed differences so the motion branch can reason about
    direction instead of only magnitude.
    """

    def __init__(self, channels: int, metadata_channels: int):
        super().__init__()
        self.metadata_channels = int(metadata_channels)
        self.frame_block = nn.Sequential(
            ConvNormAct(channels * 6 + metadata_channels, channels),
            ResidualConvBlock(channels),
        )
        self.temporal_mix = TemporalMixBlock(channels)

    @staticmethod
    def _metadata_maps(metadata, height: int, width: int):
        return metadata.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, height, width)

    @staticmethod
    def _gather_anchors(feat_seq, anchor_indices):
        batch = feat_seq.shape[0]
        batch_index = torch.arange(batch, device=feat_seq.device)
        anchor0 = feat_seq[batch_index, anchor_indices[:, 0]]
        anchor1 = feat_seq[batch_index, anchor_indices[:, 1]]
        return anchor0, anchor1

    def forward(self, frame_seq, metadata, anchor_indices):
        batch, seq_len, channels, height, width = frame_seq.shape
        anchor0, anchor1 = self._gather_anchors(frame_seq, anchor_indices)
        anchor0_seq = anchor0.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        anchor1_seq = anchor1.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        signed_pair = (anchor1 - anchor0).unsqueeze(1).expand(-1, seq_len, -1, -1, -1)

        metadata_maps = self._metadata_maps(metadata, height, width).to(
            device=frame_seq.device,
            dtype=frame_seq.dtype,
        )
        x = torch.cat(
            [
                frame_seq,
                anchor0_seq,
                anchor1_seq,
                signed_pair,
                frame_seq - anchor0_seq,
                frame_seq - anchor1_seq,
                metadata_maps,
            ],
            dim=2,
        )
        x = x.reshape(batch * seq_len, channels * 6 + self.metadata_channels, height, width)
        x = self.frame_block(x).reshape(batch, seq_len, channels, height, width)
        return self.temporal_mix(x)


class DetailEvidenceBlock(nn.Module):
    """
    Builds detail/structure evidence before target alignment.

    This branch keeps absolute residuals because detail reconstruction needs to
    know where local structures change or become unreliable across frames.
    """

    def __init__(self, channels: int, metadata_channels: int):
        super().__init__()
        self.metadata_channels = int(metadata_channels)
        self.frame_block = nn.Sequential(
            ConvNormAct(channels * 6 + metadata_channels, channels),
            ResidualConvBlock(channels),
        )
        self.temporal_mix = TemporalMixBlock(channels)

    @staticmethod
    def _metadata_maps(metadata, height: int, width: int):
        return metadata.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, height, width)

    @staticmethod
    def _gather_anchors(feat_seq, anchor_indices):
        batch = feat_seq.shape[0]
        batch_index = torch.arange(batch, device=feat_seq.device)
        anchor0 = feat_seq[batch_index, anchor_indices[:, 0]]
        anchor1 = feat_seq[batch_index, anchor_indices[:, 1]]
        return anchor0, anchor1

    def forward(self, frame_seq, metadata, anchor_indices):
        batch, seq_len, channels, height, width = frame_seq.shape
        anchor0, anchor1 = self._gather_anchors(frame_seq, anchor_indices)
        anchor0_seq = anchor0.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        anchor1_seq = anchor1.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        signed_pair = (anchor1 - anchor0).unsqueeze(1).expand(-1, seq_len, -1, -1, -1)

        metadata_maps = self._metadata_maps(metadata, height, width).to(
            device=frame_seq.device,
            dtype=frame_seq.dtype,
        )
        x = torch.cat(
            [
                frame_seq,
                anchor0_seq,
                anchor1_seq,
                signed_pair,
                torch.abs(frame_seq - anchor0_seq),
                torch.abs(frame_seq - anchor1_seq),
                metadata_maps,
            ],
            dim=2,
        )
        x = x.reshape(batch * seq_len, channels * 6 + self.metadata_channels, height, width)
        x = self.frame_block(x).reshape(batch, seq_len, channels, height, width)
        return self.temporal_mix(x)


class TargetCenteredMotionFusion(nn.Module):
    """
    Converts local motion evidence into a target-time motion context for MTMC.

    The temporal read is spatially varying: every pixel may attend to different
    frames in the local window.
    """

    def __init__(self, channels: int, metadata_channels: int):
        super().__init__()
        self.metadata_channels = int(metadata_channels)
        self.query = nn.Sequential(
            ConvNormAct(channels * 4 + 1, channels),
            ResidualConvBlock(channels),
        )
        self.score = nn.Sequential(
            ConvNormAct(channels * 3 + metadata_channels, channels),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        self.context = nn.Sequential(
            ConvNormAct(channels * 6, channels),
            ResidualConvBlock(channels),
        )
        self.gate = nn.Sequential(
            ConvNormAct(channels * 3, channels),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _gather_anchors(feat_seq, anchor_indices):
        batch = feat_seq.shape[0]
        batch_index = torch.arange(batch, device=feat_seq.device)
        anchor0 = feat_seq[batch_index, anchor_indices[:, 0]]
        anchor1 = feat_seq[batch_index, anchor_indices[:, 1]]
        return anchor0, anchor1

    @staticmethod
    def _metadata_maps(metadata, height: int, width: int):
        return metadata.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, height, width)

    @staticmethod
    def _target_map(target_time, height: int, width: int):
        return target_time.view(target_time.shape[0], 1, 1, 1).expand(-1, 1, height, width)

    def forward(self, evidence_seq, metadata, temporal_prior, anchor_indices, target_time):
        batch, seq_len, channels, height, width = evidence_seq.shape
        anchor0, anchor1 = self._gather_anchors(evidence_seq, anchor_indices)
        signed_pair = anchor1 - anchor0
        abs_pair = torch.abs(signed_pair)
        target_map = self._target_map(target_time, height, width).to(
            device=evidence_seq.device,
            dtype=evidence_seq.dtype,
        )

        query = self.query(torch.cat([anchor0, anchor1, signed_pair, abs_pair, target_map], dim=1))
        query_seq = query.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        metadata_maps = self._metadata_maps(metadata, height, width).to(
            device=evidence_seq.device,
            dtype=evidence_seq.dtype,
        )

        score_input = torch.cat([evidence_seq, query_seq, evidence_seq - query_seq, metadata_maps], dim=2)
        score_input = score_input.reshape(
            batch * seq_len,
            channels * 3 + self.metadata_channels,
            height,
            width,
        )
        logits = self.score(score_input).reshape(batch, seq_len, 1, height, width)
        prior_bias = torch.log(temporal_prior.to(dtype=evidence_seq.dtype).clamp_min(1.0e-6))
        logits = logits + prior_bias.view(batch, seq_len, 1, 1, 1)

        valid = metadata[..., -1].view(batch, seq_len, 1, 1, 1).to(
            device=evidence_seq.device,
            dtype=torch.bool,
        )
        logits = logits.masked_fill(~valid, -1.0e4)
        attention = torch.softmax(logits, dim=1)
        read = torch.sum(evidence_seq * attention, dim=1)

        context = self.context(torch.cat([anchor0, anchor1, signed_pair, abs_pair, read, query], dim=1))
        gate = self.gate(torch.cat([context, read, abs_pair], dim=1))
        return context * gate, gate, attention


class SoftDeformableAligner(nn.Module):
    """
    Learns a soft target-frame alignment without requiring external optical flow.

    For each local frame and target pixel, it predicts an offset and a confidence
    logit, samples the frame feature with grid_sample, and aggregates across time.
    """

    def __init__(self, channels: int, metadata_channels: int, max_offset: float = 3.0):
        super().__init__()
        self.metadata_channels = int(metadata_channels)
        self.max_offset = float(max_offset)
        self.offset_score = nn.Sequential(
            ConvNormAct(channels * 3 + metadata_channels, channels),
            ResidualConvBlock(channels),
            nn.Conv2d(channels, 3, 3, padding=1),
        )
        nn.init.zeros_(self.offset_score[-1].weight)
        nn.init.zeros_(self.offset_score[-1].bias)

    @staticmethod
    def _metadata_maps(metadata, height: int, width: int):
        return metadata.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, height, width)

    @staticmethod
    def _base_grid(height: int, width: int, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).view(1, height, width, 2)

    def forward(self, frame_seq, target_query, metadata, temporal_prior):
        batch, seq_len, channels, height, width = frame_seq.shape
        query_seq = target_query.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        metadata_maps = self._metadata_maps(metadata, height, width).to(
            device=frame_seq.device,
            dtype=frame_seq.dtype,
        )
        x = torch.cat([frame_seq, query_seq, frame_seq - query_seq, metadata_maps], dim=2)
        x = x.reshape(batch * seq_len, channels * 3 + self.metadata_channels, height, width)
        offset_score = self.offset_score(x)
        offsets = torch.tanh(offset_score[:, :2]) * self.max_offset
        logits = offset_score[:, 2:3].reshape(batch, seq_len, 1, height, width)

        base_grid = self._base_grid(height, width, frame_seq.device, frame_seq.dtype)
        norm_x = offsets[:, 0] * (2.0 / max(width - 1, 1))
        norm_y = offsets[:, 1] * (2.0 / max(height - 1, 1))
        sampling_grid = base_grid + torch.stack([norm_x, norm_y], dim=-1)

        aligned = F.grid_sample(
            frame_seq.reshape(batch * seq_len, channels, height, width),
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(batch, seq_len, channels, height, width)

        prior_bias = torch.log(temporal_prior.to(dtype=frame_seq.dtype).clamp_min(1.0e-6))
        logits = logits + prior_bias.view(batch, seq_len, 1, 1, 1)
        valid = metadata[..., -1].view(batch, seq_len, 1, 1, 1).to(
            device=frame_seq.device,
            dtype=torch.bool,
        )
        logits = logits.masked_fill(~valid, -1.0e4)
        attention = torch.softmax(logits, dim=1)

        aligned_read = torch.sum(aligned * attention, dim=1)
        offsets = offsets.reshape(batch, seq_len, 2, height, width)
        expected_offset = torch.sum(offsets * attention, dim=1)
        return aligned_read, attention, expected_offset


class TargetAlignedDetailFusion(nn.Module):
    """
    Converts local detail evidence into target-frame-aligned structure context.
    """

    def __init__(self, channels: int, metadata_channels: int, max_offset: float = 3.0):
        super().__init__()
        self.query = nn.Sequential(
            ConvNormAct(channels * 4 + 1, channels),
            ResidualConvBlock(channels),
        )
        self.aligner = SoftDeformableAligner(
            channels,
            metadata_channels=metadata_channels,
            max_offset=max_offset,
        )
        self.context = nn.Sequential(
            ConvNormAct(channels * 6, channels),
            ResidualConvBlock(channels),
        )
        self.gate = nn.Sequential(
            ConvNormAct(channels * 3, channels),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _gather_anchors(feat_seq, anchor_indices):
        batch = feat_seq.shape[0]
        batch_index = torch.arange(batch, device=feat_seq.device)
        anchor0 = feat_seq[batch_index, anchor_indices[:, 0]]
        anchor1 = feat_seq[batch_index, anchor_indices[:, 1]]
        return anchor0, anchor1

    @staticmethod
    def _target_map(target_time, height: int, width: int):
        return target_time.view(target_time.shape[0], 1, 1, 1).expand(-1, 1, height, width)

    def forward(self, evidence_seq, metadata, temporal_prior, anchor_indices, target_time):
        batch, _, _, height, width = evidence_seq.shape
        anchor0, anchor1 = self._gather_anchors(evidence_seq, anchor_indices)
        signed_pair = anchor1 - anchor0
        abs_pair = torch.abs(signed_pair)
        target_map = self._target_map(target_time, height, width).to(
            device=evidence_seq.device,
            dtype=evidence_seq.dtype,
        )

        target_query = self.query(
            torch.cat([anchor0, anchor1, signed_pair, abs_pair, target_map], dim=1)
        )
        aligned_read, attention, expected_offset = self.aligner(
            evidence_seq,
            target_query,
            metadata,
            temporal_prior,
        )

        context = self.context(
            torch.cat([anchor0, anchor1, signed_pair, abs_pair, aligned_read, target_query], dim=1)
        )
        gate = self.gate(torch.cat([context, aligned_read, abs_pair], dim=1))
        return context * gate, gate, attention, expected_offset


class LTCEv2(nn.Module):
    """
    Target-aware local temporal context encoder for sequence-conditioned VFI.

    This prototype keeps LTCE local and dense:
    - motion_context_pyr is target-centered local motion evidence for MTMC.
    - detail_context_pyr is target-aligned local structure evidence for TCAR.

    It intentionally does not depend on global memory or project-specific
    integration code. The file is meant as a standalone design implementation.
    """

    metadata_channels = 7

    def __init__(
        self,
        feat_channels: int,
        image_channels: int = 3,
        temporal_sigma: float = 2.0,
        channel_multipliers=(1, 2, 4),
        anchor_prior_strength: float = 0.25,
        detail_max_offset: float = 3.0,
    ):
        super().__init__()
        self.image_channels = int(image_channels)
        self.temporal_sigma = float(temporal_sigma)
        self.anchor_prior_strength = float(anchor_prior_strength)
        self.channel_multipliers = tuple(int(multiplier) for multiplier in channel_multipliers)
        self.channel_list = [int(feat_channels) * multiplier for multiplier in self.channel_multipliers]
        self.num_levels = len(self.channel_list)
        if self.num_levels <= 0:
            raise ValueError("LTCEv2 requires at least one pyramid level.")

        self.raw_projs = nn.ModuleList(
            [self._make_raw_proj(self.image_channels, ch) for ch in self.channel_list]
        )
        self.motion_frame_adapters = nn.ModuleList(
            [FrameAdapter(ch, self.metadata_channels) for ch in self.channel_list]
        )
        self.detail_frame_adapters = nn.ModuleList(
            [FrameAdapter(ch, self.metadata_channels) for ch in self.channel_list]
        )
        self.motion_evidence_blocks = nn.ModuleList(
            [MotionEvidenceBlock(ch, self.metadata_channels) for ch in self.channel_list]
        )
        self.detail_evidence_blocks = nn.ModuleList(
            [DetailEvidenceBlock(ch, self.metadata_channels) for ch in self.channel_list]
        )
        self.motion_fusions = nn.ModuleList(
            [TargetCenteredMotionFusion(ch, self.metadata_channels) for ch in self.channel_list]
        )
        self.detail_fusions = nn.ModuleList(
            [
                TargetAlignedDetailFusion(
                    ch,
                    self.metadata_channels,
                    max_offset=detail_max_offset,
                )
                for ch in self.channel_list
            ]
        )

    @staticmethod
    def _make_raw_proj(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            ConvNormAct(in_channels, out_channels),
            ResidualConvBlock(out_channels),
        )

    @staticmethod
    def _reshape_scalar(value, batch_size: int, device, dtype):
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=device, dtype=dtype)
        else:
            value = value.to(device=device, dtype=dtype)

        if value.dim() == 0:
            value = value.view(1).repeat(batch_size)
        elif value.numel() == 1 and batch_size > 1:
            value = value.reshape(1).repeat(batch_size)
        else:
            value = value.reshape(batch_size)
        return value

    @staticmethod
    def _reshape_anchor_indices(anchor_indices, batch_size: int, seq_len: int, device):
        if anchor_indices is None:
            raise ValueError("LTCEv2 requires anchor_indices for local window fusion.")
        anchor_indices = anchor_indices.to(device=device, dtype=torch.long)
        if anchor_indices.dim() == 1:
            anchor_indices = anchor_indices.view(1, -1).expand(batch_size, -1)
        if anchor_indices.shape[1] < 2:
            raise ValueError(
                f"anchor_indices must contain at least two indices, got {tuple(anchor_indices.shape)}"
            )
        return anchor_indices[:, :2].clamp(min=0, max=max(seq_len - 1, 0))

    @staticmethod
    def _reshape_frame_positions(frame_positions, batch_size: int, seq_len: int, device, dtype):
        if frame_positions is None:
            return None
        return frame_positions.to(device=device, dtype=dtype).reshape(batch_size, seq_len)

    @staticmethod
    def _reshape_feature_sequence(feat, batch_size: int, seq_len: int, stream_name: str, level: int):
        if feat.dim() == 5:
            if feat.shape[0] != batch_size or feat.shape[1] != seq_len:
                raise ValueError(
                    f"{stream_name} feature level {level} shape {tuple(feat.shape)} does not match "
                    f"local window batch/seq ({batch_size}, {seq_len}, ...)."
                )
            return feat
        if feat.dim() == 4:
            return feat.reshape(batch_size, seq_len, *feat.shape[1:])
        raise ValueError(
            f"{stream_name} feature level {level} must be 4D or 5D, got {feat.dim()} "
            f"for shape {tuple(feat.shape)}."
        )

    def _build_relative_positions(
        self,
        batch_size: int,
        seq_len: int,
        device,
        dtype,
        anchor_indices,
        frame_relative_positions=None,
    ):
        frame_relative_positions = self._reshape_frame_positions(
            frame_relative_positions, batch_size, seq_len, device, dtype
        )
        if frame_relative_positions is not None:
            return frame_relative_positions

        positions = torch.arange(seq_len, device=device, dtype=dtype).view(1, seq_len)
        anchor_start = anchor_indices.to(device=device, dtype=dtype)[:, 0:1]
        return positions - anchor_start

    def _build_temporal_metadata(
        self,
        batch_size: int,
        seq_len: int,
        device,
        dtype,
        time_step,
        frame_mask,
        anchor_indices,
        frame_relative_positions=None,
        target_local_position=None,
    ):
        if frame_mask is None:
            valid = torch.ones((batch_size, seq_len), device=device, dtype=dtype)
        else:
            valid = frame_mask.to(device=device, dtype=dtype)

        relative_positions = self._build_relative_positions(
            batch_size,
            seq_len,
            device,
            dtype,
            anchor_indices,
            frame_relative_positions=frame_relative_positions,
        )
        if target_local_position is None:
            target_time = self._reshape_scalar(time_step, batch_size, device, dtype)
        else:
            target_time = self._reshape_scalar(target_local_position, batch_size, device, dtype)
        target_time = target_time.view(batch_size, 1)

        delta = relative_positions - target_time
        abs_delta = torch.abs(delta)
        anchor0_flag = torch.zeros_like(valid)
        anchor1_flag = torch.zeros_like(valid)
        anchor0_flag.scatter_(1, anchor_indices[:, 0:1], 1.0)
        anchor1_flag.scatter_(1, anchor_indices[:, 1:2], 1.0)

        metadata = torch.stack(
            [
                relative_positions,
                target_time.expand(-1, seq_len),
                delta,
                abs_delta,
                anchor0_flag,
                anchor1_flag,
                valid,
            ],
            dim=-1,
        )
        metadata = metadata * valid.unsqueeze(-1)

        sigma = max(self.temporal_sigma, 1.0e-6)
        prior = torch.exp(-0.5 * (delta / sigma) ** 2) * valid
        if self.anchor_prior_strength > 0:
            prior = prior + self.anchor_prior_strength * (anchor0_flag + anchor1_flag) * valid
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return metadata, prior, relative_positions, target_time

    def forward(
        self,
        local_clip,
        motion_feat_seq_pyr,
        detail_feat_seq_pyr,
        time_step,
        frame_mask=None,
        anchor_indices=None,
        frame_relative_positions=None,
        target_local_position=None,
        return_debug: bool = False,
    ):
        if local_clip.dim() != 5:
            raise ValueError(f"LTCEv2 expects local_clip [B, T, C, H, W], got {local_clip.shape}")
        if local_clip.shape[2] != self.image_channels:
            raise ValueError(
                f"LTCEv2 was initialized for {self.image_channels} image channels, "
                f"but local_clip has {local_clip.shape[2]}."
            )
        if len(motion_feat_seq_pyr) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} motion pyramid levels, got {len(motion_feat_seq_pyr)}."
            )
        if len(detail_feat_seq_pyr) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} detail pyramid levels, got {len(detail_feat_seq_pyr)}."
            )

        batch_size, seq_len, channels, height, width = local_clip.shape
        anchor_indices = self._reshape_anchor_indices(
            anchor_indices,
            batch_size,
            seq_len,
            local_clip.device,
        )
        clip_flat = local_clip.reshape(batch_size * seq_len, channels, height, width)
        metadata, temporal_prior, relative_positions, target_time = self._build_temporal_metadata(
            batch_size=batch_size,
            seq_len=seq_len,
            device=local_clip.device,
            dtype=local_clip.dtype,
            time_step=time_step,
            frame_mask=frame_mask,
            anchor_indices=anchor_indices,
            frame_relative_positions=frame_relative_positions,
            target_local_position=target_local_position,
        )

        motion_context_pyr = []
        detail_context_pyr = []
        motion_gate_pyr = []
        detail_gate_pyr = []
        debug = {
            "motion_attention_pyr": [],
            "detail_attention_pyr": [],
            "detail_expected_offset_pyr": [],
        }

        for level, (
            motion_feat,
            detail_feat,
            raw_proj,
            motion_adapter,
            detail_adapter,
            motion_evidence,
            detail_evidence,
            motion_fusion,
            detail_fusion,
        ) in enumerate(
            zip(
                motion_feat_seq_pyr,
                detail_feat_seq_pyr,
                self.raw_projs,
                self.motion_frame_adapters,
                self.detail_frame_adapters,
                self.motion_evidence_blocks,
                self.detail_evidence_blocks,
                self.motion_fusions,
                self.detail_fusions,
            )
        ):
            motion_seq = self._reshape_feature_sequence(
                motion_feat, batch_size, seq_len, "motion", level
            )
            detail_seq = self._reshape_feature_sequence(
                detail_feat, batch_size, seq_len, "detail", level
            )
            if motion_seq.shape != detail_seq.shape:
                raise ValueError(
                    f"Motion/detail feature level {level} shapes must match, got "
                    f"{tuple(motion_seq.shape)} and {tuple(detail_seq.shape)}."
                )

            level_h, level_w = motion_seq.shape[-2:]
            raw_level = F.interpolate(
                clip_flat,
                size=(level_h, level_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            raw_level = raw_proj(raw_level).reshape(batch_size, seq_len, -1, level_h, level_w)
            metadata_level = metadata.to(device=motion_seq.device, dtype=motion_seq.dtype)
            prior_level = temporal_prior.to(device=motion_seq.device, dtype=motion_seq.dtype)
            target_time_level = target_time.to(device=motion_seq.device, dtype=motion_seq.dtype)
            raw_level = raw_level.to(device=motion_seq.device, dtype=motion_seq.dtype)

            motion_frames = motion_adapter(motion_seq, raw_level, metadata_level)
            detail_frames = detail_adapter(detail_seq, raw_level, metadata_level)

            motion_evidence_seq = motion_evidence(motion_frames, metadata_level, anchor_indices)
            detail_evidence_seq = detail_evidence(detail_frames, metadata_level, anchor_indices)

            motion_context, motion_gate, motion_attention = motion_fusion(
                motion_evidence_seq,
                metadata_level,
                prior_level,
                anchor_indices,
                target_time_level,
            )
            detail_context, detail_gate, detail_attention, detail_expected_offset = detail_fusion(
                detail_evidence_seq,
                metadata_level,
                prior_level,
                anchor_indices,
                target_time_level,
            )

            motion_context_pyr.append(motion_context)
            detail_context_pyr.append(detail_context)
            motion_gate_pyr.append(motion_gate)
            detail_gate_pyr.append(detail_gate)
            if return_debug:
                debug["motion_attention_pyr"].append(motion_attention)
                debug["detail_attention_pyr"].append(detail_attention)
                debug["detail_expected_offset_pyr"].append(detail_expected_offset)

        result = {
            "motion_context_pyr": motion_context_pyr,
            "detail_context_pyr": detail_context_pyr,
            "motion_gate_pyr": motion_gate_pyr,
            "detail_gate_pyr": detail_gate_pyr,
        }
        if return_debug:
            debug.update(
                {
                    "temporal_prior": temporal_prior,
                    "relative_positions": relative_positions,
                    "target_time": target_time,
                    "metadata": metadata,
                }
            )
            result["debug"] = debug
        return result


LocalTemporalContextEncoderV2 = LTCEv2
TargetAwareLocalTemporalContextEncoder = LTCEv2
