import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_encoder import BasicBlock


class LTCE(nn.Module):
    """
    Local fine-grained context extractor for sequence-conditioned VFI.

    Expected data flow:
    1. local_clip [B, T, 3, H, W] is flattened outside this module.
    2. The flattened local frames are passed through MFE and CFE separately.
    3. Their multi-scale outputs are reshaped to [B, T, C_l, H_l, W_l].
    4. This module fuses temporal/window context into two pyramids:
       - motion_context_pyr: for motion/BiM-prior related branches
       - detail_context_pyr: for detail/SN related branches

    Coarse levels can share parameters between motion and detail streams to
    reduce parameter count, while fine levels keep separate parameters so each
    stream can adapt to its own context needs.
    """

    def __init__(
        self,
        feat_channels: int,
        temporal_sigma: float = 2.0,
        channel_multipliers=(1, 2, 4),
        coarse_shared_levels=(2,),
        fine_separate_levels=(0, 1),
    ):
        super(LTCE, self).__init__()
        self.temporal_sigma = float(temporal_sigma)
        self.channel_multipliers = tuple(int(multiplier) for multiplier in channel_multipliers)
        self.channel_list = [int(feat_channels) * multiplier for multiplier in self.channel_multipliers]
        self.num_levels = len(self.channel_list)

        if self.num_levels <= 0:
            raise ValueError("LTCE requires at least one pyramid level.")

        self.coarse_shared_levels = self._normalize_level_set(coarse_shared_levels, self.num_levels)
        requested_fine_levels = self._normalize_level_set(fine_separate_levels, self.num_levels)
        self.fine_separate_levels = requested_fine_levels | (
            set(range(self.num_levels)) - self.coarse_shared_levels
        )
        overlap = self.coarse_shared_levels & requested_fine_levels
        if overlap:
            raise ValueError(
                f"coarse_shared_levels and fine_separate_levels overlap at levels {sorted(overlap)}"
            )

        self.raw_projs = nn.ModuleList([self._make_raw_proj(channels) for channels in self.channel_list])
        self.level_fusions = nn.ModuleList(
            [self._make_level_fusions(level, channels) for level, channels in enumerate(self.channel_list)]
        )

    @staticmethod
    def _normalize_level_set(levels, num_levels: int):
        if levels is None:
            return set()
        level_set = {int(level) for level in levels}
        invalid = [level for level in level_set if level < 0 or level >= num_levels]
        if invalid:
            raise ValueError(f"Invalid pyramid levels {invalid}; expected values in [0, {num_levels - 1}]")
        return level_set

    @staticmethod
    def _make_raw_proj(out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(out_channels),
        )

    @staticmethod
    def _make_frame_fusion(channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels * 2 + 3, channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(channels),
            BasicBlock(channels, channels, norm_layer=nn.InstanceNorm2d),
        )

    @staticmethod
    def _make_context_fusion(channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(channels),
            BasicBlock(channels, channels, norm_layer=nn.InstanceNorm2d),
        )

    def _make_stream_fusion(self, channels: int) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "frame": self._make_frame_fusion(channels),
                "context": self._make_context_fusion(channels),
            }
        )

    def _make_level_fusions(self, level: int, channels: int) -> nn.ModuleDict:
        if level in self.coarse_shared_levels:
            return nn.ModuleDict({"shared": self._make_stream_fusion(channels)})

        return nn.ModuleDict(
            {
                "motion": self._make_stream_fusion(channels),
                "detail": self._make_stream_fusion(channels),
            }
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
    def _reshape_anchor_indices(anchor_indices, batch_size: int, device):
        if anchor_indices is None:
            raise ValueError("LTCE requires anchor_indices for local window fusion.")
        anchor_indices = anchor_indices.to(device=device, dtype=torch.long)
        if anchor_indices.dim() == 1:
            anchor_indices = anchor_indices.view(1, -1).expand(batch_size, -1)
        return anchor_indices[:, :2]

    @staticmethod
    def _reshape_frame_positions(frame_positions, batch_size: int, seq_len: int, device, dtype):
        if frame_positions is None:
            return None
        return frame_positions.to(device=device, dtype=dtype).reshape(batch_size, seq_len)

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

    def _build_temporal_weights(
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
            weights = torch.ones((batch_size, seq_len), device=device, dtype=dtype)
        else:
            weights = frame_mask.to(device=device, dtype=dtype)

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

        sigma = max(self.temporal_sigma, 1.0e-6)
        local_weights = torch.exp(-0.5 * ((relative_positions - target_time) / sigma) ** 2)
        weights = weights * local_weights

        anchor_boost = torch.zeros_like(weights)
        anchor_boost.scatter_(1, anchor_indices[:, 0:1], 1.0)
        anchor_boost.scatter_(1, anchor_indices[:, 1:2], 1.0)
        weights = weights + anchor_boost
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return weights, relative_positions, target_time

    @staticmethod
    def _build_time_maps(relative_positions, target_time, height: int, width: int):
        rel_map = relative_positions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        target_map = target_time.view(target_time.shape[0], 1, 1, 1).expand(
            -1, rel_map.shape[1], height, width
        )
        delta_map = rel_map - target_map
        return torch.stack([rel_map, target_map, delta_map], dim=2)

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

    def _select_stream_fusion(self, level_fusion: nn.ModuleDict, stream_name: str):
        if "shared" in level_fusion:
            return level_fusion["shared"]
        return level_fusion[stream_name]

    def _fuse_stream_level(
        self,
        stream_name: str,
        feat,
        raw_level,
        time_maps,
        weights,
        anchor_indices,
        level_fusion,
    ):
        batch_size, seq_len, channels, level_h, level_w = feat.shape
        stream_fusion = self._select_stream_fusion(level_fusion, stream_name)

        fused_input = torch.cat([feat, raw_level, time_maps], dim=2)
        fused_input = fused_input.reshape(batch_size * seq_len, fused_input.shape[2], level_h, level_w)
        fused_seq = stream_fusion["frame"](fused_input).reshape(
            batch_size, seq_len, channels, level_h, level_w
        )

        batch_index = torch.arange(batch_size, device=feat.device)
        anchor0_feat = fused_seq[batch_index, anchor_indices[:, 0]]
        anchor1_feat = fused_seq[batch_index, anchor_indices[:, 1]]
        temporal_context = torch.sum(
            fused_seq * weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            dim=1,
        )

        return stream_fusion["context"](
            torch.cat(
                [anchor0_feat, anchor1_feat, torch.abs(anchor1_feat - anchor0_feat), temporal_context],
                dim=1,
            )
        )

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
    ):
        if local_clip.dim() != 5:
            raise ValueError(f"LTCE expects local_clip [B, T, C, H, W], got {local_clip.shape}")
        if len(motion_feat_seq_pyr) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} motion pyramid levels, got {len(motion_feat_seq_pyr)}."
            )
        if len(detail_feat_seq_pyr) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} detail pyramid levels, got {len(detail_feat_seq_pyr)}."
            )

        batch_size, seq_len, channels, height, width = local_clip.shape
        anchor_indices = self._reshape_anchor_indices(anchor_indices, batch_size, local_clip.device)
        clip_flat = local_clip.reshape(batch_size * seq_len, channels, height, width)

        weights, relative_positions, target_time = self._build_temporal_weights(
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

        for level, (motion_feat, detail_feat, raw_proj, level_fusion) in enumerate(
            zip(motion_feat_seq_pyr, detail_feat_seq_pyr, self.raw_projs, self.level_fusions)
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

            time_maps = self._build_time_maps(relative_positions, target_time, level_h, level_w)
            time_maps = time_maps.to(device=motion_seq.device, dtype=motion_seq.dtype)
            raw_level = raw_level.to(device=motion_seq.device, dtype=motion_seq.dtype)

            motion_context_pyr.append(
                self._fuse_stream_level(
                    "motion",
                    motion_seq,
                    raw_level,
                    time_maps,
                    weights.to(device=motion_seq.device, dtype=motion_seq.dtype),
                    anchor_indices,
                    level_fusion,
                )
            )
            detail_context_pyr.append(
                self._fuse_stream_level(
                    "detail",
                    detail_seq,
                    raw_level,
                    time_maps,
                    weights.to(device=detail_seq.device, dtype=detail_seq.dtype),
                    anchor_indices,
                    level_fusion,
                )
            )

        return {
            "motion_context_pyr": motion_context_pyr,
            "detail_context_pyr": detail_context_pyr,
        }


LocalTemporalContextEncoder = LTCE
