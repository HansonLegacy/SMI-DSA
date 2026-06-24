import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_encoder import BasicBlock


class LocalVideoContextPyramid(nn.Module):
    """
    Build a pair-conditioned multi-scale context pyramid from:
    1. multi-frame CFE features,
    2. raw support frames in the local sparse window, and
    3. the interpolation time t.

    The support window is expected to contain sparse frames around the current pair,
    e.g. for window=4: [prev, img0, img1, next]. The output stays multi-scale so it
    can be injected into a dedicated sequence-aware synthesis network.
    """

    def __init__(self, feat_channels: int, temporal_sigma: float = 2.0):
        super(LocalVideoContextPyramid, self).__init__()
        self.temporal_sigma = float(temporal_sigma)

        channel_list = [feat_channels, feat_channels * 2, feat_channels * 4]
        self.raw_projs = nn.ModuleList([self._make_raw_proj(ch) for ch in channel_list])
        self.frame_fusions = nn.ModuleList([self._make_frame_fusion(ch) for ch in channel_list])
        self.context_fusions = nn.ModuleList([self._make_context_fusion(ch) for ch in channel_list])

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

    @staticmethod
    def _reshape_time_step(time_step, batch_size: int, device, dtype):
        if not torch.is_tensor(time_step):
            time_step = torch.tensor(time_step, device=device, dtype=dtype)
        else:
            time_step = time_step.to(device=device, dtype=dtype)

        if time_step.dim() == 0:
            time_step = time_step.view(1).repeat(batch_size)
        elif time_step.dim() == 1 and time_step.shape[0] == 1 and batch_size > 1:
            time_step = time_step.repeat(batch_size)
        else:
            time_step = time_step.reshape(batch_size)
        return time_step

    @staticmethod
    def _build_relative_positions(batch_size: int, seq_len: int, device, dtype, anchor_indices):
        if anchor_indices is None:
            raise ValueError("LocalVideoContextPyramid requires anchor_indices for sparse support windows.")

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
        frame_mask=None,
        anchor_indices=None,
    ):
        if frame_mask is None:
            weights = torch.ones((batch_size, seq_len), device=device, dtype=dtype)
        else:
            weights = frame_mask.to(device=device, dtype=dtype)

        relative_positions = self._build_relative_positions(
            batch_size, seq_len, device, dtype, anchor_indices
        )
        target_time = self._reshape_time_step(time_step, batch_size, device, dtype).view(batch_size, 1)

        sigma = max(self.temporal_sigma, 1e-6)
        local_weights = torch.exp(-0.5 * ((relative_positions - target_time) / sigma) ** 2)
        weights = weights * local_weights

        if anchor_indices is not None:
            anchor_indices = anchor_indices.to(device=device, dtype=torch.long)
            anchor_boost = torch.zeros_like(weights)
            anchor_boost.scatter_(1, anchor_indices[:, 0:1], 1.0)
            anchor_boost.scatter_(1, anchor_indices[:, 1:2], 1.0)
            weights = weights + anchor_boost

        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return weights, relative_positions, target_time

    @staticmethod
    def _build_time_maps(relative_positions, target_time, height: int, width: int):
        rel_map = relative_positions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        target_map = target_time.view(target_time.shape[0], 1, 1, 1).expand(-1, rel_map.shape[1], height, width)
        delta_map = rel_map - target_map
        return torch.stack([rel_map, target_map, delta_map], dim=2)

    def forward(self, clip, cfeat_seq_pyr, time_step, frame_mask=None, anchor_indices=None):
        if clip.dim() != 5:
            raise ValueError(f"LocalVideoContextPyramid expects [B, T, C, H, W], got {clip.shape}")
        if len(cfeat_seq_pyr) != 3:
            raise ValueError("LocalVideoContextPyramid expects a 3-level CFE pyramid sequence.")
        if anchor_indices is None:
            raise ValueError("LocalVideoContextPyramid requires anchor_indices for sparse support fusion.")

        if anchor_indices.dim() == 1:
            anchor_indices = anchor_indices.unsqueeze(0)

        batch_size, seq_len, channels, height, width = clip.shape
        clip_flat = clip.reshape(batch_size * seq_len, channels, height, width)

        weights, relative_positions, target_time = self._build_temporal_weights(
            batch_size=batch_size,
            seq_len=seq_len,
            device=clip.device,
            dtype=clip.dtype,
            time_step=time_step,
            frame_mask=frame_mask,
            anchor_indices=anchor_indices,
        )

        batch_index = torch.arange(batch_size, device=clip.device)
        context_pyr = []

        for feat_flat, raw_proj, frame_fusion, context_fusion in zip(
            cfeat_seq_pyr, self.raw_projs, self.frame_fusions, self.context_fusions
        ):
            if feat_flat.dim() == 5:
                feat_seq = feat_flat
                if feat_seq.shape[0] != batch_size or feat_seq.shape[1] != seq_len:
                    raise ValueError(
                        f"CFE sequence shape {tuple(feat_seq.shape)} does not match clip batch/seq "
                        f"({batch_size}, {seq_len}, ...)."
                    )
            elif feat_flat.dim() == 4:
                feat_seq = feat_flat.reshape(batch_size, seq_len, *feat_flat.shape[1:])
            else:
                raise ValueError(
                    f"LocalVideoContextPyramid expects CFE tensors with 4 or 5 dims, got {feat_flat.dim()} "
                    f"for shape {tuple(feat_flat.shape)}."
                )
            level_h, level_w = feat_seq.shape[-2:]

            raw_level = F.interpolate(
                clip_flat,
                size=(level_h, level_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            raw_level = raw_proj(raw_level).reshape(batch_size, seq_len, -1, level_h, level_w)

            time_maps = self._build_time_maps(relative_positions, target_time, level_h, level_w)
            time_maps = time_maps.to(device=feat_seq.device, dtype=feat_seq.dtype)

            fused_input = torch.cat([feat_seq, raw_level, time_maps], dim=2)
            fused_input = fused_input.reshape(batch_size * seq_len, fused_input.shape[2], level_h, level_w)
            fused_seq = frame_fusion(fused_input).reshape(batch_size, seq_len, -1, level_h, level_w)

            anchor0_feat = fused_seq[batch_index, anchor_indices[:, 0]]
            anchor1_feat = fused_seq[batch_index, anchor_indices[:, 1]]
            temporal_context = torch.sum(
                fused_seq * weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                dim=1,
            )

            fused_context = context_fusion(
                torch.cat(
                    [anchor0_feat, anchor1_feat, torch.abs(anchor1_feat - anchor0_feat), temporal_context],
                    dim=1,
                )
            )
            context_pyr.append(fused_context)

        return context_pyr
