import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_encoder import BasicBlock, ResNetPyramid


class GTME(nn.Module):
    """
    Target-centered global memory extractor for sequence-conditioned VFI.

    This module is deliberately non-causal: every valid observed frame in the
    sparse input sequence can contribute to the target memory, including frames
    after the interpolation interval. It returns a 3-level context pyramid that
    mirrors ResNetPyramid channel sizes:
    - level 0: [B, C, H, W]
    - level 1: [B, 2C, H/2, W/2]
    - level 2: [B, 4C, H/4, W/4]
    """

    def __init__(
        self,
        feat_channels: int,
        local_window_size: int = 6,
        metadata_channels: int = 8,
        score_hidden_channels: int = 64,
        spatial_downsample_factor: int = 1,
    ):
        super(GTME, self).__init__()
        self.local_window_size = int(local_window_size)
        self.metadata_channels = int(metadata_channels)
        self.spatial_downsample_factor = int(spatial_downsample_factor)
        if self.spatial_downsample_factor < 1:
            raise ValueError(
                f"spatial_downsample_factor must be >= 1, got {spatial_downsample_factor}."
            )
        self.encoder = ResNetPyramid(feat_channels)

        channel_list = [feat_channels, feat_channels * 2, feat_channels * 4]
        self.frame_fusions = nn.ModuleList(
            [self._make_frame_fusion(ch, self.metadata_channels) for ch in channel_list]
        )
        self.metadata_mlps = nn.ModuleList(
            [self._make_metadata_mlp(ch, self.metadata_channels) for ch in channel_list]
        )
        self.score_heads = nn.ModuleList(
            [self._make_score_head(ch, score_hidden_channels) for ch in channel_list]
        )
        self.context_fusions = nn.ModuleList(
            [self._make_context_fusion(ch) for ch in channel_list]
        )

    @staticmethod
    def _make_frame_fusion(channels: int, metadata_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels + metadata_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(channels),
            BasicBlock(channels, channels, norm_layer=nn.InstanceNorm2d),
        )

    @staticmethod
    def _make_metadata_mlp(channels: int, metadata_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(metadata_channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels),
        )

    @staticmethod
    def _make_score_head(channels: int, hidden_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, 1),
        )

    @staticmethod
    def _make_context_fusion(channels: int) -> nn.Module:
        # anchor0, anchor1, abs-diff, local read, global read, target-position map
        return nn.Sequential(
            nn.Conv2d(channels * 5 + 1, channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(channels),
            BasicBlock(channels, channels, norm_layer=nn.InstanceNorm2d),
        )

    @staticmethod
    def _reshape_target_position(target_position, batch_size: int, device, dtype):
        if not torch.is_tensor(target_position):
            target_position = torch.tensor(target_position, device=device, dtype=dtype)
        else:
            target_position = target_position.to(device=device, dtype=dtype)

        if target_position.dim() == 0:
            target_position = target_position.view(1).repeat(batch_size)
        elif target_position.numel() == 1 and batch_size > 1:
            target_position = target_position.reshape(1).repeat(batch_size)
        else:
            target_position = target_position.reshape(batch_size)
        return target_position

    @staticmethod
    def _default_mask(batch_size: int, seq_len: int, device, dtype):
        return torch.ones((batch_size, seq_len), device=device, dtype=dtype)

    @staticmethod
    def _clamp_anchor_indices(anchor_indices, batch_size: int, seq_len: int, device):
        if anchor_indices is None:
            return None
        anchor_indices = anchor_indices.to(device=device, dtype=torch.long)
        if anchor_indices.dim() == 1:
            anchor_indices = anchor_indices.view(1, -1).expand(batch_size, -1)
        return anchor_indices[:, :2].clamp(min=0, max=max(seq_len - 1, 0))

    def _build_temporal_metadata(
        self,
        frame_positions,
        target_position,
        frame_mask,
        anchor_indices,
    ):
        batch_size, seq_len = frame_positions.shape
        device = frame_positions.device
        dtype = frame_positions.dtype

        target = target_position.view(batch_size, 1).expand(batch_size, seq_len)
        delta = frame_positions - target
        abs_delta = torch.abs(delta)
        left_flag = (delta <= 0).to(dtype)
        right_flag = (delta >= 0).to(dtype)
        anchor0_flag = torch.zeros((batch_size, seq_len), device=device, dtype=dtype)
        anchor1_flag = torch.zeros((batch_size, seq_len), device=device, dtype=dtype)

        if anchor_indices is not None:
            anchor0_flag.scatter_(1, anchor_indices[:, 0:1], 1.0)
            anchor1_flag.scatter_(1, anchor_indices[:, 1:2], 1.0)

        metadata = torch.stack(
            [
                frame_positions,
                target,
                delta,
                abs_delta,
                left_flag,
                right_flag,
                anchor0_flag,
                anchor1_flag,
            ],
            dim=-1,
        )
        return metadata * frame_mask.unsqueeze(-1)

    @staticmethod
    def _metadata_maps(metadata, height: int, width: int):
        return metadata.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, height, width)

    @staticmethod
    def _masked_softmax(scores, mask):
        mask = mask.to(device=scores.device, dtype=scores.dtype)
        scores = scores.masked_fill(mask <= 0, -1.0e4)
        weights = torch.softmax(scores, dim=1) * mask
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

    def _build_local_mask(self, metadata, frame_mask, anchor_indices):
        if self.local_window_size <= 0 or self.local_window_size >= metadata.shape[1]:
            return frame_mask

        batch_size, seq_len = frame_mask.shape
        abs_delta = metadata[..., 3].masked_fill(frame_mask <= 0, float("inf"))
        topk = min(self.local_window_size, seq_len)
        local_indices = torch.topk(abs_delta, k=topk, dim=1, largest=False).indices
        local_mask = torch.zeros_like(frame_mask)
        local_mask.scatter_(1, local_indices, 1.0)

        if anchor_indices is not None:
            local_mask.scatter_(1, anchor_indices[:, 0:1], 1.0)
            local_mask.scatter_(1, anchor_indices[:, 1:2], 1.0)

        return local_mask * frame_mask

    @staticmethod
    def _weighted_sum(feat_seq, weights):
        return torch.sum(
            feat_seq * weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            dim=1,
        )

    @staticmethod
    def _gather_anchor_features(feat_seq, anchor_indices):
        batch_size = feat_seq.shape[0]
        batch_index = torch.arange(batch_size, device=feat_seq.device)
        if anchor_indices is None:
            return feat_seq[:, 0], feat_seq[:, -1]
        return feat_seq[batch_index, anchor_indices[:, 0]], feat_seq[batch_index, anchor_indices[:, 1]]

    def _downsample_global_clip(self, global_clip):
        factor = self.spatial_downsample_factor
        if factor == 1:
            return global_clip

        batch_size, seq_len, channels, height, width = global_clip.shape
        target_h = max(1, height // factor)
        target_w = max(1, width // factor)
        if target_h == height and target_w == width:
            return global_clip

        clip_flat = global_clip.reshape(batch_size * seq_len, channels, height, width)
        clip_flat = F.interpolate(
            clip_flat,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return clip_flat.reshape(batch_size, seq_len, channels, target_h, target_w)

    def forward(
        self,
        global_clip,
        global_frame_positions,
        target_global_position,
        global_frame_mask=None,
        global_anchor_indices=None,
    ):
        if global_clip.dim() != 5:
            raise ValueError(f"GTME expects [B, T, C, H, W], got {global_clip.shape}")

        global_clip = self._downsample_global_clip(global_clip)
        batch_size, seq_len, channels, height, width = global_clip.shape
        device = global_clip.device
        dtype = global_clip.dtype

        if global_frame_mask is None:
            global_frame_mask = self._default_mask(batch_size, seq_len, device, dtype)
        else:
            global_frame_mask = global_frame_mask.to(device=device, dtype=dtype)

        global_frame_positions = global_frame_positions.to(device=device, dtype=dtype).reshape(batch_size, seq_len)
        target_global_position = self._reshape_target_position(
            target_global_position, batch_size, device, dtype
        )
        global_anchor_indices = self._clamp_anchor_indices(
            global_anchor_indices, batch_size, seq_len, device
        )

        metadata = self._build_temporal_metadata(
            frame_positions=global_frame_positions,
            target_position=target_global_position,
            frame_mask=global_frame_mask,
            anchor_indices=global_anchor_indices,
        )
        local_mask = self._build_local_mask(metadata, global_frame_mask, global_anchor_indices)

        clip_flat = global_clip.reshape(batch_size * seq_len, channels, height, width)
        feat_flat_pyr = self.encoder(clip_flat)

        context_pyr = []
        for feat_flat, frame_fusion, metadata_mlp, score_head, context_fusion in zip(
            feat_flat_pyr,
            self.frame_fusions,
            self.metadata_mlps,
            self.score_heads,
            self.context_fusions,
        ):
            level_channels, level_h, level_w = feat_flat.shape[1:]
            feat_seq = feat_flat.reshape(batch_size, seq_len, level_channels, level_h, level_w)
            metadata_maps = self._metadata_maps(metadata, level_h, level_w).to(
                device=feat_seq.device,
                dtype=feat_seq.dtype,
            )

            fused_seq = torch.cat([feat_seq, metadata_maps], dim=2)
            fused_seq = fused_seq.reshape(batch_size * seq_len, level_channels + self.metadata_channels, level_h, level_w)
            fused_seq = frame_fusion(fused_seq).reshape(batch_size, seq_len, level_channels, level_h, level_w)
            fused_seq = fused_seq * global_frame_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            tokens = fused_seq.mean(dim=(-1, -2)) + metadata_mlp(metadata.to(dtype=fused_seq.dtype))
            scores = score_head(tokens).squeeze(-1)
            global_weights = self._masked_softmax(scores, global_frame_mask)
            local_weights = self._masked_softmax(scores, local_mask)

            global_read = self._weighted_sum(fused_seq, global_weights)
            local_read = self._weighted_sum(fused_seq, local_weights)
            anchor0_feat, anchor1_feat = self._gather_anchor_features(fused_seq, global_anchor_indices)
            target_map = target_global_position.view(batch_size, 1, 1, 1).expand(
                batch_size, 1, level_h, level_w
            )

            context = context_fusion(
                torch.cat(
                    [
                        anchor0_feat,
                        anchor1_feat,
                        torch.abs(anchor1_feat - anchor0_feat),
                        local_read,
                        global_read,
                        target_map,
                    ],
                    dim=1,
                )
            )
            context_pyr.append(context)

        return context_pyr


GlobalTemporalMemoryEncoder = GTME
