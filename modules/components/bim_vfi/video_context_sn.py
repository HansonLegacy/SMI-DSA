import torch
import torch.nn as nn
import torch.nn.functional as F

from .backwarp import backwarp
from .eamamba_blocks import ALLOWED_CHANNEL_MIXER_TYPE
from .eamamba_blocks import Downsample as EAMambaDownsample
from .eamamba_blocks import OverlapPatchEmbed as EAMambaOverlapPatchEmbed
from .eamamba_blocks import ScanTransform
from .eamamba_blocks import Upsample as EAMambaUpsample
from .eamamba_blocks import create_blocks as create_eamamba_blocks
from .sn import EAMambaSynthesisNetwork


def _context_level_or_zeros(video_context_pyr, level, ref_tensor, channels):
    if video_context_pyr is None or level >= len(video_context_pyr):
        return ref_tensor.new_zeros((ref_tensor.shape[0], channels, ref_tensor.shape[-2], ref_tensor.shape[-1]))

    context_feat = video_context_pyr[level]
    if context_feat.shape[-2:] != ref_tensor.shape[-2:]:
        context_feat = F.interpolate(
            context_feat, size=ref_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
    return context_feat


def _reshape_time_step(time_step, batch_size, device, dtype):
    if time_step is None:
        return torch.full((batch_size,), 0.5, device=device, dtype=dtype)
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


def _aggregate_rgb_context(video_frames, time_step, video_anchor_indices, video_frame_mask, ref_tensor):
    if video_frames is None:
        return ref_tensor.new_zeros((ref_tensor.shape[0], 3, ref_tensor.shape[-2], ref_tensor.shape[-1]))

    batch_size, seq_len, channels, height, width = video_frames.shape
    if channels != 3:
        raise ValueError(f"Expected RGB support frames, got {video_frames.shape}")

    device = ref_tensor.device
    dtype = ref_tensor.dtype
    weights = torch.ones((batch_size, seq_len), device=device, dtype=dtype)
    if video_frame_mask is not None:
        weights = video_frame_mask.to(device=device, dtype=dtype)

    positions = torch.arange(seq_len, device=device, dtype=dtype).view(1, seq_len)
    if video_anchor_indices is not None:
        anchor_start = video_anchor_indices.to(device=device, dtype=dtype)[:, 0:1]
    else:
        anchor_start = positions[:, seq_len // 2:seq_len // 2 + 1].expand(batch_size, 1)
    relative_positions = positions - anchor_start
    target_time = _reshape_time_step(time_step, batch_size, device, dtype).view(batch_size, 1)
    local_weights = torch.exp(-0.5 * ((relative_positions - target_time) / 1.0) ** 2)
    weights = weights * local_weights
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    rgb_context = torch.sum(
        video_frames.to(device=device, dtype=dtype) * weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        dim=1,
    )
    if rgb_context.shape[-2:] != ref_tensor.shape[-2:]:
        rgb_context = F.interpolate(rgb_context, size=ref_tensor.shape[-2:], mode="bilinear", align_corners=False)
    return rgb_context


class VideoContextSynthesisNetwork(nn.Module):
    def __init__(self, feat_channels):
        super(VideoContextSynthesisNetwork, self).__init__()
        self.num_downsamples = 2
        self.feat_channels = feat_channels

        self.conv_down1 = nn.Sequential(
            nn.Conv2d(6 + 1 + 3 + feat_channels, feat_channels, 7, padding=3),
            nn.PReLU(feat_channels),
            nn.Conv2d(feat_channels, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_down2 = nn.Sequential(
            nn.Conv2d(feat_channels * 5, feat_channels * 2, 2, stride=2, padding=0),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_down3 = nn.Sequential(
            nn.Conv2d(feat_channels * 8, feat_channels * 4, 2, stride=2, padding=0),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
        )
        self.conv_up1 = nn.Sequential(
            torch.nn.Conv2d(feat_channels * 16, feat_channels * 8, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_up2 = nn.Sequential(
            torch.nn.Conv2d(feat_channels * 6, feat_channels * 4, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(feat_channels),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.PReLU(feat_channels),
        )
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_out = nn.Conv2d(feat_channels * 2, 4, 3, padding=1)

    def get_warped_representations(self, bi_flow, c0, c1, i0=None, i1=None):
        flow_t0 = bi_flow[:, :2]
        flow_t1 = bi_flow[:, 2:4]
        warped_c0 = backwarp(c0, flow_t0)
        warped_c1 = backwarp(c1, flow_t1)
        if (i0 is None) and (i1 is None):
            return warped_c0, warped_c1
        warped_img0 = backwarp(i0, flow_t0)
        warped_img1 = backwarp(i1, flow_t1)
        return warped_img0, warped_img1, warped_c0, warped_c1

    def forward(
        self,
        i0,
        i1,
        c0_pyr,
        c1_pyr,
        bi_flow_pyr,
        occ,
        video_context_pyr=None,
        video_frames=None,
        video_frame_mask=None,
        video_anchor_indices=None,
        time_step=None,
    ):
        context_lvl0 = _context_level_or_zeros(video_context_pyr, 0, c0_pyr[0], self.feat_channels)
        context_lvl1 = _context_level_or_zeros(video_context_pyr, 1, c0_pyr[1], self.feat_channels * 2)
        context_lvl2 = _context_level_or_zeros(video_context_pyr, 2, c0_pyr[2], self.feat_channels * 4)
        rgb_context = _aggregate_rgb_context(video_frames, time_step, video_anchor_indices, video_frame_mask, c0_pyr[0])

        warped_img0, warped_img1, warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1
        )
        s0 = self.conv_down1(torch.cat((warped_img0, warped_img1, occ, rgb_context, context_lvl0), dim=1))
        s1 = self.conv_down2(torch.cat((s0, warped_c0, warped_c1, context_lvl0), dim=1))

        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[1], c0_pyr[1], c1_pyr[1], None, None
        )
        s2 = self.conv_down3(torch.cat((s1, warped_c0, warped_c1, context_lvl1), dim=1))

        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[2], c0_pyr[2], c1_pyr[2], None, None
        )
        x = self.conv_up1(torch.cat((s2, warped_c0, warped_c1, context_lvl2), dim=1))
        x = self.conv_up2(torch.cat((x, s1, context_lvl1), dim=1))
        x = self.conv_up3(torch.cat((x, s0, context_lvl0), dim=1))

        refine = self.conv_out(x)
        refine_res = refine[:, :3]
        occ_res = refine[:, 3:]
        occ_out = occ + occ_res
        blending_mask = torch.sigmoid(occ_out)
        merged_img = (warped_img0 * blending_mask + warped_img1 * (1 - blending_mask)) + refine_res

        extra_dict = {
            "refine_res": refine_res,
            "refine_mask": occ_out,
            "warped_img0": warped_img0,
            "warped_img1": warped_img1,
            "merged_img": merged_img,
            "video_rgb_context": rgb_context,
            "video_context_lvl0": context_lvl0,
            "video_context_lvl1": context_lvl1,
            "video_context_lvl2": context_lvl2,
        }
        return merged_img, occ_out, extra_dict


class VideoContextEAMambaSynthesisNetwork(nn.Module):
    def __init__(
        self,
        feat_channels,
        num_blocks=(2, 3, 3, 4),
        num_refinement_blocks=1,
        ffn_expansion_factor=2.0,
        bias=False,
        layernorm_type="WithBias",
        checkpoint_percentage=0.0,
        channel_mixer_type="Simple",
        mamba_cfg=None,
    ):
        super().__init__()
        if len(num_blocks) != 4:
            raise ValueError("VideoContextEAMambaSynthesisNetwork expects 4 stage depths.")

        self.num_downsamples = 3
        self.feat_channels = feat_channels
        self.mamba_cfg = EAMambaSynthesisNetwork._build_mamba_cfg(mamba_cfg)

        channel_mixer_type = channel_mixer_type.lower() if channel_mixer_type is not None else None
        if channel_mixer_type not in ALLOWED_CHANNEL_MIXER_TYPE:
            raise ValueError(
                f"channel_mixer_type should be one of {ALLOWED_CHANNEL_MIXER_TYPE}, "
                f"but got {channel_mixer_type}"
            )

        scan_type = self.mamba_cfg.get("scan_type")
        scan_type = scan_type.lower() if scan_type is not None else None
        scan_count = self.mamba_cfg.get("scan_count")
        scan_merge_method = self.mamba_cfg.get("scan_merge_method")
        scan_transform = ScanTransform(scan_type, scan_count, scan_merge_method)

        shared_settings = {
            "ffn_expansion_factor": ffn_expansion_factor,
            "bias": bias,
            "layernorm_type": layernorm_type,
            "scan_transform": scan_transform,
            "mamba_cfg": self.mamba_cfg,
            "checkpoint_percentage": checkpoint_percentage,
            "channel_mixer_type": channel_mixer_type,
        }

        self.patch_embed = EAMambaOverlapPatchEmbed(6 + 1 + 3 + feat_channels, feat_channels, bias=bias)
        self.encoder_level1 = create_eamamba_blocks(
            num_blocks=num_blocks[0], dim=feat_channels, **shared_settings
        )
        self.down1_2 = EAMambaDownsample(feat_channels * 4, feat_channels * 2, bias=bias)
        self.encoder_level2 = create_eamamba_blocks(
            num_blocks=num_blocks[1], dim=feat_channels * 2, **shared_settings
        )

        self.down2_3 = EAMambaDownsample(feat_channels * 8, feat_channels * 4, bias=bias)
        self.encoder_level3 = create_eamamba_blocks(
            num_blocks=num_blocks[2], dim=feat_channels * 4, **shared_settings
        )

        self.down3_4 = EAMambaDownsample(feat_channels * 16, feat_channels * 8, bias=bias)
        self.latent = create_eamamba_blocks(
            num_blocks=num_blocks[3], dim=feat_channels * 8, **shared_settings
        )

        self.up4_3 = EAMambaUpsample(feat_channels * 8, feat_channels * 4, bias=bias)
        self.reduce_chan_level3 = nn.Conv2d(feat_channels * 12, feat_channels * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = create_eamamba_blocks(
            num_blocks=num_blocks[2], dim=feat_channels * 4, **shared_settings
        )

        self.up3_2 = EAMambaUpsample(feat_channels * 4, feat_channels * 2, bias=bias)
        self.reduce_chan_level2 = nn.Conv2d(feat_channels * 6, feat_channels * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = create_eamamba_blocks(
            num_blocks=num_blocks[1], dim=feat_channels * 2, **shared_settings
        )

        self.up2_1 = EAMambaUpsample(feat_channels * 2, feat_channels, bias=bias)
        self.reduce_chan_level1 = nn.Conv2d(feat_channels * 3, feat_channels * 2, kernel_size=1, bias=bias)
        self.decoder_level1 = create_eamamba_blocks(
            num_blocks=num_blocks[0], dim=feat_channels * 2, **shared_settings
        )
        self.refinement = create_eamamba_blocks(
            num_blocks=num_refinement_blocks, dim=feat_channels * 2, **shared_settings
        )

        self.conv_out = nn.Conv2d(feat_channels * 2, 4, 3, padding=1)

    def get_warped_representations(self, bi_flow, c0, c1, i0=None, i1=None):
        flow_t0 = bi_flow[:, :2]
        flow_t1 = bi_flow[:, 2:4]
        warped_c0 = backwarp(c0, flow_t0)
        warped_c1 = backwarp(c1, flow_t1)
        if (i0 is None) and (i1 is None):
            return warped_c0, warped_c1
        warped_img0 = backwarp(i0, flow_t0)
        warped_img1 = backwarp(i1, flow_t1)
        return warped_img0, warped_img1, warped_c0, warped_c1

    def forward(
        self,
        i0,
        i1,
        c0_pyr,
        c1_pyr,
        bi_flow_pyr,
        occ,
        video_context_pyr=None,
        video_frames=None,
        video_frame_mask=None,
        video_anchor_indices=None,
        time_step=None,
    ):
        context_lvl0 = _context_level_or_zeros(video_context_pyr, 0, c0_pyr[0], self.feat_channels)
        context_lvl1 = _context_level_or_zeros(video_context_pyr, 1, c0_pyr[1], self.feat_channels * 2)
        context_lvl2 = _context_level_or_zeros(video_context_pyr, 2, c0_pyr[2], self.feat_channels * 4)
        rgb_context = _aggregate_rgb_context(video_frames, time_step, video_anchor_indices, video_frame_mask, c0_pyr[0])

        warped_img0, warped_img1, warped_c0_lvl0, warped_c1_lvl0 = self.get_warped_representations(
            bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1
        )
        x0 = self.patch_embed(torch.cat((warped_img0, warped_img1, occ, rgb_context, context_lvl0), dim=1))
        x0 = self.encoder_level1(x0)

        warped_c0_lvl1, warped_c1_lvl1 = self.get_warped_representations(
            bi_flow_pyr[1], c0_pyr[1], c1_pyr[1], None, None
        )
        x1 = self.down1_2(torch.cat((x0, warped_c0_lvl0, warped_c1_lvl0, context_lvl0), dim=1))
        x1 = self.encoder_level2(x1)

        x2 = self.down2_3(torch.cat((x1, warped_c0_lvl1, warped_c1_lvl1, context_lvl1), dim=1))
        x2 = self.encoder_level3(x2)

        warped_c0_lvl2, warped_c1_lvl2 = self.get_warped_representations(
            bi_flow_pyr[2], c0_pyr[2], c1_pyr[2], None, None
        )
        x3 = self.down3_4(torch.cat((x2, warped_c0_lvl2, warped_c1_lvl2, context_lvl2), dim=1))
        x3 = self.latent(x3)

        d2 = self.up4_3(x3)
        d2 = self.reduce_chan_level3(torch.cat((d2, x2, context_lvl2), dim=1))
        d2 = self.decoder_level3(d2)

        d1 = self.up3_2(d2)
        d1 = self.reduce_chan_level2(torch.cat((d1, x1, context_lvl1), dim=1))
        d1 = self.decoder_level2(d1)

        d0 = self.up2_1(d1)
        d0 = self.reduce_chan_level1(torch.cat((d0, x0, context_lvl0), dim=1))
        d0 = self.decoder_level1(d0)
        d0 = self.refinement(d0)

        refine = self.conv_out(d0)
        refine_res = refine[:, :3]
        occ_res = refine[:, 3:]
        occ_out = occ + occ_res
        blending_mask = torch.sigmoid(occ_out)
        merged_img = (warped_img0 * blending_mask + warped_img1 * (1 - blending_mask)) + refine_res

        extra_dict = {
            "refine_res": refine_res,
            "refine_mask": occ_out,
            "warped_img0": warped_img0,
            "warped_img1": warped_img1,
            "merged_img": merged_img,
            "video_rgb_context": rgb_context,
            "video_context_lvl0": context_lvl0,
            "video_context_lvl1": context_lvl1,
            "video_context_lvl2": context_lvl2,
        }
        return merged_img, occ_out, extra_dict
