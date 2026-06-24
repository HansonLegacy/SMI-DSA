import torch
import torch.nn as nn
import torch.nn.functional as F

from .backwarp import backwarp


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

    batch_size, seq_len, channels, _, _ = video_frames.shape
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
        anchor_start = positions[:, seq_len // 2 : seq_len // 2 + 1].expand(batch_size, 1)
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


class VideoContextUNetSynthesisNetwork(nn.Module):
    """
    Lighter sequence-aware SN that keeps the original SynthesisNetwork U-Net layout.
    Video context is injected through small residual adapters instead of wide concatenation,
    which keeps activation sizes closer to the pair-centric baseline.
    """

    def __init__(self, feat_channels):
        super(VideoContextUNetSynthesisNetwork, self).__init__()
        self.num_downsamples = 2
        self.feat_channels = feat_channels

        self.conv_down1 = nn.Sequential(
            nn.Conv2d(6 + 1, feat_channels, 7, padding=3),
            nn.PReLU(feat_channels),
            nn.Conv2d(feat_channels, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_down2 = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 2, 2, stride=2, padding=0),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_down3 = nn.Sequential(
            nn.Conv2d(feat_channels * 6, feat_channels * 4, 2, stride=2, padding=0),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
        )
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(feat_channels * 12, feat_channels * 8, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(feat_channels),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.PReLU(feat_channels),
        )
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(feat_channels * 3, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_out = nn.Conv2d(feat_channels * 2, 4, 3, padding=1)

        self.rgb_context_adapter = nn.Sequential(
            nn.Conv2d(3, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.context_to_s0 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.context_to_s1 = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.context_to_s2 = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
        )
        self.context_to_x1 = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.context_to_x0 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.PReLU(feat_channels),
        )

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
        rgb_context = _aggregate_rgb_context(
            video_frames, time_step, video_anchor_indices, video_frame_mask, c0_pyr[0]
        )

        warped_img0, warped_img1, warped_c0_lvl0, warped_c1_lvl0 = self.get_warped_representations(
            bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1
        )
        input_feat = torch.cat((warped_img0, warped_img1, occ), dim=1)
        s0 = self.conv_down1(input_feat)
        s0 = s0 + self.rgb_context_adapter(rgb_context) + self.context_to_s0(context_lvl0)

        s1 = self.conv_down2(torch.cat((s0, warped_c0_lvl0, warped_c1_lvl0), dim=1))
        warped_c0_lvl1, warped_c1_lvl1 = self.get_warped_representations(
            bi_flow_pyr[1], c0_pyr[1], c1_pyr[1], None, None
        )
        s1 = s1 + self.context_to_s1(context_lvl1)

        s2 = self.conv_down3(torch.cat((s1, warped_c0_lvl1, warped_c1_lvl1), dim=1))
        warped_c0_lvl2, warped_c1_lvl2 = self.get_warped_representations(
            bi_flow_pyr[2], c0_pyr[2], c1_pyr[2], None, None
        )
        s2 = s2 + self.context_to_s2(context_lvl2)

        x = self.conv_up1(torch.cat((s2, warped_c0_lvl2, warped_c1_lvl2), dim=1))
        x = x + self.context_to_x1(context_lvl1)
        x = self.conv_up2(torch.cat((x, s1), dim=1))
        x = x + self.context_to_x0(context_lvl0)
        x = self.conv_up3(torch.cat((x, s0), dim=1))

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
