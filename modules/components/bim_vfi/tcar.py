import torch
import torch.nn as nn
import torch.nn.functional as F

from .backwarp import backwarp


def _context_level_or_zeros(context_pyr, level, ref_tensor, channels):
    if context_pyr is None or level >= len(context_pyr):
        return ref_tensor.new_zeros((ref_tensor.shape[0], channels, ref_tensor.shape[-2], ref_tensor.shape[-1]))

    context_feat = context_pyr[level]
    if context_feat.shape[-2:] != ref_tensor.shape[-2:]:
        context_feat = F.interpolate(
            context_feat, size=ref_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
    return context_feat


class TokenBiasAdapter(nn.Module):
    def __init__(self, token_channels, out_channels, hidden_channels=None, init_zero=True):
        super().__init__()
        hidden_channels = hidden_channels or max(out_channels * 4, token_channels)
        self.norm = nn.LayerNorm(token_channels)
        self.mlp = nn.Sequential(
            nn.Linear(token_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )
        if init_zero:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens, ref_tensor):
        if tokens is None:
            return ref_tensor.new_zeros(ref_tensor.shape)
        if tokens.dim() == 3:
            tokens = tokens.mean(dim=1)
        tokens = self.norm(tokens.to(device=ref_tensor.device, dtype=ref_tensor.dtype))
        bias = self.mlp(tokens).view(ref_tensor.shape[0], ref_tensor.shape[1], 1, 1)
        return bias.expand_as(ref_tensor)


class TCAR(nn.Module):
    """
    Temporal-Context Anchored Reconstructor.

    This module keeps the original BiM-VFI U-Net reconstruction backbone
    unchanged and injects two extra context sources through residual adapters:
    1. detail_context_pyr from LTCE
    2. global_context_pyr from GTME

    Forward inputs are compatible with the original pair-centric UNet inputs
    plus the two optional context pyramids:
    - i0, i1, c0_pyr, c1_pyr, bi_flow_pyr, occ
    - detail_context_pyr
    - global_context_pyr
    """

    def __init__(self, feat_channels):
        super(TCAR, self).__init__()
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

        self.detail_to_s0 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.global_to_s0 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.detail_to_s1 = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.global_to_s1 = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.detail_to_s2 = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
        )
        self.global_to_s2 = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
        )
        self.detail_to_x1 = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.global_to_x1 = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.detail_to_x0 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.PReLU(feat_channels),
        )
        self.global_to_x0 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.PReLU(feat_channels),
        )

    def _global_context_residual(self, adapter, context_pyr, level, ref_tensor, channels):
        context_feat = _context_level_or_zeros(context_pyr, level, ref_tensor, channels)
        return adapter(context_feat)

    def _structure_token_bias(self, level_name, ref_tensor, global_structure_tokens):
        return ref_tensor.new_zeros(ref_tensor.shape)

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
        detail_context_pyr=None,
        global_context_pyr=None,
        global_structure_tokens=None,
        **kwargs,
    ):
        detail_lvl0 = _context_level_or_zeros(detail_context_pyr, 0, c0_pyr[0], self.feat_channels)
        detail_lvl1 = _context_level_or_zeros(detail_context_pyr, 1, c0_pyr[1], self.feat_channels * 2)
        detail_lvl2 = _context_level_or_zeros(detail_context_pyr, 2, c0_pyr[2], self.feat_channels * 4)

        global_lvl0 = _context_level_or_zeros(global_context_pyr, 0, c0_pyr[0], self.feat_channels)
        global_lvl1 = _context_level_or_zeros(global_context_pyr, 1, c0_pyr[1], self.feat_channels * 2)
        global_lvl2 = _context_level_or_zeros(global_context_pyr, 2, c0_pyr[2], self.feat_channels * 4)

        warped_img0, warped_img1, warped_c0_lvl0, warped_c1_lvl0 = self.get_warped_representations(
            bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1
        )
        input_feat = torch.cat((warped_img0, warped_img1, occ), dim=1)
        s0 = self.conv_down1(input_feat)
        s0 = (
            s0
            + self.detail_to_s0(detail_lvl0)
            + self._global_context_residual(
                self.global_to_s0, global_context_pyr, 0, s0, self.feat_channels
            )
            + self._structure_token_bias("s0", s0, global_structure_tokens)
        )

        s1 = self.conv_down2(torch.cat((s0, warped_c0_lvl0, warped_c1_lvl0), dim=1))
        warped_c0_lvl1, warped_c1_lvl1 = self.get_warped_representations(
            bi_flow_pyr[1], c0_pyr[1], c1_pyr[1], None, None
        )
        s1 = (
            s1
            + self.detail_to_s1(detail_lvl1)
            + self._global_context_residual(
                self.global_to_s1, global_context_pyr, 1, s1, self.feat_channels * 2
            )
            + self._structure_token_bias("s1", s1, global_structure_tokens)
        )

        s2 = self.conv_down3(torch.cat((s1, warped_c0_lvl1, warped_c1_lvl1), dim=1))
        warped_c0_lvl2, warped_c1_lvl2 = self.get_warped_representations(
            bi_flow_pyr[2], c0_pyr[2], c1_pyr[2], None, None
        )
        s2 = (
            s2
            + self.detail_to_s2(detail_lvl2)
            + self._global_context_residual(
                self.global_to_s2, global_context_pyr, 2, s2, self.feat_channels * 4
            )
            + self._structure_token_bias("s2", s2, global_structure_tokens)
        )

        x = self.conv_up1(torch.cat((s2, warped_c0_lvl2, warped_c1_lvl2), dim=1))
        x = (
            x
            + self.detail_to_x1(detail_lvl1)
            + self._global_context_residual(
                self.global_to_x1, global_context_pyr, 1, x, self.feat_channels * 2
            )
            + self._structure_token_bias("x1", x, global_structure_tokens)
        )
        x = self.conv_up2(torch.cat((x, s1), dim=1))
        x = (
            x
            + self.detail_to_x0(detail_lvl0)
            + self._global_context_residual(
                self.global_to_x0, global_context_pyr, 0, x, self.feat_channels
            )
            + self._structure_token_bias("x0", x, global_structure_tokens)
        )
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
            "detail_context_lvl0": detail_lvl0,
            "detail_context_lvl1": detail_lvl1,
            "detail_context_lvl2": detail_lvl2,
            "global_context_lvl0": global_lvl0,
            "global_context_lvl1": global_lvl1,
            "global_context_lvl2": global_lvl2,
        }
        return merged_img, occ_out, extra_dict


class PairGlobalMemoryTCAR(TCAR):
    """
    TCAR variant that directly consumes PairConditionedGlobalMemory
    structure_condition.

    The original TCAR reconstruction path and context pyramid adapters retain
    their parameter names. New token-to-bias adapters are separate and
    zero-initialized for backward-compatible checkpoint loading.
    """

    accepts_global_structure_tokens = True

    def __init__(
        self,
        feat_channels,
        global_token_channels=256,
        token_hidden_channels=None,
        token_init_zero=True,
    ):
        super().__init__(feat_channels=feat_channels)
        self.structure_token_to_s0 = TokenBiasAdapter(
            global_token_channels,
            feat_channels * 2,
            hidden_channels=token_hidden_channels,
            init_zero=token_init_zero,
        )
        self.structure_token_to_s1 = TokenBiasAdapter(
            global_token_channels,
            feat_channels * 2,
            hidden_channels=token_hidden_channels,
            init_zero=token_init_zero,
        )
        self.structure_token_to_s2 = TokenBiasAdapter(
            global_token_channels,
            feat_channels * 4,
            hidden_channels=token_hidden_channels,
            init_zero=token_init_zero,
        )
        self.structure_token_to_x1 = TokenBiasAdapter(
            global_token_channels,
            feat_channels * 2,
            hidden_channels=token_hidden_channels,
            init_zero=token_init_zero,
        )
        self.structure_token_to_x0 = TokenBiasAdapter(
            global_token_channels,
            feat_channels,
            hidden_channels=token_hidden_channels,
            init_zero=token_init_zero,
        )

    def _global_context_residual(self, adapter, context_pyr, level, ref_tensor, channels):
        if context_pyr is None or level >= len(context_pyr):
            return ref_tensor.new_zeros(ref_tensor.shape)
        return super()._global_context_residual(adapter, context_pyr, level, ref_tensor, channels)

    def _structure_token_bias(self, level_name, ref_tensor, global_structure_tokens):
        adapters = {
            "s0": self.structure_token_to_s0,
            "s1": self.structure_token_to_s1,
            "s2": self.structure_token_to_s2,
            "x1": self.structure_token_to_x1,
            "x0": self.structure_token_to_x0,
        }
        return adapters[level_name](global_structure_tokens, ref_tensor)


TemporalContextAnchoredReconstructor = TCAR
PairGlobalMemoryReconstructor = PairGlobalMemoryTCAR
