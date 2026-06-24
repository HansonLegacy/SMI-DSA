import torch
import torch.nn as nn
import torch.nn.functional as F

from .sn import SynthesisNetwork


class ASD(SynthesisNetwork):
    """
    Anatomical Structure Decoder.

    ASD keeps the original SynthesisNetwork backbone unchanged and only adds
    three zero-initialized anatomical context calibration adapters. The ADAMS
    anatomical context calibrates the warped anchor CFE features at each pyramid
    level:

        warped_c0_l <- warped_c0_l + A_l(C_anatomical,t_l)
        warped_c1_l <- warped_c1_l + A_l(C_anatomical,t_l)

    With zero-initialized adapter output convolutions, ASD starts exactly as the
    original SynthesisNetwork.
    """

    def __init__(self, feat_channels: int):
        super().__init__(feat_channels)
        self.anat_ctx0 = self._make_anatomical_adapter(feat_channels)
        self.anat_ctx1 = self._make_anatomical_adapter(feat_channels * 2)
        self.anat_ctx2 = self._make_anatomical_adapter(feat_channels * 4)

    @staticmethod
    def _make_anatomical_adapter(channels: int):
        adapter = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    @staticmethod
    def _select_context_level(anatomical_context_pyr, level: int, tau_index: int):
        if anatomical_context_pyr is None:
            return None
        if level >= len(anatomical_context_pyr):
            raise ValueError(
                f"ASD expected anatomical_context_pyr to contain level {level}, "
                f"but got {len(anatomical_context_pyr)} levels."
            )
        context = anatomical_context_pyr[level]
        if context.dim() == 5:
            return context[:, int(tau_index)]
        if context.dim() == 4:
            return context
        raise ValueError(
            f"ASD anatomical context level {level} must be [B, C, H, W] "
            f"or [B, S, C, H, W], got {tuple(context.shape)}."
        )

    def _calibrate_warped_features(
        self,
        warped_c0,
        warped_c1,
        anatomical_context_pyr,
        level: int,
        adapter,
        tau_index: int,
    ):
        context = self._select_context_level(anatomical_context_pyr, level, tau_index)
        if context is None:
            return warped_c0, warped_c1, None
        context = context.to(device=warped_c0.device, dtype=warped_c0.dtype)
        if context.shape[-2:] != warped_c0.shape[-2:]:
            context = F.interpolate(
                context,
                size=warped_c0.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        residual = adapter(context)
        return warped_c0 + residual, warped_c1 + residual, residual

    def forward(
        self,
        i0,
        i1,
        c0_pyr,
        c1_pyr,
        bi_flow_pyr,
        occ,
        anatomical_context_pyr=None,
        tau_index: int = 0,
        return_context_residuals: bool = False,
        **kwargs,
    ):
        if anatomical_context_pyr is None:
            anatomical_context_pyr = kwargs.get("anat_context_pyr", None)
        if anatomical_context_pyr is None:
            anatomical_context_pyr = kwargs.get("structure_context_pyr", None)

        context_residuals = []

        warped_img0, warped_img1, warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1
        )
        warped_c0, warped_c1, ctx0 = self._calibrate_warped_features(
            warped_c0,
            warped_c1,
            anatomical_context_pyr,
            level=0,
            adapter=self.anat_ctx0,
            tau_index=tau_index,
        )
        context_residuals.append(ctx0)

        input_feat = torch.cat((warped_img0, warped_img1, occ), 1)
        s0 = self.conv_down1(input_feat)
        s1 = self.conv_down2(torch.cat((s0, warped_c0, warped_c1), 1))

        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[1], c0_pyr[1], c1_pyr[1], None, None
        )
        warped_c0, warped_c1, ctx1 = self._calibrate_warped_features(
            warped_c0,
            warped_c1,
            anatomical_context_pyr,
            level=1,
            adapter=self.anat_ctx1,
            tau_index=tau_index,
        )
        context_residuals.append(ctx1)
        s2 = self.conv_down3(torch.cat((s1, warped_c0, warped_c1), 1))

        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[2], c0_pyr[2], c1_pyr[2], None, None
        )
        warped_c0, warped_c1, ctx2 = self._calibrate_warped_features(
            warped_c0,
            warped_c1,
            anatomical_context_pyr,
            level=2,
            adapter=self.anat_ctx2,
            tau_index=tau_index,
        )
        context_residuals.append(ctx2)

        x = self.conv_up1(torch.cat((s2, warped_c0, warped_c1), 1))
        x = self.conv_up2(torch.cat((x, s1), 1))
        x = self.conv_up3(torch.cat((x, s0), 1))

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
        }
        if return_context_residuals:
            extra_dict["anatomical_context_residuals"] = context_residuals

        return merged_img, occ_out, extra_dict


AnatomicalStructureDecoder = ASD
