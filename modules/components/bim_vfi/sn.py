import math

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
class SynthesisNetwork(nn.Module):
    def __init__(self, feat_channels):
        super(SynthesisNetwork, self).__init__()
        self.num_downsamples = 2
        input_channels = 6 + 1
        self.conv_down1 = nn.Sequential(
            nn.Conv2d(input_channels, feat_channels, 7, padding=3),
            nn.PReLU(feat_channels),
            nn.Conv2d(feat_channels, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2))
        self.conv_down2 = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 2, 2, stride=2, padding=0),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2))
        self.conv_down3 = nn.Sequential(
            nn.Conv2d(feat_channels * 6, feat_channels * 4, 2, stride=2, padding=0),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4))
        self.conv_up1 = nn.Sequential(
            torch.nn.Conv2d(feat_channels * 12, feat_channels * 8, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2))
        self.conv_up2 = nn.Sequential(
            torch.nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(feat_channels * 1),
            nn.Conv2d(feat_channels * 1, feat_channels * 1, 3, padding=1),
            nn.PReLU(feat_channels * 1))
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(feat_channels * 3, feat_channels * 2, 3, padding=1),
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

    def forward(self, i0, i1, c0_pyr, c1_pyr, bi_flow_pyr, occ):
        warped_img0, warped_img1, warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1
        )
        input_feat = torch.cat((warped_img0, warped_img1, occ), 1)
        s0 = self.conv_down1(input_feat)
        s1 = self.conv_down2(torch.cat((s0, warped_c0, warped_c1), 1))
        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[1], c0_pyr[1], c1_pyr[1], None, None
        )
        s2 = self.conv_down3(torch.cat((s1, warped_c0, warped_c1), 1))
        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[2], c0_pyr[2], c1_pyr[2], None, None
        )

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
        return merged_img, occ_out, extra_dict


class DeepDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DeepDownBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.PReLU(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DeepUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepUpBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.PReLU(out_channels),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv(x)
        return x


class DeepSynthesisNetwork(nn.Module):
    def __init__(self, feat_channels):
        super(DeepSynthesisNetwork, self).__init__()
        self.num_downsamples = 3
        input_channels = 6 + 1
        self.conv_down1 = nn.Sequential(
            nn.Conv2d(input_channels, feat_channels, 7, padding=3),
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
        self.conv_down4 = DeepDownBlock(feat_channels * 12, feat_channels * 8, stride=2)
        self.supple = DeepDownBlock(feat_channels * 4, feat_channels * 8, stride=2)

        self.deep_up0 = DeepUpBlock(feat_channels * 24, feat_channels * 4)
        self.deep_up1 = DeepUpBlock(feat_channels * 8, feat_channels * 2)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(feat_channels * 1),
            nn.Conv2d(feat_channels * 1, feat_channels * 1, 3, padding=1),
            nn.PReLU(feat_channels * 1),
        )
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(feat_channels * 3, feat_channels * 2, 3, padding=1),
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

    def forward(self, i0, i1, c0_pyr, c1_pyr, bi_flow_pyr, occ):
        warped_img0, warped_img1, warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1
        )
        input_feat = torch.cat((warped_img0, warped_img1, occ), 1)
        s0 = self.conv_down1(input_feat)
        s1 = self.conv_down2(torch.cat((s0, warped_c0, warped_c1), 1))

        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[1], c0_pyr[1], c1_pyr[1], None, None
        )
        s2 = self.conv_down3(torch.cat((s1, warped_c0, warped_c1), 1))

        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[2], c0_pyr[2], c1_pyr[2], None, None
        )
        s3 = self.conv_down4(torch.cat((s2, warped_c0, warped_c1), 1))
        c0_deep = self.supple(warped_c0)
        c1_deep = self.supple(warped_c1)

        x = self.deep_up0(torch.cat((s3, c0_deep, c1_deep), 1))
        x = self.deep_up1(torch.cat((x, s2), 1))
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
        return merged_img, occ_out, extra_dict


class EAMambaSynthesisNetwork(nn.Module):
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
            raise ValueError("EAMambaSynthesisNetwork expects 4 stage depths.")

        self.num_downsamples = 3
        self.feat_channels = feat_channels
        self.mamba_cfg = self._build_mamba_cfg(mamba_cfg)

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

        input_channels = 6 + 1
        self.patch_embed = EAMambaOverlapPatchEmbed(input_channels, feat_channels, bias=bias)

        self.encoder_level1 = create_eamamba_blocks(
            num_blocks=num_blocks[0], dim=feat_channels, **shared_settings
        )
        self.down1_2 = EAMambaDownsample(feat_channels * 3, feat_channels * 2, bias=bias)
        self.encoder_level2 = create_eamamba_blocks(
            num_blocks=num_blocks[1], dim=feat_channels * 2, **shared_settings
        )

        self.down2_3 = EAMambaDownsample(feat_channels * 6, feat_channels * 4, bias=bias)
        self.encoder_level3 = create_eamamba_blocks(
            num_blocks=num_blocks[2], dim=feat_channels * 4, **shared_settings
        )

        self.down3_4 = EAMambaDownsample(feat_channels * 12, feat_channels * 8, bias=bias)
        self.latent = create_eamamba_blocks(
            num_blocks=num_blocks[3], dim=feat_channels * 8, **shared_settings
        )

        self.up4_3 = EAMambaUpsample(feat_channels * 8, feat_channels * 4, bias=bias)
        self.reduce_chan_level3 = nn.Conv2d(feat_channels * 8, feat_channels * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = create_eamamba_blocks(
            num_blocks=num_blocks[2], dim=feat_channels * 4, **shared_settings
        )

        self.up3_2 = EAMambaUpsample(feat_channels * 4, feat_channels * 2, bias=bias)
        self.reduce_chan_level2 = nn.Conv2d(feat_channels * 4, feat_channels * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = create_eamamba_blocks(
            num_blocks=num_blocks[1], dim=feat_channels * 2, **shared_settings
        )

        self.up2_1 = EAMambaUpsample(feat_channels * 2, feat_channels, bias=bias)
        self.decoder_level1 = create_eamamba_blocks(
            num_blocks=num_blocks[0], dim=feat_channels * 2, **shared_settings
        )
        self.refinement = create_eamamba_blocks(
            num_blocks=num_refinement_blocks, dim=feat_channels * 2, **shared_settings
        )

        self.conv_out = nn.Conv2d(feat_channels * 2, 4, 3, padding=1)

    @staticmethod
    def _build_mamba_cfg(mamba_cfg):
        cfg = {
            "scan_type": "diagonal",
            "scan_count": 4,
            "scan_merge_method": "concate",
            "disable_z_branch": False,
            "d_state": 16,
            "d_conv": 3,
            "expand": 1,
            "conv_2d": True,
        }
        if mamba_cfg is not None:
            cfg.update(mamba_cfg)
        return cfg

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

    def forward(self, i0, i1, c0_pyr, c1_pyr, bi_flow_pyr, occ):
        warped_img0, warped_img1, warped_c0_lvl0, warped_c1_lvl0 = self.get_warped_representations(
            bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1
        )
        x0 = self.patch_embed(torch.cat((warped_img0, warped_img1, occ), dim=1))
        x0 = self.encoder_level1(x0)

        warped_c0_lvl1, warped_c1_lvl1 = self.get_warped_representations(
            bi_flow_pyr[1], c0_pyr[1], c1_pyr[1], None, None
        )
        x1 = self.down1_2(torch.cat((x0, warped_c0_lvl0, warped_c1_lvl0), dim=1))
        x1 = self.encoder_level2(x1)

        x2 = self.down2_3(torch.cat((x1, warped_c0_lvl1, warped_c1_lvl1), dim=1))
        x2 = self.encoder_level3(x2)

        warped_c0_lvl2, warped_c1_lvl2 = self.get_warped_representations(
            bi_flow_pyr[2], c0_pyr[2], c1_pyr[2], None, None
        )
        x3 = self.down3_4(torch.cat((x2, warped_c0_lvl2, warped_c1_lvl2), dim=1))
        x3 = self.latent(x3)

        d2 = self.up4_3(x3)
        d2 = self.reduce_chan_level3(torch.cat((d2, x2), dim=1))
        d2 = self.decoder_level3(d2)

        d1 = self.up3_2(d2)
        d1 = self.reduce_chan_level2(torch.cat((d1, x1), dim=1))
        d1 = self.decoder_level2(d1)

        d0 = self.up2_1(d1)
        d0 = self.decoder_level1(torch.cat((d0, x0), dim=1))
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
        }
        return merged_img, occ_out, extra_dict


class TransUNetSynthesisNetwork(nn.Module):
    def __init__(
        self,
        feat_channels,
        img_size=256,
        use_transunet=True,
        vit_config="R50-ViT-B_16",
        adapter_hidden_size=None,
        adapter_num_heads=None,
        adapter_mlp_dim=None,
        adapter_num_layers=2,
        adapter_pool_size=8,
        adapter_init_scale=1e-3,
        adapter_init_mode="identity",
    ):
        super(TransUNetSynthesisNetwork, self).__init__()
        self.num_downsamples = 2
        self.feat_channels = feat_channels
        self.img_size = img_size
        self.use_transunet = use_transunet
        self.vit_config = vit_config

        input_channels = 6 + 1
        self.conv_down1 = nn.Sequential(
            nn.Conv2d(input_channels, feat_channels, 7, padding=3),
            nn.PReLU(feat_channels),
            nn.Conv2d(feat_channels, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2))
        self.conv_down2 = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 2, 2, stride=2, padding=0),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2))
        self.conv_down3 = nn.Sequential(
            nn.Conv2d(feat_channels * 6, feat_channels * 4, 2, stride=2, padding=0),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4))
        self.conv_up1 = nn.Sequential(
            torch.nn.Conv2d(feat_channels * 12, feat_channels * 8, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2))
        self.conv_up2 = nn.Sequential(
            torch.nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(feat_channels * 1),
            nn.Conv2d(feat_channels * 1, feat_channels * 1, 3, padding=1),
            nn.PReLU(feat_channels * 1))
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(feat_channels * 3, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_out = nn.Conv2d(feat_channels * 2, 4, 3, padding=1)

        bottleneck_channels = feat_channels * 12
        hidden_size = adapter_hidden_size or min(256, bottleneck_channels)
        num_heads = adapter_num_heads or self._pick_num_heads(hidden_size)
        self.bottleneck_adapter = BottleneckTransformerAdapter(
            in_channels=bottleneck_channels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_dim=adapter_mlp_dim or min(1024, hidden_size * 4),
            num_layers=adapter_num_layers,
            pool_size=adapter_pool_size,
            init_scale=adapter_init_scale,
            init_mode=adapter_init_mode,
            enabled=use_transunet,
        )

    @staticmethod
    def _pick_num_heads(hidden_size):
        for num_heads in (8, 4, 2, 1):
            if hidden_size % num_heads == 0:
                return num_heads
        return 1

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

    def forward(self, i0, i1, c0_pyr, c1_pyr, bi_flow_pyr, occ):
        warped_img0, warped_img1, warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1
        )
        input_feat = torch.cat((warped_img0, warped_img1, occ), 1)
        s0 = self.conv_down1(input_feat)
        s1 = self.conv_down2(torch.cat((s0, warped_c0, warped_c1), 1))
        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[1], c0_pyr[1], c1_pyr[1], None, None
        )
        s2 = self.conv_down3(torch.cat((s1, warped_c0, warped_c1), 1))
        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[2], c0_pyr[2], c1_pyr[2], None, None
        )

        bottleneck_feat = torch.cat((s2, warped_c0, warped_c1), 1)
        bottleneck_feat = self.bottleneck_adapter(bottleneck_feat)

        x = self.conv_up1(bottleneck_feat)
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
        return merged_img, occ_out, extra_dict

    def load_pretrained_vit(self, pretrained_path):
        print(
            f"Conservative TransUNet graft keeps the original SN backbone; "
            f"external ViT weights are skipped: {pretrained_path}"
        )


class BottleneckTransformerAdapter(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        num_heads=8,
        mlp_dim=1024,
        num_layers=2,
        pool_size=8,
        init_scale=1e-3,
        init_mode="identity",
        enabled=True,
    ):
        super(BottleneckTransformerAdapter, self).__init__()
        self.enabled = enabled
        self.pool_size = pool_size

        self.in_proj = nn.Conv2d(in_channels, hidden_size, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, pool_size * pool_size, hidden_size))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads=num_heads, mlp_dim=mlp_dim)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Conv2d(hidden_size, in_channels, 1)
        if init_mode == "identity":
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            self.res_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.res_scale = nn.Parameter(torch.tensor([init_scale], dtype=torch.float32))

    def forward(self, x):
        if not self.enabled:
            return x

        pooled = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        pooled = self.in_proj(pooled)

        b, c, h, w = pooled.shape
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]

        for block in self.transformer_blocks:
            tokens = block(tokens)

        tokens = self.layer_norm(tokens)
        pooled = tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        pooled = self.out_proj(pooled)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.res_scale * pooled


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.dropout(attention_output)
        return attention_output


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=12, mlp_dim=3072, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.attn = Attention(hidden_size, num_heads)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class DecoderBlockTrans(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super(DecoderBlockTrans, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        if skip_channels > 0:
            self.skip_conv = nn.Conv2d(skip_channels, skip_channels, 1)
        else:
            self.skip_conv = None

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None and self.skip_conv is not None:
            skip = self.skip_conv(skip)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
