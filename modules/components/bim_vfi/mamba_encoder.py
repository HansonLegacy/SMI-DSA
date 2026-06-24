import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    def to_2tuple(value):
        return value if isinstance(value, tuple) else (value, value)

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


class ChannelAttention(nn.Module):
    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        hidden_dim = max(num_feat // squeeze_factor, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, hidden_dim, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class CAB(nn.Module):
    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30):
        super().__init__()
        hidden_dim = max(num_feat // compress_ratio, 1)
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor=squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


class ConvStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, depth: int = 2, act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for layer_idx in range(depth):
            layers.append(nn.Conv2d(in_dim if layer_idx == 0 else out_dim, out_dim, 3, 1, 1))
            layers.append(act_layer(out_dim))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=3, stride=2, in_chans=3, embed_dim=64):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(in_chans)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        return self.proj(x)


class SS2D(nn.Module):
    """Single-image 2D selective scan adapted from VFIMamba's extractor.

    The original VFIMamba implementation interleaves two frames along the batch
    dimension for inter-frame modeling. Here we keep BiM-VFI's current encoder
    interface unchanged, so the scan is reduced to a single-image 2D version
    while preserving the same four-direction scan pattern.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank="auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError(
                "mamba_ssm is required to use MambaPyramid. "
                "Please install mamba_ssm before switching BiM-VFI to this encoder."
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(
            torch.stack(
                [nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False).weight for _ in range(4)],
                dim=0,
            )
        )
        dt_projs = [
            self._dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale=dt_scale,
                dt_init=dt_init,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
            )
            for _ in range(4)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([proj.weight for proj in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([proj.bias for proj in dt_projs], dim=0))

        self.A_logs = self._a_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._d_init(self.d_inner, copies=4, merge=True)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def _dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _a_log_init(d_state: int, d_inner: int, copies: int = 1, merge: bool = True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _d_init(d_inner: int, copies: int = 1, merge: bool = True):
        D = torch.ones(d_inner)
        if copies > 1:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x):
        B, _, H, W = x.shape
        L = H * W
        K = 4

        x_h = x.view(B, -1, L)
        x_w = x.transpose(2, 3).contiguous().view(B, -1, L)
        xs = torch.stack([x_h, x_w], dim=1)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = out_y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
        invwh_y = inv_y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = y.transpose(1, 2).contiguous().view(B, H, W, self.d_inner)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            expand=mlp_ratio,
            dropout=attn_drop_rate,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        shortcut = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.ln_1(shortcut)
        x = shortcut * self.skip_scale + self.drop_path(self.self_attention(x_norm))

        conv_inp = self.ln_2(x).permute(0, 3, 1, 2).contiguous()
        conv_out = self.conv_blk(conv_inp).permute(0, 2, 3, 1).contiguous()
        x = x * self.skip_scale2 + self.drop_path(conv_out)
        return x.permute(0, 3, 1, 2).contiguous()


class MambaStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path_rates: Sequence[float] | None = None,
        d_state: int = 16,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path_rates[idx],
                    norm_layer=nn.LayerNorm,
                    d_state=d_state,
                    mlp_ratio=mlp_ratio,
                )
                for idx in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class MambaPyramid(nn.Module):
    """BiM-VFI-compatible pyramid encoder inspired by VFIMamba.

    Input:
        img: [B, 3, H, W]
    Output:
        [C0, C1, C2] where
        - C0: [B, feat_channels, H, W]
        - C1: [B, feat_channels * 2, H/2, W/2]
        - C2: [B, feat_channels * 4, H/4, W/4]
    """

    def __init__(
        self,
        feat_channels: int,
        depths: Sequence[int] = (2, 2, 2),
        conv_stages: int = 1,
        d_state: int = 16,
        mlp_ratio: float = 2.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        if len(depths) != 3:
            raise ValueError("MambaPyramid expects exactly 3 stages to match BiM-VFI's pyramid interface.")

        embed_dims = [feat_channels, feat_channels * 2, feat_channels * 4]
        self.num_stages = len(embed_dims)
        self.conv_stages = conv_stages

        total_mamba_blocks = sum(depths[self.conv_stages :])
        dpr = torch.linspace(0, drop_path_rate, total_mamba_blocks).tolist() if total_mamba_blocks > 0 else []
        dpr_offset = 0

        for stage_idx in range(self.num_stages):
            if stage_idx == 0:
                block = ConvStage(3, embed_dims[stage_idx], depth=depths[stage_idx])
            else:
                if stage_idx < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[stage_idx - 1], embed_dims[stage_idx], 3, 2, 1),
                        nn.PReLU(embed_dims[stage_idx]),
                    )
                    block = ConvStage(embed_dims[stage_idx], embed_dims[stage_idx], depth=depths[stage_idx])
                else:
                    patch_embed = OverlapPatchEmbed(
                        patch_size=3,
                        stride=2,
                        in_chans=embed_dims[stage_idx - 1],
                        embed_dim=embed_dims[stage_idx],
                    )
                    stage_dpr = dpr[dpr_offset : dpr_offset + depths[stage_idx]]
                    dpr_offset += depths[stage_idx]
                    block = MambaStage(
                        embed_dims[stage_idx],
                        depths[stage_idx],
                        drop_path_rates=stage_dpr,
                        d_state=d_state,
                        mlp_ratio=mlp_ratio,
                    )
                setattr(self, f"patch_embed{stage_idx}", patch_embed)

            setattr(self, f"block{stage_idx}", block)

        self.conv_last = nn.Conv2d(embed_dims[-1], embed_dims[-1], 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img):
        x = img
        features = []

        for stage_idx in range(self.num_stages):
            if stage_idx > 0:
                patch_embed = getattr(self, f"patch_embed{stage_idx}")
                x = patch_embed(x)
            block = getattr(self, f"block{stage_idx}")
            x = block(x)
            features.append(x)

        features[-1] = self.conv_last(features[-1])
        return features
