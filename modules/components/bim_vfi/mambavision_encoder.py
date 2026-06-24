import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import DropPath, trunc_normal_
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

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


def window_partition(x: torch.Tensor, window_size: int):
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int):
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.reshape(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, windows.shape[2], height, width)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Stem(nn.Module):
    """MambaVision-style conv stem adapted to keep BiM-VFI's full-resolution first stage."""

    def __init__(self, in_chans: int = 3, hidden_dim: int = 16, out_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim, eps=1e-4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim: int, keep_dim: bool = False):
        super().__init__()
        dim_out = dim if keep_dim else dim * 2
        self.reduction = nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False)

    def forward(self, x):
        return self.reduction(x)


class ConvBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale: Optional[float] = None, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate="tanh")
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        if self.layer_scale:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        return shortcut + self.drop_path(x)


class MambaVisionMixer(nn.Module):
    """Sequence mixer transplanted from MambaVision's dense-prediction backbone."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError(
                "mamba_ssm is required to use MambaVisionPyramid. "
                "Please install mamba_ssm before switching BiM-VFI to this encoder."
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        if self.d_inner % 2 != 0:
            raise ValueError("MambaVisionMixer expects an even inner dimension.")
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(torch.rand(self.d_inner // 2) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner // 2, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            padding=d_conv // 2,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            padding=d_conv // 2,
        )

    def forward(self, hidden_states: torch.Tensor):
        batch, seq_len, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2).contiguous()
        x, z = xz.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())
        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))

        x_dbl = self.x_proj(x.transpose(1, 2).reshape(batch * seq_len, -1))
        dt, B_param, C_param = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt).view(batch, seq_len, -1).transpose(1, 2).contiguous()
        B_param = B_param.view(batch, seq_len, self.d_state).transpose(1, 2).contiguous()
        C_param = C_param.view(batch, seq_len, self.d_state).transpose(1, 2).contiguous()

        y = selective_scan_fn(
            x,
            dt,
            A,
            B_param,
            C_param,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )
        y = torch.cat([y, z], dim=1).transpose(1, 2).contiguous()
        return self.out_proj(y)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = attn @ v

        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HybridBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_index: int,
        transformer_blocks: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        layer_scale: Optional[float] = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if block_index in transformer_blocks:
            self.mixer = Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=8,
                d_conv=3,
                expand=1,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1.0
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1.0

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionStage(nn.Module):
    """Stage wrapper matching the dense-prediction MambaVision forward style."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        conv: bool = False,
        downsample: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Sequence[float] | float = 0.0,
        layer_scale: Optional[float] = None,
        layer_scale_conv: Optional[float] = None,
        transformer_blocks: Sequence[int] = (),
    ):
        super().__init__()
        self.conv = conv
        self.transformer_block = not conv
        if conv:
            self.blocks = nn.ModuleList(
                [
                    ConvBlock(
                        dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        layer_scale=layer_scale_conv,
                    )
                    for i in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    HybridBlock(
                        dim=dim,
                        block_index=i,
                        transformer_blocks=transformer_blocks,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        layer_scale=layer_scale,
                    )
                    for i in range(depth)
                ]
            )

        self.downsample = Downsample(dim=dim) if downsample else None
        self.window_size = window_size

    def forward(self, x):
        _, _, h, w = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, pad_r, 0, pad_b))
                _, _, hp, wp = x.shape
            else:
                hp, wp = h, w
            x = window_partition(x, self.window_size)

        for blk in self.blocks:
            x = blk(x)

        if self.transformer_block:
            x = window_reverse(x, self.window_size, hp, wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :h, :w].contiguous()

        out = x
        if self.downsample is None:
            return x, out
        return self.downsample(x), out


class MambaVisionPyramid(nn.Module):
    """MambaVision-style 3-level pyramid encoder adapted for BiM-VFI.

    The original MambaVision dense-prediction backbones expose multi-scale
    outputs before each stage's downsample. Here we keep that behavior, but
    compress the hierarchy to three levels so the output contract matches
    BiM-VFI's existing pyramid encoder:
    [B, C, H, W], [B, 2C, H/2, W/2], [B, 4C, H/4, W/4].
    """

    def __init__(
        self,
        feat_channels: int,
        depths: Sequence[int] = (2, 2, 4),
        num_heads: Sequence[int] = (1, 2, 4),
        window_size: Sequence[int] = (8, 8, 7),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        layer_scale: Optional[float] = None,
        layer_scale_conv: Optional[float] = None,
        stem_hidden_dim: Optional[int] = None,
        output_norm: str = "bn",
        last_conv: bool = True,
    ):
        super().__init__()
        if len(depths) != 3 or len(num_heads) != 3 or len(window_size) != 3:
            raise ValueError("MambaVisionPyramid expects 3-stage depths/num_heads/window_size to match BiM-VFI.")

        dims = [feat_channels, feat_channels * 2, feat_channels * 4]
        stem_hidden_dim = stem_hidden_dim or max(feat_channels // 2, 8)
        self.stem = Stem(in_chans=3, hidden_dim=stem_hidden_dim, out_dim=dims[0])
        self.dims = dims
        self.last_conv = nn.Conv2d(dims[-1], dims[-1], 3, 1, 1) if last_conv else nn.Identity()

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.levels = nn.ModuleList()
        for stage_idx in range(len(depths)):
            conv_stage = stage_idx < 2
            depth = depths[stage_idx]
            transformer_blocks = (
                list(range(depth // 2 + 1, depth)) if depth % 2 != 0 else list(range(depth // 2, depth))
            )
            level = MambaVisionStage(
                dim=dims[stage_idx],
                depth=depth,
                num_heads=num_heads[stage_idx],
                window_size=window_size[stage_idx],
                conv=conv_stage,
                downsample=(stage_idx < len(depths) - 1),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:stage_idx]):sum(depths[: stage_idx + 1])],
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=transformer_blocks,
            )
            self.levels.append(level)

        norm_layers = {
            "bn": nn.BatchNorm2d,
            "ln2d": LayerNorm2d,
            "identity": nn.Identity,
        }
        norm_cls = norm_layers.get(output_norm.lower())
        if norm_cls is None:
            raise ValueError(f"Unsupported output_norm: {output_norm}")
        self.out_norms = nn.ModuleList([norm_cls(dim) for dim in dims])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img):
        x = self.stem(img)
        outs = []
        for stage_idx, level in enumerate(self.levels):
            x, out = level(x)
            out = self.out_norms[stage_idx](out)
            outs.append(out.contiguous())

        outs[-1] = self.last_conv(outs[-1])
        return outs
