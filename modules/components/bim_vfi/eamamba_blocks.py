import math
import numbers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


ALLOWED_SCAN_TYPE = [None, "diagonal", "zorder", "zigzag", "hilbert"]
ALLOWED_SCAN_MERGE_METHOD = ["add", "concate"]
ALLOWED_SCAN_COUNT = [1, 2, 4, 8]
ALLOWED_CHANNEL_MIXER_TYPE = ["gdfn", "simple", "ffn", "cca"]


def to_tokens(x):
    b, c, h, w = x.shape
    return x.view(b, c, h * w).transpose(1, 2).contiguous()


def to_feature(x, h, w):
    b, _, c = x.shape
    return x.transpose(1, 2).contiguous().view(b, c, h, w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("BiasFreeLayerNorm expects a 1D normalized shape.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("WithBiasLayerNorm expects a 1D normalized shape.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, layernorm_type="WithBias"):
        super().__init__()
        if layernorm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_feature(self.body(to_tokens(x)), h, w)


class FFN(nn.Module):
    def __init__(self, dim, bias, ffn_expansion_factor=2):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim, 1, bias=bias)
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.act(x)
        x = self.project_out(x)
        return x


class CCABlock(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, bias=bias)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=bias),
            nn.Sigmoid(),
        )
        self.conv3 = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x * self.ca(x)
        x = self.conv3(x)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleFFN(nn.Module):
    def __init__(self, dim, bias, ffn_expansion_factor=2):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        if hidden_dim % 2 != 0:
            raise ValueError("SimpleFFN expects an even hidden dimension.")
        self.project_in = nn.Conv2d(dim, hidden_dim, 1, bias=bias)
        self.simple_gate = SimpleGate()
        self.project_out = nn.Conv2d(hidden_dim // 2, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.simple_gate(x)
        x = self.project_out(x)
        return x


class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_dim * 2,
            hidden_dim * 2,
            3,
            stride=1,
            padding=1,
            groups=hidden_dim * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


def get_channel_mixer_layer(channel_mixer_type, dim, ffn_expansion_factor, bias):
    if channel_mixer_type == "ffn":
        return FFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
    if channel_mixer_type == "cca":
        return CCABlock(dim=dim, bias=bias)
    if channel_mixer_type == "simple":
        return SimpleFFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
    if channel_mixer_type == "gdfn":
        return GDFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
    raise NotImplementedError(f"Unsupported channel mixer type: {channel_mixer_type}")


class ScanTransform:
    def __init__(self, scan_type, scan_count, merge_method):
        if scan_type not in ALLOWED_SCAN_TYPE:
            raise ValueError(f"scan_type should be one of {ALLOWED_SCAN_TYPE}, but got {scan_type}")
        if scan_count not in ALLOWED_SCAN_COUNT:
            raise ValueError(f"scan_count should be one of {ALLOWED_SCAN_COUNT}, but got {scan_count}")
        if scan_count > 1 and merge_method not in ALLOWED_SCAN_MERGE_METHOD:
            raise ValueError(
                f"scan_merge_method should be one of {ALLOWED_SCAN_MERGE_METHOD}, but got {merge_method}"
            )

        self.scan_type = scan_type
        self.scan_count = scan_count
        self.merge_method = merge_method
        self.index_dict = {}
        self.invert_index_dict = {}

        if self.scan_type == "diagonal":
            self.scan_method = self.diagonal_scan
        elif self.scan_type == "zorder":
            self.scan_method = self.z_order_scan
        elif self.scan_type == "zigzag":
            self.scan_method = self.zigzag_scan
        elif self.scan_type == "hilbert":
            self.scan_method = self.hilbert_scan
        else:
            self.scan_method = None

    @staticmethod
    def diagonal_scan(size):
        height, width = size
        indices = np.arange(height * width).reshape(height, width)
        result = []
        for sum_idx in range(height + width - 1):
            start_row = max(0, sum_idx - width + 1)
            end_row = min(sum_idx + 1, height)
            diagonal = [indices[i, sum_idx - i] for i in range(start_row, end_row)]
            result.extend(diagonal)
        return torch.tensor(result, dtype=torch.long)

    @staticmethod
    def hilbert_scan(size):
        from hilbert import encode

        height, width = size
        max_dim = 2 ** int(np.ceil(np.log2(max(height, width))))
        coords = np.array([[i, j] for i in range(max_dim) for j in range(max_dim)])
        order = int(np.log2(max_dim))
        indices = encode(coords, 2, order)
        sorted_indices = np.argsort(indices)
        sorted_coords = coords[sorted_indices]
        valid_mask = (sorted_coords[:, 0] < height) & (sorted_coords[:, 1] < width)
        valid_sorted_coords = sorted_coords[valid_mask]
        valid_sorted_indices = valid_sorted_coords[:, 0] * width + valid_sorted_coords[:, 1]
        return torch.tensor(valid_sorted_indices, dtype=torch.long)

    @staticmethod
    def zigzag_scan(size):
        height, width = size
        indices = np.arange(height * width).reshape(height, width)
        zigzag = np.concatenate(
            [np.diagonal(indices[::-1, :], k)[:: (2 * (k % 2) - 1)] for k in range(1 - height, width)]
        )
        return torch.tensor(zigzag, dtype=torch.long)

    @staticmethod
    def z_order_scan(size):
        height, width = size
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        y, x = y.flatten().to(torch.int32), x.flatten().to(torch.int32)

        def interleave_bits(a, b):
            magic = torch.tensor(
                [0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF],
                dtype=torch.int32,
            )

            def part1by1(n):
                n = (n | (n << 8)) & magic[4]
                n = (n | (n << 4)) & magic[3]
                n = (n | (n << 2)) & magic[2]
                n = (n | (n << 1)) & magic[1]
                return n

            a = part1by1(a)
            b = part1by1(b)
            return a | (b << 1)

        morton_codes = interleave_bits(x, y)
        return torch.argsort(morton_codes).to(torch.long)

    def tensor_reorder(self, x, size):
        if self.scan_count == 1:
            return x.unsqueeze(1)

        if self.merge_method == "add":
            x = x.repeat(1, self.scan_count, 1)

        height, width = size
        batch, channels, length = x.shape
        if channels % self.scan_count != 0:
            raise ValueError("Channel dimension must be divisible by scan_count for ScanTransform.")

        to_stack = []
        sections = torch.split(x, channels // self.scan_count, dim=1)
        for idx, section in enumerate(sections):
            if idx in range(2, len(sections), 4) or idx in range(3, len(sections), 4):
                if self.scan_type is None or idx >= 4:
                    section = section.view(batch, -1, height, width).permute(0, 1, 3, 2).reshape(
                        batch, -1, length
                    )
                else:
                    section = torch.flip(section.view(batch, -1, height, width), dims=(3,)).reshape(
                        batch, -1, length
                    )
            if idx % 2 == 1:
                section = torch.flip(section, dims=(2,))
            to_stack.append(section)

        return torch.stack(to_stack, dim=1)

    def tensor_restore(self, x, size):
        if self.scan_count == 1:
            return x.squeeze(1)

        height, width = size
        batch, _, channels, length = x.shape
        to_stack = []
        sections = torch.split(x, 1, dim=1)
        for idx, section in enumerate(sections):
            if idx % 2 == 1:
                section = torch.flip(section, dims=(3,))
            if idx in range(2, len(sections), 4) or idx in range(3, len(sections), 4):
                if self.scan_type is None or idx >= 4:
                    section = section.view(batch, 1, channels, width, height).permute(0, 1, 2, 4, 3).reshape(
                        batch, 1, channels, length
                    )
                else:
                    section = torch.flip(section.view(batch, 1, channels, height, width), dims=(4,)).reshape(
                        batch, 1, channels, length
                    )
            to_stack.append(section)

        x = torch.cat(to_stack, dim=1)
        if self.merge_method == "add":
            return x.sum(dim=1)
        return x.reshape(batch, self.scan_count * channels, length)

    def get_entry(self, size, device, get_invert=False):
        key = str(size)
        if key not in self.index_dict:
            if self.scan_method is None:
                raise ValueError("Scan index requested when scan_type is None.")
            index = self.scan_method(size).to(torch.long).cpu()
            invert_index = torch.empty_like(index)
            invert_index[index] = torch.arange(index.numel(), dtype=torch.long)
            self.index_dict[key] = index
            self.invert_index_dict[key] = invert_index
        entry = self.invert_index_dict[key] if get_invert else self.index_dict[key]
        return entry.to(device=device)

    def apply_scan_transform(self, x, size):
        x = self.tensor_reorder(x, size)
        if self.scan_type is None:
            return x

        index = self.get_entry(size, x.device, get_invert=False)
        if self.scan_count == 8:
            x[:, :4] = x[:, :4, :, index]
        else:
            x = x[:, :, :, index]
        return x

    def restore_scan_transform(self, x, size):
        if self.scan_type is not None:
            index = self.get_entry(size, x.device, get_invert=True)
            if self.scan_count == 8:
                x[:, :4] = x[:, :4, :, index]
            else:
                x = x[:, :, :, index]
        return self.tensor_restore(x, size)


class ExtendedMamba(nn.Module):
    def __init__(
        self,
        d_model,
        scan_transform,
        use_checkpoint=False,
        conv_2d=False,
        disable_z_branch=False,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError(
                "mamba_ssm is required to use EAMambaSynthesisNetwork. "
                "Please install mamba_ssm before enabling the EAMamba synthesis head."
            )

        self.use_checkpoint = use_checkpoint
        self.scan_transform = scan_transform
        self.scan_count = scan_transform.scan_count
        self.scan_merge_method = scan_transform.merge_method
        self.conv_2d = conv_2d

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.full_d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.disable_z_branch = disable_z_branch
        proj_out_dim = self.full_d_inner if disable_z_branch else self.full_d_inner * 2
        self.in_proj = nn.Linear(self.d_model, proj_out_dim, bias=bias)

        if conv_2d:
            self.conv = nn.Conv2d(
                self.full_d_inner,
                self.full_d_inner,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                groups=self.full_d_inner,
                bias=conv_bias,
            )
        else:
            self.conv = nn.Conv1d(
                self.full_d_inner,
                self.full_d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.full_d_inner,
                bias=conv_bias,
            )

        self.act = nn.SiLU()
        self.out_norm = nn.LayerNorm(self.full_d_inner)
        self.out_proj = nn.Linear(self.full_d_inner, self.d_model, bias=bias)

        if self.scan_merge_method == "concate":
            if self.full_d_inner % self.scan_count != 0:
                raise ValueError("expand * d_model must be divisible by scan_count for concate merge.")
            self.scan_d_inner = self.full_d_inner // self.scan_count
        else:
            self.scan_d_inner = self.full_d_inner

        x_proj = [nn.Linear(self.scan_d_inner, self.dt_rank + self.d_state * 2, bias=False) for _ in range(self.scan_count)]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in x_proj], dim=0))

        dt_projs = [
            self.dt_init(
                self.dt_rank,
                self.scan_d_inner,
                dt_scale=dt_scale,
                dt_init=dt_init,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
            )
            for _ in range(self.scan_count)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_projs], dim=0))

        self.A_logs = self.a_log_init(self.d_state, self.scan_d_inner, copies=self.scan_count, merge=True)
        self.Ds = self.d_init(self.scan_d_inner, copies=self.scan_count, merge=True)

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
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
    def a_log_init(d_state, d_inner, copies=1, merge=True):
        a = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1).contiguous()
        a_log = torch.log(a)
        if copies > 1:
            a_log = a_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                a_log = a_log.flatten(0, 1)
        a_log = nn.Parameter(a_log)
        a_log._no_weight_decay = True
        return a_log

    @staticmethod
    def d_init(d_inner, copies=1, merge=True):
        d = torch.ones(d_inner)
        if copies > 1:
            d = d.unsqueeze(0).repeat(copies, 1)
            if merge:
                d = d.flatten(0, 1)
        d = nn.Parameter(d)
        d._no_weight_decay = True
        return d

    def forward(self, hidden_states, x_size):
        batch, seqlen, _ = hidden_states.shape
        height, width = x_size

        if self.use_checkpoint:
            xz = checkpoint(self.in_proj, hidden_states, use_reentrant=False)
        else:
            xz = self.in_proj(hidden_states)
        xz = xz.transpose(1, 2).contiguous()

        if self.disable_z_branch:
            x = xz
            z = None
        else:
            x, z = xz.chunk(2, dim=1)

        if self.conv_2d:
            x = x.view(batch, self.full_d_inner, height, width)
        if self.use_checkpoint:
            x = checkpoint(self.conv, x, use_reentrant=False)
        else:
            x = self.conv(x)
        if self.conv_2d:
            x = x.view(batch, self.full_d_inner, seqlen)
        else:
            x = x[..., :seqlen]
        x = self.act(x)

        x = self.scan_transform.apply_scan_transform(x, x_size)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x, self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = x.float().reshape(batch, -1, seqlen)
        dts = dts.contiguous().float().reshape(batch, -1, seqlen)
        bs = bs.float().reshape(batch, self.scan_count, -1, seqlen)
        cs = cs.float().reshape(batch, self.scan_count, -1, seqlen)
        ds = self.Ds.float().view(-1)
        a_logs = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        y = selective_scan_fn(
            xs,
            dts,
            a_logs,
            bs,
            cs,
            ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(batch, self.scan_count, -1, seqlen)

        y = self.scan_transform.restore_scan_transform(y, x_size)
        y = y.transpose(1, 2).contiguous()
        y = self.out_norm(y)

        if z is not None:
            z = z.transpose(1, 2).contiguous()
            y = y * F.silu(z)

        if self.use_checkpoint:
            return checkpoint(self.out_proj, y, use_reentrant=False)
        return self.out_proj(y)


class MambaFormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_expansion_factor,
        bias,
        layernorm_type,
        scan_transform,
        mamba_cfg=None,
        use_checkpoint=False,
        channel_mixer_type="simple",
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm(dim, layernorm_type)
        self.mamba = ExtendedMamba(
            dim,
            scan_transform,
            use_checkpoint=use_checkpoint,
            **(mamba_cfg or {}),
        )
        self.norm2 = LayerNorm(dim, layernorm_type)
        self.ffn = get_channel_mixer_layer(channel_mixer_type, dim, ffn_expansion_factor, bias)

    def forward(self, x):
        batch, channels, height, width = x.shape
        shortcut = x
        x = to_tokens(self.norm1(x))
        x = self.mamba(x, (height, width))
        x = shortcut + to_feature(x, height, width)

        if self.use_checkpoint:
            x = x + checkpoint(self.ffn, self.norm2(x), use_reentrant=False)
        else:
            x = x + self.ffn(self.norm2(x))
        return x


def create_blocks(
    num_blocks,
    dim,
    ffn_expansion_factor,
    bias,
    layernorm_type,
    scan_transform,
    mamba_cfg,
    checkpoint_percentage,
    channel_mixer_type="simple",
):
    blocks = []
    num_checkpointed = math.ceil(checkpoint_percentage * num_blocks)
    for block_idx in range(num_blocks):
        blocks.append(
            MambaFormerBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                layernorm_type=layernorm_type,
                scan_transform=scan_transform,
                mamba_cfg=mamba_cfg,
                use_checkpoint=block_idx < num_checkpointed,
                channel_mixer_type=channel_mixer_type,
            )
        )
    return nn.Sequential(*blocks)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)
