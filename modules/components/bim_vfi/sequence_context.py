import torch.nn as nn


class TemporalContextInteraction(nn.Module):
    """
    Lightweight temporal interaction on the last encoder feature.
    Input shape: [B, T, C, H, W]
    """

    def __init__(self, channels, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super(TemporalContextInteraction, self).__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )

        hidden_channels = max(channels, int(channels * mlp_ratio))
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Linear(channels, channels * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, feat_seq, frame_mask=None):
        if feat_seq.dim() != 5:
            raise ValueError(f"TemporalContextInteraction expects 5D input, got {feat_seq.shape}")

        B, T, C, _, _ = feat_seq.shape
        if T <= 1:
            return feat_seq

        tokens = feat_seq.mean(dim=(-1, -2))
        valid_mask = None
        key_padding_mask = None
        if frame_mask is not None:
            valid_mask = frame_mask.to(device=feat_seq.device, dtype=feat_seq.dtype).unsqueeze(-1)
            key_padding_mask = frame_mask.to(device=feat_seq.device) <= 0
            tokens = tokens * valid_mask

        attn_tokens = self.norm1(tokens)
        attn_tokens, _ = self.attn(
            attn_tokens,
            attn_tokens,
            attn_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + attn_tokens
        tokens = tokens + self.ffn(self.norm2(tokens))

        scale, bias = self.modulation(tokens).chunk(2, dim=-1)
        feat_seq = feat_seq * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + bias.unsqueeze(-1).unsqueeze(-1)

        if frame_mask is not None:
            valid_mask = valid_mask.unsqueeze(-1).unsqueeze(-1)
            feat_seq = feat_seq * valid_mask

        return feat_seq
