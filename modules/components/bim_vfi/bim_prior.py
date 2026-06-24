import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act1 = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act2 = nn.PReLU(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.act1(residual)
        residual = self.conv2(residual)
        return self.act2(x + residual)


class FeatureProjector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.PReLU(out_channels),
        )

    def forward(self, x, target_hw):
        x = self.block(x)
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x


class BiMPriorPredictor(nn.Module):
    def __init__(
        self,
        in_channels=None,
        hidden_channels=32,
        feat_channels=32,
        img_channels=3,
        use_img_diff=True,
        use_state=True,
    ):
        super().__init__()
        c = hidden_channels
        self.hidden_channels = c
        self.use_img_diff = use_img_diff
        self.use_state = use_state

        raw_in_channels = img_channels * 2 + 1
        if use_img_diff:
            raw_in_channels += img_channels

        self.raw_encoder = ConvBlock(raw_in_channels, c)
        self.level0_proj = FeatureProjector(feat_channels * 3, c)
        self.level1_proj = FeatureProjector(feat_channels * 6, c)
        self.level2_proj = FeatureProjector(feat_channels * 12, c)
        self.state_proj = FeatureProjector(5, c) if use_state else None

        fusion_in_channels = c * 4 + (c if use_state else 0)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, c * 4, 3, padding=1),
            nn.PReLU(c * 4),
            ResidualBlock(c * 4),
            ResidualBlock(c * 4),
            nn.Conv2d(c * 4, c * 2, 3, padding=1),
            nn.PReLU(c * 2),
            ResidualBlock(c * 2),
            nn.Conv2d(c * 2, c, 3, padding=1),
            nn.PReLU(c),
        )

        # Predict residual corrections around the uniform BiM prior so training
        # starts from the original baseline behavior.
        self.head_r = nn.Conv2d(c, 1, 3, padding=1)
        self.head_phi = nn.Conv2d(c, 2, 3, padding=1)
        nn.init.zeros_(self.head_r.weight)
        nn.init.zeros_(self.head_r.bias)
        nn.init.zeros_(self.head_phi.weight)
        nn.init.zeros_(self.head_phi.bias)

    def _build_time_map(self, time_step, batch, height, width, device, dtype):
        if not torch.is_tensor(time_step):
            time_step = torch.tensor(time_step, device=device, dtype=dtype)
        else:
            time_step = time_step.to(device=device, dtype=dtype)

        if time_step.dim() == 0:
            time_step = time_step.view(1).repeat(batch)
        elif time_step.dim() == 1 and time_step.shape[0] == 1 and batch > 1:
            time_step = time_step.repeat(batch)
        else:
            time_step = time_step.reshape(batch)

        return time_step.view(batch, 1, 1, 1).expand(batch, 1, height, width)

    def _zeros(self, ref_tensor, target_hw):
        batch = ref_tensor.shape[0]
        return ref_tensor.new_zeros((batch, self.hidden_channels, target_hw[0], target_hw[1]))

    def _build_feature_pair(self, feat0, feat1):
        return torch.cat([feat0, feat1, torch.abs(feat0 - feat1)], dim=1)

    def forward(
        self,
        img0,
        img1,
        time_step,
        feat0_pyr=None,
        feat1_pyr=None,
        last_flow=None,
        last_occ=None,
        target_hw=None,
    ):
        batch, _, height, width = img0.shape
        time_map = self._build_time_map(
            time_step, batch, height, width, img0.device, img0.dtype
        )

        if target_hw is None:
            if feat0_pyr is not None:
                target_hw = feat0_pyr[-1].shape[-2:]
            else:
                target_hw = (height, width)

        raw_inputs = [img0, img1]
        if self.use_img_diff:
            raw_inputs.append(torch.abs(img1 - img0))
        raw_inputs.append(time_map)

        raw_feat = self.raw_encoder(torch.cat(raw_inputs, dim=1))
        if raw_feat.shape[-2:] != target_hw:
            raw_feat = F.interpolate(raw_feat, size=target_hw, mode="bilinear", align_corners=False)

        fused_feats = [raw_feat]

        if feat0_pyr is not None and feat1_pyr is not None:
            fused_feats.append(self.level0_proj(self._build_feature_pair(feat0_pyr[0], feat1_pyr[0]), target_hw))
            fused_feats.append(self.level1_proj(self._build_feature_pair(feat0_pyr[1], feat1_pyr[1]), target_hw))
            fused_feats.append(self.level2_proj(self._build_feature_pair(feat0_pyr[2], feat1_pyr[2]), target_hw))
        else:
            fused_feats.extend([self._zeros(raw_feat, target_hw) for _ in range(3)])

        if self.use_state:
            if last_flow is not None and last_occ is not None:
                state_input = torch.cat([last_flow, last_occ], dim=1)
                fused_feats.append(self.state_proj(state_input, target_hw))
            else:
                fused_feats.append(self._zeros(raw_feat, target_hw))

        fused = self.fusion(torch.cat(fused_feats, dim=1))

        time_map_target = F.interpolate(time_map, size=target_hw, mode="bilinear", align_corners=False)
        uniform_r = time_map_target
        uniform_phi = fused.new_zeros((batch, 2, target_hw[0], target_hw[1]))
        uniform_phi[:, 0] = -1.0

        # Old parameterization:
        # r_delta = 0.5 * torch.tanh(self.head_r(fused))
        # r = (uniform_r + r_delta).clamp(0.0, 1.0)
        uniform_r_safe = uniform_r.clamp(1e-4, 1.0 - 1e-4)
        uniform_r_logit = torch.log(uniform_r_safe) - torch.log1p(-uniform_r_safe)
        r_delta = 0.5 * self.head_r(fused)
        r = torch.sigmoid(uniform_r_logit + r_delta)
        phi_delta = 0.1 * self.head_phi(fused)
        phi = F.normalize(uniform_phi + phi_delta, dim=1, eps=1e-6)

        return r, phi
