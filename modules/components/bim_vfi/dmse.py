import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
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
    def __init__(self, channels: int):
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
    def __init__(self, in_channels: int, out_channels: int):
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


class TokenFiLM(nn.Module):
    def __init__(
        self,
        token_channels: int,
        out_channels: int,
        hidden_channels: int = None,
        init_zero: bool = True,
    ):
        super().__init__()
        hidden_channels = hidden_channels or max(out_channels * 4, token_channels)
        self.norm = nn.LayerNorm(token_channels)
        self.mlp = nn.Sequential(
            nn.Linear(token_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )
        if init_zero:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, tokens):
        if tokens is None:
            return x
        if tokens.dim() == 3:
            tokens = tokens.mean(dim=1)
        tokens = self.norm(tokens.to(device=x.device, dtype=x.dtype))
        gamma, beta = self.mlp(tokens).chunk(2, dim=1)
        gamma = gamma.view(x.shape[0], x.shape[1], 1, 1)
        beta = beta.view(x.shape[0], x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta


class DMSE(nn.Module):
    """
    Dynamic Motion-State Estimator.

    This standalone module keeps the useful MTMC ingredients, but exposes them
    as a gated residual motion-state estimator:

        rho = (1 - g) * tau + g * sigmoid(logit(tau) + Delta rho)
        d   = Norm((1 - g) * d0 + g * (d0 + Delta d)), d0 = (-1, 0)

    Inputs are intentionally named around ADAMS:
    - dynamic_feat0_pyr / dynamic_feat1_pyr: two anchor Dynamic feature pyramids.
    - dynamic_context_pyr: target-time Dynamic context maps from ADAMS.
    - global_dynamic_tokens_pyr: target-time global Dynamic tokens from ADAMS.

    The global tokens are used only as FiLM modulation over the fused state
    feature. They are not expanded into dense spatial maps.
    """

    accepts_global_dynamic_tokens = True

    def __init__(
        self,
        hidden_channels: int = 32,
        feat_channels: int = 32,
        img_channels: int = 3,
        use_img_diff: bool = True,
        use_state: bool = True,
        use_dynamic_context: bool = True,
        use_global_token_film: bool = True,
        global_token_hidden_channels: int = None,
        token_init_zero: bool = True,
        rho_delta_scale: float = 0.5,
        direction_delta_scale: float = 0.1,
        gate_init_bias: float = -2.0,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        c = int(hidden_channels)
        self.hidden_channels = c
        self.feat_channels = int(feat_channels)
        self.img_channels = int(img_channels)
        self.use_img_diff = bool(use_img_diff)
        self.use_state = bool(use_state)
        self.use_dynamic_context = bool(use_dynamic_context)
        self.use_global_token_film = bool(use_global_token_film)
        self.rho_delta_scale = float(rho_delta_scale)
        self.direction_delta_scale = float(direction_delta_scale)
        self.eps = float(eps)

        raw_in_channels = self.img_channels * 2 + 1
        if self.use_img_diff:
            raw_in_channels += self.img_channels
        self.raw_encoder = ConvBlock(raw_in_channels, c)

        self.level0_proj = FeatureProjector(self.feat_channels * 3, c)
        self.level1_proj = FeatureProjector(self.feat_channels * 6, c)
        self.level2_proj = FeatureProjector(self.feat_channels * 12, c)
        self.state_proj = FeatureProjector(5, c) if self.use_state else None

        if self.use_dynamic_context:
            self.dynamic_context_level0_proj = FeatureProjector(self.feat_channels, c)
            self.dynamic_context_level1_proj = FeatureProjector(self.feat_channels * 2, c)
            self.dynamic_context_level2_proj = FeatureProjector(self.feat_channels * 4, c)
        else:
            self.dynamic_context_level0_proj = None
            self.dynamic_context_level1_proj = None
            self.dynamic_context_level2_proj = None

        fusion_in_channels = c * 4
        if self.use_state:
            fusion_in_channels += c
        if self.use_dynamic_context:
            fusion_in_channels += c * 3

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

        if self.use_global_token_film:
            self.global_token_level0_film = TokenFiLM(
                self.feat_channels,
                c,
                hidden_channels=global_token_hidden_channels,
                init_zero=token_init_zero,
            )
            self.global_token_level1_film = TokenFiLM(
                self.feat_channels * 2,
                c,
                hidden_channels=global_token_hidden_channels,
                init_zero=token_init_zero,
            )
            self.global_token_level2_film = TokenFiLM(
                self.feat_channels * 4,
                c,
                hidden_channels=global_token_hidden_channels,
                init_zero=token_init_zero,
            )
        else:
            self.global_token_level0_film = None
            self.global_token_level1_film = None
            self.global_token_level2_film = None

        self.head_rho_delta = nn.Conv2d(c, 1, 3, padding=1)
        self.head_direction_delta = nn.Conv2d(c, 2, 3, padding=1)
        self.head_gate = nn.Conv2d(c, 1, 3, padding=1)
        self._init_state_heads(gate_init_bias)

    def _init_state_heads(self, gate_init_bias: float):
        nn.init.zeros_(self.head_rho_delta.weight)
        nn.init.zeros_(self.head_rho_delta.bias)
        nn.init.zeros_(self.head_direction_delta.weight)
        nn.init.zeros_(self.head_direction_delta.bias)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.constant_(self.head_gate.bias, float(gate_init_bias))

    @staticmethod
    def _select_target_tensor(x, tau_index: int = 0):
        if x is None:
            return None
        if x.dim() in (3, 5):
            return x[:, int(tau_index)]
        return x

    def _normalize_tau(self, tau, batch: int, device, dtype, tau_index: int = 0):
        if tau is None:
            tau = torch.tensor(0.5, device=device, dtype=dtype)
        elif not torch.is_tensor(tau):
            tau = torch.tensor(tau, device=device, dtype=dtype)
        else:
            tau = tau.to(device=device, dtype=dtype)

        if tau.dim() == 0:
            return tau.view(1).expand(batch)
        if tau.dim() == 1:
            if tau.numel() == 1 and batch > 1:
                return tau.reshape(1).expand(batch)
            if tau.numel() == batch:
                return tau.reshape(batch)
            return tau.reshape(1, -1).expand(batch, -1)[:, int(tau_index)]
        if tau.dim() == 2:
            if tau.shape[0] == 1 and batch > 1:
                tau = tau.expand(batch, -1)
            if tau.shape[0] != batch:
                raise ValueError(f"tau batch mismatch: expected {batch}, got {tau.shape[0]}.")
            return tau[:, int(tau_index)]
        return tau.reshape(batch, -1)[:, int(tau_index)]

    def _build_tau_map(self, tau, batch: int, height: int, width: int, device, dtype, tau_index: int = 0):
        tau = self._normalize_tau(tau, batch, device, dtype, tau_index=tau_index)
        return tau.view(batch, 1, 1, 1).expand(batch, 1, height, width)

    def _zeros(self, ref_tensor, target_hw):
        batch = ref_tensor.shape[0]
        return ref_tensor.new_zeros((batch, self.hidden_channels, target_hw[0], target_hw[1]))

    @staticmethod
    def _build_feature_pair(feat0, feat1):
        return torch.cat([feat0, feat1, torch.abs(feat0 - feat1)], dim=1)

    def _project_feature_pair_pyramid(self, feat0_pyr, feat1_pyr, target_hw, ref_tensor):
        if feat0_pyr is None or feat1_pyr is None:
            return [self._zeros(ref_tensor, target_hw) for _ in range(3)]
        if len(feat0_pyr) < 3 or len(feat1_pyr) < 3:
            raise ValueError(
                f"Expected at least 3 dynamic anchor feature levels, got "
                f"{len(feat0_pyr)} and {len(feat1_pyr)}."
            )

        projectors = [self.level0_proj, self.level1_proj, self.level2_proj]
        projected = []
        for feat0, feat1, projector in zip(feat0_pyr[:3], feat1_pyr[:3], projectors):
            feat0 = self._select_target_tensor(feat0)
            feat1 = self._select_target_tensor(feat1)
            projected.append(projector(self._build_feature_pair(feat0, feat1), target_hw))
        return projected

    def _project_context_pyramid(self, context_pyr, target_hw, ref_tensor, tau_index: int = 0):
        if not self.use_dynamic_context:
            return []
        if context_pyr is None:
            return [self._zeros(ref_tensor, target_hw) for _ in range(3)]
        if len(context_pyr) < 3:
            raise ValueError(f"Expected at least 3 dynamic context levels, got {len(context_pyr)}.")

        projectors = [
            self.dynamic_context_level0_proj,
            self.dynamic_context_level1_proj,
            self.dynamic_context_level2_proj,
        ]
        projected = []
        for context_feat, projector in zip(context_pyr[:3], projectors):
            context_feat = self._select_target_tensor(context_feat, tau_index=tau_index)
            projected.append(projector(context_feat, target_hw))
        return projected

    def _apply_global_dynamic_tokens(self, fused, global_dynamic_tokens_pyr, tau_index: int = 0):
        if not self.use_global_token_film or global_dynamic_tokens_pyr is None:
            return fused
        if torch.is_tensor(global_dynamic_tokens_pyr):
            return self.global_token_level2_film(
                fused,
                self._select_target_tensor(global_dynamic_tokens_pyr, tau_index=tau_index),
            )
        if len(global_dynamic_tokens_pyr) == 0:
            return fused

        film_layers = [
            self.global_token_level0_film,
            self.global_token_level1_film,
            self.global_token_level2_film,
        ]
        for tokens, film in zip(global_dynamic_tokens_pyr[:3], film_layers):
            tokens = self._select_target_tensor(tokens, tau_index=tau_index)
            fused = film(fused, tokens)
        return fused

    def _build_prior_direction(self, batch: int, target_hw, ref_tensor):
        direction = ref_tensor.new_zeros((batch, 2, target_hw[0], target_hw[1]))
        direction[:, 0] = -1.0
        return direction

    def forward(
        self,
        img0,
        img1,
        tau=None,
        dynamic_feat0_pyr=None,
        dynamic_feat1_pyr=None,
        dynamic_context_pyr=None,
        global_dynamic_tokens_pyr=None,
        last_flow=None,
        last_occ=None,
        target_hw=None,
        tau_index: int = 0,
        return_dict: bool = True,
        feat0_pyr=None,
        feat1_pyr=None,
        motion_context_pyr=None,
        global_motion_tokens=None,
    ):
        if dynamic_feat0_pyr is None:
            dynamic_feat0_pyr = feat0_pyr
        if dynamic_feat1_pyr is None:
            dynamic_feat1_pyr = feat1_pyr
        if dynamic_context_pyr is None:
            dynamic_context_pyr = motion_context_pyr
        if global_dynamic_tokens_pyr is None:
            global_dynamic_tokens_pyr = global_motion_tokens

        batch, _, height, width = img0.shape
        tau_map = self._build_tau_map(
            tau,
            batch,
            height,
            width,
            img0.device,
            img0.dtype,
            tau_index=tau_index,
        )

        if target_hw is None:
            if dynamic_feat0_pyr is not None:
                target_hw = dynamic_feat0_pyr[-1].shape[-2:]
            elif last_flow is not None:
                target_hw = last_flow.shape[-2:]
            else:
                target_hw = (height, width)

        raw_inputs = [img0, img1]
        if self.use_img_diff:
            raw_inputs.append(torch.abs(img1 - img0))
        raw_inputs.append(tau_map)

        raw_feat = self.raw_encoder(torch.cat(raw_inputs, dim=1))
        if raw_feat.shape[-2:] != target_hw:
            raw_feat = F.interpolate(raw_feat, size=target_hw, mode="bilinear", align_corners=False)

        fused_feats = [raw_feat]
        fused_feats.extend(
            self._project_feature_pair_pyramid(
                dynamic_feat0_pyr,
                dynamic_feat1_pyr,
                target_hw,
                raw_feat,
            )
        )

        if self.use_state:
            if last_flow is not None and last_occ is not None:
                state_input = torch.cat([last_flow, last_occ], dim=1)
                fused_feats.append(self.state_proj(state_input, target_hw))
            else:
                fused_feats.append(self._zeros(raw_feat, target_hw))

        fused_feats.extend(
            self._project_context_pyramid(
                dynamic_context_pyr,
                target_hw,
                raw_feat,
                tau_index=tau_index,
            )
        )

        hidden = self.fusion(torch.cat(fused_feats, dim=1))
        hidden = self._apply_global_dynamic_tokens(
            hidden,
            global_dynamic_tokens_pyr,
            tau_index=tau_index,
        )

        tau_map_target = F.interpolate(tau_map, size=target_hw, mode="bilinear", align_corners=False)
        tau_safe = tau_map_target.clamp(self.eps, 1.0 - self.eps)
        tau_logit = torch.log(tau_safe) - torch.log1p(-tau_safe)

        rho_delta = self.rho_delta_scale * self.head_rho_delta(hidden)
        rho_candidate = torch.sigmoid(tau_logit + rho_delta)

        prior_direction = self._build_prior_direction(batch, target_hw, hidden)
        direction_delta = self.direction_delta_scale * self.head_direction_delta(hidden)
        direction_candidate = prior_direction + direction_delta

        gate = torch.sigmoid(self.head_gate(hidden))
        rho = (1.0 - gate) * tau_map_target + gate * rho_candidate
        direction = F.normalize(
            (1.0 - gate) * prior_direction + gate * direction_candidate,
            dim=1,
            eps=self.eps,
        )

        if not return_dict:
            return rho, direction

        return {
            "rho": rho,
            "direction": direction,
            "r": rho,
            "phi": direction,
            "gate": gate,
            "rho_delta": rho_delta,
            "direction_delta": direction_delta,
            "rho_candidate": rho_candidate,
            "direction_candidate": F.normalize(direction_candidate, dim=1, eps=self.eps),
            "prior_rho": tau_map_target,
            "prior_direction": prior_direction,
            "hidden": hidden,
            "state": (rho, direction),
        }


DynamicMotionStateEstimator = DMSE
