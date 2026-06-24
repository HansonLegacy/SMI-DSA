import torch
import torch.nn.functional as F
import torch.nn as nn
import os

from .backwarp import backwarp
from .resnet_encoder import ResNetPyramid
from .caun import CAUN
from .bimfn import BiMFN
from .bim_prior import BiMPriorPredictor
from .sn import DeepSynthesisNetwork
from .sn import EAMambaSynthesisNetwork
from .sn import SynthesisNetwork
from .local_temporal_context_encoder import LTCE
from .global_temporal_memory_encoder import GTME
from .pair_conditioned_global_memory import PairConditionedGlobalMemory, TokenContextPyramidAdapter
from .mtmc import MTMC, PairGlobalMemoryMTMC
from .tcar import TCAR, PairGlobalMemoryTCAR
from .adams_encoder import ADAMS
from .dmse import DMSE
from .asd import ASD
from .video_context_encoder import LocalVideoContextPyramid
from .video_context_sn import VideoContextEAMambaSynthesisNetwork
from .video_context_unet_sn import VideoContextUNetSynthesisNetwork

# TransUNet 作为新sn
from .sn import TransUNetSynthesisNetwork

from ..components import register

from utils.padder import InputPadder


@register('bim_vfi')
class BiMVFI(nn.Module):
    def __init__(
        self,
        pyr_level=3,
        feat_channels=32,
        use_transunet=False,
        use_deep_synthesis=False,
        use_eamamba_synthesis=False,
        transunet_pretrained=None,
        adapter_hidden_size=None,
        adapter_num_heads=None,
        adapter_mlp_dim=None,
        adapter_num_layers=2,
        adapter_pool_size=8,
        adapter_init_mode="identity",
        adapter_init_scale=1e-3,
        use_bim_prior_predictor=False,
        bim_prior_hidden_channels=32,
        bim_prior_pretrained=None,
        freeze_bim_prior=False,
        bim_prior_use_img_diff=True,
        bim_prior_use_last_state=True,
        eamamba_num_blocks=(2, 3, 3, 4),
        eamamba_num_refinement_blocks=1,
        eamamba_ffn_expansion_factor=2.0,
        eamamba_bias=False,
        eamamba_layernorm_type="WithBias",
        eamamba_checkpoint_percentage=0.0,
        eamamba_channel_mixer_type="Simple",
        eamamba_mamba_cfg=None,
        use_video_sequence_context=False,
        sequence_context_on_motion=True,
        sequence_context_on_context=True,
        sequence_context_num_heads=4,
        sequence_context_mlp_ratio=2.0,
        sequence_context_dropout=0.0,
        video_context_temporal_sigma=2.0,
        use_local_fine_context=False,
        use_ltce=None,
        use_local_motion_context_for_bim_prior=False,
        use_local_detail_context_for_sn=False,
        local_fine_context=None,
        use_global_memory_context=False,
        use_gtme=None,
        use_global_memory_for_sn=False,
        global_memory_context=None,
        global_memory_encoder=None,
        use_pair_global_memory=None,
        pair_global_memory=None,
        use_mtmc=False,
        mtmc_hidden_channels=32,
        mtmc_use_img_diff=True,
        mtmc_use_last_state=True,
        mtmc_use_local_motion_context=True,
        mtmc_use_global_context=True,
        mtmc_pretrained=None,
        freeze_mtmc=False,
        mtmc=None,
        use_adams_encoder=False,
        adams_encoder=None,
        use_dmse=False,
        dmse=None,
        use_asd=False,
        asd=None,
        use_tcar=False,
        tcar=None,
        use_teacher_branch=True,
        **kwargs
    ):
        super(BiMVFI, self).__init__()

        def _component_name(component_cfg, default_name):
            component_cfg = {} if component_cfg is None else dict(component_cfg)
            return str(component_cfg.get("name", default_name)).lower()

        def _extract_component_args(component_cfg, expected_name, accepted_names=None):
            component_cfg = {} if component_cfg is None else dict(component_cfg)
            component_name = component_cfg.get("name", expected_name)
            valid_names = [str(expected_name).lower()]
            if accepted_names is not None:
                valid_names.extend(str(name).lower() for name in accepted_names)
            if component_name is not None and str(component_name).lower() not in valid_names:
                raise ValueError(
                    f"Expected component one of {valid_names}, but got '{component_name}'."
                )
            return dict(component_cfg.get("args", {}))

        self.pyr_level = pyr_level
        self.use_teacher_branch = bool(use_teacher_branch)
        self.mfe = ResNetPyramid(feat_channels)
        self.cfe = ResNetPyramid(feat_channels)
        self.bimfn = BiMFN(feat_channels)
        self.use_adams_encoder = bool(use_adams_encoder)
        self.adams_encoder = None
        self.adams_output = None
        if self.use_adams_encoder:
            adams_args = _extract_component_args(adams_encoder, "adams")
            adams_args.pop("dynamic_encoder", None)
            adams_args.pop("anatomical_encoder", None)
            adams_args.setdefault("feat_channels", feat_channels)
            adams_args.setdefault("image_channels", 3)
            adams_args.setdefault("stem_image_channels", 3)
            self.adams_encoder = ADAMS(
                dynamic_encoder=self.mfe,
                anatomical_encoder=self.cfe,
                **adams_args,
            )
        self.use_video_sequence_context = use_video_sequence_context
        self.video_sequence_input = None
        self.temporal_context_input = None
        self.video_context_extractor = None
        self.sn_accepts_video_context = False
        self.use_local_fine_context = bool(use_local_fine_context)
        self.use_ltce = self.use_local_fine_context and (
            True if use_ltce is None else bool(use_ltce)
        )
        self.use_local_motion_context_for_bim_prior = (
            self.use_ltce and bool(use_local_motion_context_for_bim_prior)
        )
        self.use_local_detail_context_for_sn = (
            self.use_ltce and bool(use_local_detail_context_for_sn)
        )
        self.local_fine_context_cfg = {} if local_fine_context is None else dict(local_fine_context)
        self.ltce = None
        self.local_temporal_context_encoder = None
        if self.use_ltce:
            ltce_args = _extract_component_args(
                self.local_fine_context_cfg.get("encoder", {}),
                "ltce",
            )
            ltce_args.setdefault("feat_channels", feat_channels)
            ltce_args.setdefault("temporal_sigma", video_context_temporal_sigma)
            self.ltce = LTCE(**ltce_args)
            self.local_temporal_context_encoder = self.ltce

        self.use_global_memory_context = bool(use_global_memory_context)
        self.global_memory_context_cfg = (
            {} if global_memory_context is None else dict(global_memory_context)
        )
        pair_global_memory_switch = (
            self.global_memory_context_cfg.get("use_pair_conditioned_memory", False)
            if use_pair_global_memory is None
            else use_pair_global_memory
        )
        self.use_pair_global_memory = self.use_global_memory_context and bool(pair_global_memory_switch)
        self.use_gtme = (
            self.use_global_memory_context
            and not self.use_pair_global_memory
            and (True if use_gtme is None else bool(use_gtme))
        )
        self.use_global_memory_for_sn = (
            (self.use_gtme or self.use_pair_global_memory) and bool(use_global_memory_for_sn)
        )
        self.gtme = None
        self.pair_global_memory = None
        self.pair_motion_context_adapter = None
        self.pair_structure_context_adapter = None
        self.pair_global_memory_output = None
        self.pair_global_memory_token_channels = None
        self.global_temporal_memory_encoder = None
        if self.use_pair_global_memory:
            self.pair_global_memory_cfg = (
                {} if pair_global_memory is None else dict(pair_global_memory)
            )
            pair_memory_args = _extract_component_args(
                self.pair_global_memory_cfg,
                "pair_conditioned_global_memory",
            )
            pair_memory_args.setdefault("channels", 256)
            pair_memory_args.setdefault("image_channels", 3)
            pair_memory_args.setdefault("use_motion_prior_head", False)
            self.pair_global_memory = PairConditionedGlobalMemory(**pair_memory_args)
            pair_token_channels = int(pair_memory_args.get("channels", 256))
            self.pair_global_memory_token_channels = pair_token_channels

            adapter_cfg = dict(self.pair_global_memory_cfg.get("adapter", {}))
            adapter_cfg.setdefault("token_channels", pair_token_channels)
            adapter_cfg.setdefault("feat_channels", feat_channels)
            self.pair_motion_context_adapter = TokenContextPyramidAdapter(**adapter_cfg)
            self.pair_structure_context_adapter = TokenContextPyramidAdapter(**adapter_cfg)
            self.global_temporal_memory_encoder = self.pair_global_memory
        elif self.use_gtme:
            self.pair_global_memory_cfg = {}
            gtme_args = _extract_component_args(global_memory_encoder, "gtme")
            gtme_args.setdefault("feat_channels", feat_channels)
            if (
                "local_window_size" not in gtme_args
                and self.global_memory_context_cfg.get("local_window_size") is not None
            ):
                gtme_args["local_window_size"] = self.global_memory_context_cfg["local_window_size"]
            if (
                "spatial_downsample_factor" not in gtme_args
                and self.global_memory_context_cfg.get("spatial_downsample_factor") is not None
            ):
                gtme_args["spatial_downsample_factor"] = self.global_memory_context_cfg[
                    "spatial_downsample_factor"
                ]
            self.gtme = GTME(**gtme_args)
            self.global_temporal_memory_encoder = self.gtme

        if self.use_video_sequence_context:
            self.video_context_extractor = LocalVideoContextPyramid(
                feat_channels,
                temporal_sigma=video_context_temporal_sigma,
            )

        self.use_asd = bool(use_asd)
        self.use_tcar = bool(use_tcar)
        if self.use_asd and self.use_tcar:
            raise ValueError("use_asd and use_tcar are mutually exclusive. Enable only one decoder.")
        if self.use_asd and not self.use_adams_encoder:
            raise ValueError("use_asd=True requires use_adams_encoder=True for anatomical context.")

        synthesis_variant_count = sum(
            bool(flag)
            for flag in (
                use_transunet,
                use_deep_synthesis,
                use_eamamba_synthesis,
                self.use_tcar,
                self.use_asd,
            )
        )
        if synthesis_variant_count > 1:
            raise ValueError(
                "Only one synthesis replacement can be enabled at a time: "
                "use_transunet, use_deep_synthesis, use_eamamba_synthesis, use_tcar, use_asd."
            )

        if self.use_asd:
            asd_args = _extract_component_args(asd, "asd")
            asd_args.setdefault("feat_channels", feat_channels)
            self.sn = ASD(**asd_args)
        elif self.use_tcar:
            tcar_name = _component_name(tcar, "tcar")
            if tcar_name in ("pair_global_memory_tcar", "pair_conditioned_global_memory_tcar"):
                tcar_args = _extract_component_args(
                    tcar,
                    "pair_global_memory_tcar",
                    accepted_names=("pair_conditioned_global_memory_tcar",),
                )
                tcar_args.setdefault("feat_channels", feat_channels)
                tcar_args.setdefault(
                    "global_token_channels",
                    self.pair_global_memory_token_channels or 256,
                )
                self.sn = PairGlobalMemoryTCAR(**tcar_args)
            else:
                tcar_args = _extract_component_args(tcar, "tcar")
                tcar_args.setdefault("feat_channels", feat_channels)
                self.sn = TCAR(**tcar_args)
        elif use_eamamba_synthesis:
            synthesis_cls = (
                VideoContextEAMambaSynthesisNetwork if self.use_video_sequence_context else EAMambaSynthesisNetwork
            )
            self.sn = synthesis_cls(
                feat_channels=feat_channels,
                num_blocks=eamamba_num_blocks,
                num_refinement_blocks=eamamba_num_refinement_blocks,
                ffn_expansion_factor=eamamba_ffn_expansion_factor,
                bias=eamamba_bias,
                layernorm_type=eamamba_layernorm_type,
                checkpoint_percentage=eamamba_checkpoint_percentage,
                channel_mixer_type=eamamba_channel_mixer_type,
                mamba_cfg=eamamba_mamba_cfg,
            )
            self.sn_accepts_video_context = self.use_video_sequence_context
        elif use_transunet:
            if self.use_video_sequence_context:
                raise ValueError(
                    "Video sequence context is not wired into TransUNetSynthesisNetwork. "
                    "Use the dedicated sequence SN variant instead."
                )
            self.sn = TransUNetSynthesisNetwork(
                feat_channels=feat_channels,
                img_size=256,
                use_transunet=True,
                adapter_hidden_size=adapter_hidden_size,
                adapter_num_heads=adapter_num_heads,
                adapter_mlp_dim=adapter_mlp_dim,
                adapter_num_layers=adapter_num_layers,
                adapter_pool_size=adapter_pool_size,
                adapter_init_mode=adapter_init_mode,
                adapter_init_scale=adapter_init_scale,
            )
            if transunet_pretrained and os.path.exists(transunet_pretrained):
                self.sn.load_pretrained_vit(transunet_pretrained)
                print(f"[BiMVFI] Loaded TransUNet pretrained from {transunet_pretrained}")
        elif use_deep_synthesis:
            if self.use_video_sequence_context:
                raise ValueError(
                    "Video sequence context is not wired into DeepSynthesisNetwork. "
                    "Use the dedicated sequence SN variant instead."
                )
            self.sn = DeepSynthesisNetwork(feat_channels)
        else:
            synthesis_cls = VideoContextUNetSynthesisNetwork if self.use_video_sequence_context else SynthesisNetwork
            self.sn = synthesis_cls(feat_channels)
            self.sn_accepts_video_context = self.use_video_sequence_context
            # 原流程

        # # self.sn = SynthesisNetwork(feat_channels)

        # # TrasUNet
        # self.sn = TransUNetSynthesisNetwork(
        #     feat_channels=feat_channels,
        #     img_size=256,  # 根据你的输入尺寸调整
        #     use_transunet=True
        # )

        self.feat_channels = feat_channels
        self.caun = CAUN(feat_channels)
        self.use_dmse = bool(use_dmse)
        if self.use_dmse and not self.use_adams_encoder:
            raise ValueError("use_dmse=True requires use_adams_encoder=True for dynamic context.")
        if self.use_dmse and use_mtmc:
            raise ValueError("use_dmse and use_mtmc are mutually exclusive. Enable only one motion-state estimator.")
        self.use_mtmc = bool(use_mtmc)
        if self.use_mtmc and use_bim_prior_predictor:
            raise ValueError(
                "use_mtmc and use_bim_prior_predictor are mutually exclusive. "
                "Enable MTMC for the new motion-conditioning path, or disable it to keep the original prior path."
            )

        self.use_bim_prior_predictor = bool(use_bim_prior_predictor or self.use_mtmc or self.use_dmse)
        self.dmse = None
        self.mtmc = None
        self.bim_prior_predictor = None
        if self.use_dmse:
            dmse_args = _extract_component_args(dmse, "dmse")
            dmse_args.setdefault("hidden_channels", mtmc_hidden_channels)
            dmse_args.setdefault("feat_channels", feat_channels)
            dmse_args.setdefault("img_channels", 3)
            dmse_args.setdefault("use_img_diff", mtmc_use_img_diff)
            dmse_args.setdefault("use_state", mtmc_use_last_state)
            self.dmse = DMSE(**dmse_args)
            self.bim_prior_predictor = self.dmse
        elif self.use_mtmc:
            mtmc_name = _component_name(mtmc, "mtmc")
            mtmc_args = {
                "hidden_channels": mtmc_hidden_channels,
                "feat_channels": feat_channels,
                "use_img_diff": mtmc_use_img_diff,
                "use_state": mtmc_use_last_state,
                "use_local_motion_context": mtmc_use_local_motion_context,
                "use_global_context": mtmc_use_global_context,
            }
            if mtmc_name in ("pair_global_memory_mtmc", "pair_conditioned_global_memory_mtmc"):
                mtmc_args.update(
                    _extract_component_args(
                        mtmc,
                        "pair_global_memory_mtmc",
                        accepted_names=("pair_conditioned_global_memory_mtmc",),
                    )
                )
                mtmc_args.setdefault(
                    "global_token_channels",
                    self.pair_global_memory_token_channels or 256,
                )
                mtmc_cls = PairGlobalMemoryMTMC
            else:
                mtmc_args.update(_extract_component_args(mtmc, "mtmc"))
                mtmc_cls = MTMC
            mtmc_args.setdefault("feat_channels", feat_channels)
            self.mtmc = mtmc_cls(**mtmc_args)
            self.bim_prior_predictor = self.mtmc
            if mtmc_pretrained and os.path.exists(mtmc_pretrained):
                mtmc_state = torch.load(mtmc_pretrained, map_location="cpu")
                if "state_dict" in mtmc_state:
                    mtmc_state = mtmc_state["state_dict"]
                self.mtmc.load_state_dict(mtmc_state, strict=False)
                print(f"[BiMVFI] Loaded MTMC from {mtmc_pretrained}")
            if freeze_mtmc:
                for param in self.mtmc.parameters():
                    param.requires_grad = False
        elif self.use_bim_prior_predictor:
            self.bim_prior_predictor = BiMPriorPredictor(
                hidden_channels=bim_prior_hidden_channels,
                feat_channels=feat_channels,
                use_img_diff=bim_prior_use_img_diff,
                use_state=bim_prior_use_last_state,
            )
            if bim_prior_pretrained and os.path.exists(bim_prior_pretrained):
                prior_state = torch.load(bim_prior_pretrained, map_location="cpu")
                if "state_dict" in prior_state:
                    prior_state = prior_state["state_dict"]
                self.bim_prior_predictor.load_state_dict(prior_state, strict=True)
                print(f"[BiMVFI] Loaded BiM prior predictor from {bim_prior_pretrained}")
            if freeze_bim_prior:
                for param in self.bim_prior_predictor.parameters():
                    param.requires_grad = False

    def _build_teacher_bim(self, flow_t0_res_tea, flow_t1_res_tea):
        eps = 1e-6
        flow_t0_r_tea = torch.norm(flow_t0_res_tea, dim=1, keepdim=True)
        flow_t1_r_tea = torch.norm(flow_t1_res_tea, dim=1, keepdim=True)
        denom = (flow_t0_r_tea * flow_t1_r_tea).clamp_min(eps)
        flow_sin_tea = (
            flow_t0_res_tea[:, 0:1] * flow_t1_res_tea[:, 1:2]
            - flow_t0_res_tea[:, 1:2] * flow_t1_res_tea[:, 0:1]
        ) / denom
        flow_cos_tea = (
            flow_t0_res_tea[:, 0:1] * flow_t1_res_tea[:, 0:1]
            + flow_t0_res_tea[:, 1:2] * flow_t1_res_tea[:, 1:2]
        ) / denom
        r = flow_t0_r_tea / (flow_t1_r_tea + flow_t0_r_tea + eps)
        phi = torch.cat([flow_cos_tea, flow_sin_tea], dim=1)
        phi = F.normalize(phi, dim=1, eps=1e-6)
        return r, phi

    def _build_uniform_bim(self, batch_size, height, width, device, dtype, time_step):
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

        r = time_step.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
        phi_angle = torch.ones((batch_size, 1, height, width), device=device, dtype=dtype) * torch.pi
        phi = torch.cat([torch.cos(phi_angle), torch.sin(phi_angle)], dim=1)
        return r, phi

    def _predict_bim_prior(
        self,
        img0,
        img1,
        time_step,
        target_hw,
        feat0_pyr=None,
        feat1_pyr=None,
        last_flow=None,
        last_occ=None,
        motion_context_pyr=None,
        global_context_pyr=None,
        global_motion_tokens=None,
    ):
        predictor_kwargs = dict(
            feat0_pyr=feat0_pyr,
            feat1_pyr=feat1_pyr,
            last_flow=last_flow,
            last_occ=last_occ,
            target_hw=target_hw,
        )
        if self.use_mtmc:
            predictor_kwargs["motion_context_pyr"] = motion_context_pyr
            predictor_kwargs["global_context_pyr"] = global_context_pyr
            if getattr(self.bim_prior_predictor, "accepts_global_motion_tokens", False):
                predictor_kwargs["global_motion_tokens"] = global_motion_tokens
        elif self.use_dmse:
            predictor_kwargs["motion_context_pyr"] = motion_context_pyr
            predictor_kwargs["global_motion_tokens"] = global_motion_tokens

        prior_output = self.bim_prior_predictor(
            img0,
            img1,
            time_step,
            **predictor_kwargs,
        )
        if isinstance(prior_output, dict):
            r = prior_output.get("rho", prior_output.get("r"))
            phi = prior_output.get("direction", prior_output.get("phi"))
            if r is None or phi is None:
                raise ValueError("Motion-state estimator output must contain rho/direction or r/phi.")
        else:
            r, phi = prior_output
        if r.shape[-2:] != target_hw:
            r = F.interpolate(r, size=target_hw, mode="bilinear", align_corners=False)
        if phi.shape[-2:] != target_hw:
            phi = F.interpolate(phi, size=target_hw, mode="bilinear", align_corners=False)
            phi = F.normalize(phi, dim=1, eps=1e-6)
        return r, phi

    @staticmethod
    def _pad_sequence_clip(padder, clip):
        if clip is None:
            return None
        batch_size, seq_len, channels, height, width = clip.shape
        clip = clip.reshape(batch_size * seq_len, channels, height, width)
        clip = padder.pad(clip)
        return clip.reshape(batch_size, seq_len, channels, clip.shape[-2], clip.shape[-1])

    @staticmethod
    def _resize_sequence_clip(clip, scale_factor):
        if clip is None or scale_factor == 1.0:
            return clip
        batch_size, seq_len, channels, height, width = clip.shape
        clip = clip.reshape(batch_size * seq_len, channels, height, width)
        clip = F.interpolate(
            input=clip,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return clip.reshape(batch_size, seq_len, channels, clip.shape[-2], clip.shape[-1])

    def _build_temporal_contexts(self, time_period, target_context_hw_pyr=None):
        temporal_context_input = self.temporal_context_input
        if temporal_context_input is None:
            return None, None, None, None, None, None, False, False, False, False

        route_local_to_mtmc = (
            self.use_mtmc
            and self.use_ltce
            and bool(self.local_fine_context_cfg.get("motion_to_mtmc", True))
        )
        route_detail_to_tcar = (
            self.use_tcar
            and self.use_ltce
            and bool(
                self.local_fine_context_cfg.get(
                    "detail_to_sn",
                    self.use_local_detail_context_for_sn,
                )
            )
        )
        route_global_to_mtmc = (
            self.use_mtmc
            and (self.use_gtme or self.use_pair_global_memory)
            and bool(self.global_memory_context_cfg.get("memory_to_mtmc", True))
        )
        route_global_to_tcar = (
            self.use_tcar
            and (self.use_gtme or self.use_pair_global_memory)
            and bool(
                self.global_memory_context_cfg.get(
                    "memory_to_sn",
                    self.use_global_memory_for_sn,
                )
            )
        )

        motion_context_pyr = None
        detail_context_pyr = None
        global_motion_context_pyr = None
        global_structure_context_pyr = None
        global_motion_tokens = None
        global_structure_tokens = None

        need_local_context = (
            (route_local_to_mtmc or route_detail_to_tcar)
            and temporal_context_input.get("local_clip") is not None
            and temporal_context_input.get("local_anchor_indices") is not None
        )
        if need_local_context:
            local_clip = temporal_context_input["local_clip"]
            batch_size, window_size, channels, height, width = local_clip.shape
            local_clip_flat = local_clip.reshape(batch_size * window_size, channels, height, width)
            local_mfeat_flat_pyr = self.mfe(local_clip_flat)
            local_cfeat_flat_pyr = self.cfe(local_clip_flat)
            motion_feat_seq_pyr = [
                feat.reshape(batch_size, window_size, *feat.shape[1:]) for feat in local_mfeat_flat_pyr
            ]
            detail_feat_seq_pyr = [
                feat.reshape(batch_size, window_size, *feat.shape[1:]) for feat in local_cfeat_flat_pyr
            ]
            local_context_dict = self.ltce(
                local_clip,
                motion_feat_seq_pyr,
                detail_feat_seq_pyr,
                time_period,
                frame_mask=temporal_context_input.get("local_frame_mask"),
                anchor_indices=temporal_context_input.get("local_anchor_indices"),
                frame_relative_positions=temporal_context_input.get("local_frame_relative_positions"),
                target_local_position=temporal_context_input.get("target_local_position"),
            )
            if route_local_to_mtmc:
                motion_context_pyr = local_context_dict["motion_context_pyr"]
            if route_detail_to_tcar:
                detail_context_pyr = local_context_dict["detail_context_pyr"]

        need_global_context = route_global_to_mtmc or route_global_to_tcar
        if need_global_context:
            if self.use_pair_global_memory:
                if self.pair_global_memory_output is None:
                    raise ValueError("Pair global memory is enabled, but no pair memory output was prepared.")
                mtmc_accepts_tokens = getattr(
                    self.bim_prior_predictor,
                    "accepts_global_motion_tokens",
                    False,
                )
                tcar_accepts_tokens = getattr(
                    self.sn,
                    "accepts_global_structure_tokens",
                    False,
                )
                need_pair_adapter = (
                    (route_global_to_mtmc and not mtmc_accepts_tokens)
                    or (route_global_to_tcar and not tcar_accepts_tokens)
                )
                if need_pair_adapter and target_context_hw_pyr is None:
                    raise ValueError("Pair global memory adapters require target_context_hw_pyr.")
                if route_global_to_mtmc:
                    pair_motion_tokens = self.pair_global_memory_output.get("C_motion")
                    if mtmc_accepts_tokens:
                        global_motion_tokens = pair_motion_tokens
                    else:
                        global_motion_context_pyr = self.pair_motion_context_adapter(
                            pair_motion_tokens,
                            target_context_hw_pyr,
                        )
                if route_global_to_tcar:
                    pair_structure_tokens = self.pair_global_memory_output.get(
                        "structure_condition",
                        self.pair_global_memory_output.get("C_structure"),
                    )
                    if tcar_accepts_tokens:
                        global_structure_tokens = pair_structure_tokens
                    else:
                        global_structure_context_pyr = self.pair_structure_context_adapter(
                            pair_structure_tokens,
                            target_context_hw_pyr,
                        )
            else:
                need_legacy_gtme = (
                    temporal_context_input.get("global_clip") is not None
                    and temporal_context_input.get("global_frame_positions") is not None
                    and temporal_context_input.get("target_global_position") is not None
                )
                if need_legacy_gtme:
                    global_context_pyr = self.gtme(
                        temporal_context_input["global_clip"],
                        temporal_context_input["global_frame_positions"],
                        temporal_context_input["target_global_position"],
                        global_frame_mask=temporal_context_input.get("global_frame_mask"),
                        global_anchor_indices=temporal_context_input.get("global_anchor_indices"),
                    )
                    if route_global_to_mtmc:
                        global_motion_context_pyr = global_context_pyr
                    if route_global_to_tcar:
                        global_structure_context_pyr = global_context_pyr

        return (
            motion_context_pyr,
            detail_context_pyr,
            global_motion_context_pyr,
            global_structure_context_pyr,
            global_motion_tokens,
            global_structure_tokens,
            route_local_to_mtmc,
            route_detail_to_tcar,
            route_global_to_mtmc,
            route_global_to_tcar,
        )

    def _build_adams_context(self, time_period):
        if not self.use_adams_encoder or self.adams_encoder is None:
            return None
        temporal_context_input = self.temporal_context_input
        if temporal_context_input is None:
            return None

        adams_clip = temporal_context_input.get("adams_clip")
        adams_anchor_indices = temporal_context_input.get("adams_anchor_indices")
        if adams_clip is None or adams_anchor_indices is None:
            return None

        adams_tau_list = temporal_context_input.get("adams_tau_list")
        if adams_tau_list is None:
            adams_tau_list = time_period

        return self.adams_encoder(
            adams_clip,
            anchor_indices=adams_anchor_indices,
            tau_list=adams_tau_list,
            frame_mask=temporal_context_input.get("adams_frame_mask"),
        )

    def forward_one_lvl(self, img0, img1, last_flow, last_occ, teacher_input_dict=None, time_period=0.5):
        teacher_dict = dict()
        ### Keep the original BiM-VFI motion chain pair-centric.
        adams_out = self._build_adams_context(time_period)
        self.adams_output = adams_out
        if adams_out is not None:
            feat0_pyr = adams_out["anchor0_dynamic_pyr"]
            feat1_pyr = adams_out["anchor1_dynamic_pyr"]
        else:
            feat0_pyr = self.mfe(img0)
            feat1_pyr = self.mfe(img1)

        video_context_pyr = None
        (
            motion_context_pyr,
            detail_context_pyr,
            global_motion_context_pyr,
            global_structure_context_pyr,
            global_motion_tokens,
            global_structure_tokens,
            route_local_to_mtmc,
            route_detail_to_tcar,
            route_global_to_mtmc,
            route_global_to_tcar,
        ) = self._build_temporal_contexts(
            time_period,
            target_context_hw_pyr=[feat.shape[-2:] for feat in feat0_pyr[:3]],
        )
        if adams_out is not None:
            cfeat0_pyr = adams_out["anchor0_anatomical_pyr"]
            cfeat1_pyr = adams_out["anchor1_anatomical_pyr"]
        else:
            cfeat0_pyr = self.cfe(img0)
            cfeat1_pyr = self.cfe(img1)
        if self.use_video_sequence_context and self.video_sequence_input is not None and self.video_context_extractor is not None:
            support_clip = self.video_sequence_input["clip"]
            support_frame_mask = self.video_sequence_input["frame_mask"]
            support_anchor_indices = self.video_sequence_input["anchor_indices"]
            batch_size, window_size, channels, height, width = support_clip.shape
            support_clip_flat = support_clip.reshape(batch_size * window_size, channels, height, width)
            support_cfeat_flat_pyr = self.cfe(support_clip_flat)
            support_cfeat_seq_pyr = [
                feat.reshape(batch_size, window_size, *feat.shape[1:]) for feat in support_cfeat_flat_pyr
            ]

            batch_index = torch.arange(batch_size, device=support_clip.device)
            if adams_out is None:
                cfeat0_pyr = [feat_seq[batch_index, support_anchor_indices[:, 0]] for feat_seq in support_cfeat_seq_pyr]
                cfeat1_pyr = [feat_seq[batch_index, support_anchor_indices[:, 1]] for feat_seq in support_cfeat_seq_pyr]
            video_context_pyr = self.video_context_extractor(
                support_clip,
                support_cfeat_seq_pyr,
                time_period,
                frame_mask=self.video_sequence_input["frame_mask"],
                anchor_indices=self.video_sequence_input["anchor_indices"],
            )
        sn_kwargs = {}
        if self.sn_accepts_video_context:
            sn_kwargs["video_context_pyr"] = video_context_pyr
            if self.video_sequence_input is not None:
                sn_kwargs["video_frames"] = self.video_sequence_input["clip"]
                sn_kwargs["video_frame_mask"] = self.video_sequence_input["frame_mask"]
                sn_kwargs["video_anchor_indices"] = self.video_sequence_input["anchor_indices"]
            sn_kwargs["time_step"] = time_period
        if self.use_tcar:
            if route_detail_to_tcar and detail_context_pyr is not None:
                sn_kwargs["detail_context_pyr"] = detail_context_pyr
            if route_global_to_tcar and global_structure_context_pyr is not None:
                sn_kwargs["global_context_pyr"] = global_structure_context_pyr
            if (
                route_global_to_tcar
                and global_structure_tokens is not None
                and getattr(self.sn, "accepts_global_structure_tokens", False)
            ):
                sn_kwargs["global_structure_tokens"] = global_structure_tokens
        if self.use_asd and adams_out is not None:
            sn_kwargs["anatomical_context_pyr"] = adams_out["anatomical_context_pyr"]
            sn_kwargs["tau_index"] = 0

        B, _, H, W = feat0_pyr[-1].shape
        if self.use_dmse and adams_out is not None:
            bim_motion_context_pyr = adams_out["dynamic_context_pyr"]
            bim_global_motion_tokens = adams_out["global_dynamic_tokens_pyr"]
        else:
            bim_motion_context_pyr = motion_context_pyr if route_local_to_mtmc else None
            bim_global_motion_tokens = global_motion_tokens if route_global_to_mtmc else None
        last_flow_down = F.interpolate(
            input=last_flow.detach().clone(), scale_factor=0.5,
            mode="bilinear", align_corners=False) * 0.5
        last_occ_down = F.interpolate(
            input=last_occ.detach().clone(), scale_factor=0.5,
            mode="bilinear", align_corners=False)

        use_teacher_branch = (
            self.use_teacher_branch
            and teacher_input_dict is not None
            and bool(teacher_input_dict.get("use_teacher_branch", True))
            and 'imgt_this_lvl' in teacher_input_dict
        )
        if use_teacher_branch:
            ### If it is training, do KDVCF
            featt_pyr = self.mfe(teacher_input_dict['imgt_this_lvl'])

            ### Prepare M_t->0t and M_t->t1
            r_0 = torch.zeros(B, 1, H, W, device=feat0_pyr[-1].device)
            r_1 = torch.ones(B, 1, H, W, device=feat0_pyr[-1].device)
            # phi_tea0 = torch.rand(B, 1, H, W, device=feat0_pyr[-1].device) * torch.pi * 2
            # phi_tea0 = torch.cat([torch.cos(phi_tea0), torch.sin(phi_tea0)], dim=1)
            # phi_tea1 = torch.rand(B, 1, H, W, device=feat0_pyr[-1].device) * torch.pi * 2
            # phi_tea1 = torch.cat([torch.cos(phi_tea1), torch.sin(phi_tea1)], dim=1)
            # Keep teacher BiM supervision deterministic with a fixed flat angle.
            phi_tea_angle = feat0_pyr[-1].new_full((B, 1, H, W), torch.pi)
            phi_tea = torch.cat([torch.cos(phi_tea_angle), torch.sin(phi_tea_angle)], dim=1)

            ### Prepare flows and occlusion masks for teacher process
            last_flow_0t = torch.cat((last_flow_down[:, :2], torch.zeros_like(last_flow_down[:, :2])), dim=1)
            last_flow_t1 = torch.cat((torch.zeros_like(last_flow_down[:, 2:]), last_flow_down[:, 2:]), dim=1)

            ### Get teacher flows V_t->t|0t and V_t->0
            flow_0t_low, flow_0t_res = self.bimfn(
                feat0_pyr[-1], featt_pyr[-1], r_1, phi_tea, last_flow_0t, last_occ_down
            )
            ### Get teacher flows V_t->t|t1 and V_t->1
            flow_t1_low, flow_1t_res = self.bimfn(
                featt_pyr[-1], feat1_pyr[-1], r_0, phi_tea, last_flow_t1, last_occ_down
            )

            ### Calculate BiM of student process
            flow_t0_res_tea = (flow_0t_res[:, :2]).detach().clone()
            flow_t1_res_tea = (flow_1t_res[:, 2:]).detach().clone()
            flow_tea_low = torch.cat([flow_0t_low[:, :2], flow_t1_low[:, 2:]], dim=1)

            ### Upsample flows of teacher process
            bi_flow_tea_pyr, occ_tea = self.caun(flow_tea_low, cfeat0_pyr, cfeat1_pyr, last_occ_down)
            flow_t0_tea = bi_flow_tea_pyr[0][:, :2]
            flow_t1_tea = bi_flow_tea_pyr[0][:, 2:]

            ### Interpolate image at current level for teacher process
            interp_img_tea, occ_tea, teacher_extra_dict = self.sn(
                img0, img1, cfeat0_pyr, cfeat1_pyr, bi_flow_tea_pyr, occ_tea,
                **sn_kwargs,
            )
            teacher_dict['flow_t0_tea'] = flow_t0_tea
            teacher_dict['flow_t0_res_tea'] = flow_t0_res_tea
            teacher_dict['flow_t1_tea'] = flow_t1_tea
            teacher_dict['flow_t1_res_tea'] = flow_t1_res_tea
            teacher_dict['interp_img_tea'] = interp_img_tea
            teacher_dict['flow_0t_res'] = flow_0t_res
            teacher_dict['flow_t1_res'] = flow_1t_res
            r_gt, phi_gt = self._build_teacher_bim(flow_t0_res_tea, flow_t1_res_tea)
            teacher_dict['bim_r_gt'] = r_gt.detach().clone()
            teacher_dict['bim_phi_gt'] = phi_gt.detach().clone()
            if self.use_bim_prior_predictor:
                r, phi = self._predict_bim_prior(
                    img0, img1, time_period, target_hw=(H, W),
                    feat0_pyr=feat0_pyr, feat1_pyr=feat1_pyr,
                    last_flow=last_flow_down, last_occ=last_occ_down,
                    motion_context_pyr=bim_motion_context_pyr,
                    global_context_pyr=global_motion_context_pyr if route_global_to_mtmc else None,
                    global_motion_tokens=bim_global_motion_tokens,
                )
                teacher_dict['bim_r_pred'] = r
                teacher_dict['bim_phi_pred'] = phi
            else:
                r, phi = r_gt.detach().clone(), phi_gt.detach().clone()
        else:
            if self.use_bim_prior_predictor:
                r, phi = self._predict_bim_prior(
                    img0, img1, time_period, target_hw=(H, W),
                    feat0_pyr=feat0_pyr, feat1_pyr=feat1_pyr,
                    last_flow=last_flow_down, last_occ=last_occ_down,
                    motion_context_pyr=bim_motion_context_pyr,
                    global_context_pyr=global_motion_context_pyr if route_global_to_mtmc else None,
                    global_motion_tokens=bim_global_motion_tokens,
                )
            else:
                r, phi = self._build_uniform_bim(
                    B, H, W, feat0_pyr[-1].device, feat0_pyr[-1].dtype, time_period
                )

        # print("1111111111111111111", r, r.mean())

        # print("2222222222222222222", phi, phi[:, 0].mean().item(), phi[:, 1].mean().item())
        ### Get student flows V_t->0 and V_t->1
        flow_low, flow_res = self.bimfn(
            feat0_pyr[-1], feat1_pyr[-1], r, phi, last_flow_down, last_occ_down)

        ### Upsample student flows
        bi_flow_pyr, occ = self.caun(flow_low, cfeat0_pyr, cfeat1_pyr, last_occ_down)
        flow = bi_flow_pyr[0]

        ### Interpolate image at current level for student process
        interp_img, occ, extra_dict = self.sn(
            img0, img1, cfeat0_pyr, cfeat1_pyr, bi_flow_pyr, occ,
            **sn_kwargs,
        )
        if motion_context_pyr is not None:
            extra_dict["motion_context_pyr"] = motion_context_pyr
        if detail_context_pyr is not None:
            extra_dict["detail_context_pyr"] = detail_context_pyr
        if global_motion_context_pyr is not None:
            extra_dict["global_motion_context_pyr"] = global_motion_context_pyr
        if global_structure_context_pyr is not None:
            extra_dict["global_structure_context_pyr"] = global_structure_context_pyr
        if global_motion_tokens is not None:
            extra_dict["global_motion_tokens"] = global_motion_tokens
        if global_structure_tokens is not None:
            extra_dict["global_structure_tokens"] = global_structure_tokens
        if adams_out is not None:
            extra_dict["adams_dynamic_context_pyr"] = adams_out["dynamic_context_pyr"]
            extra_dict["adams_anatomical_context_pyr"] = adams_out["anatomical_context_pyr"]
            extra_dict["adams_global_dynamic_tokens_pyr"] = adams_out["global_dynamic_tokens_pyr"]
            extra_dict["adams_global_anatomical_tokens_pyr"] = adams_out["global_anatomical_tokens_pyr"]
        extra_dict.update({'flow_res': flow_res})
        return flow, occ, interp_img, extra_dict, teacher_dict

    def forward(self, img0, img1, time_step,
                pyr_level=None, imgt=None, run_with_gt=False, **kwargs):
        if pyr_level is None: pyr_level = self.pyr_level
        clip = kwargs.get("clip", None)
        frame_mask = kwargs.get("frame_mask", None)
        anchor_indices = kwargs.get("anchor_indices", None)
        local_clip = kwargs.get("local_clip", clip)
        local_frame_mask = kwargs.get("local_frame_mask", frame_mask)
        local_anchor_indices = kwargs.get("local_anchor_indices", anchor_indices)
        local_frame_relative_positions = kwargs.get("local_frame_relative_positions", None)
        target_local_position = kwargs.get("target_local_position", None)
        global_clip = kwargs.get("global_clip", kwargs.get("sparse_sequence_clip", None))
        global_frame_mask = kwargs.get("global_frame_mask", kwargs.get("sparse_sequence_frame_mask", None))
        global_frame_positions = kwargs.get(
            "global_frame_positions",
            kwargs.get("sparse_sequence_frame_positions", None),
        )
        global_anchor_indices = kwargs.get(
            "global_anchor_indices",
            kwargs.get("sparse_sequence_anchor_indices", None),
        )
        adams_clip = kwargs.get("global_clip", kwargs.get("sparse_sequence_clip", None))
        adams_frame_mask = kwargs.get("global_frame_mask", kwargs.get("sparse_sequence_frame_mask", None))
        adams_anchor_indices = kwargs.get(
            "global_anchor_indices",
            kwargs.get("sparse_sequence_anchor_indices", None),
        )
        adams_tau_list = kwargs.get("target_local_position", None)
        target_global_position = kwargs.get("target_global_position", None)
        use_video_sequence = (
            self.use_video_sequence_context
            and clip is not None
            and frame_mask is not None
            and anchor_indices is not None
        )
        use_local_temporal_context = (
            self.use_ltce
            and local_clip is not None
            and local_anchor_indices is not None
        )
        use_global_temporal_context = (
            (self.use_gtme or self.use_pair_global_memory)
            and global_clip is not None
            and global_frame_positions is not None
            and target_global_position is not None
        )
        use_adams_temporal_context = (
            self.use_adams_encoder
            and adams_clip is not None
            and adams_anchor_indices is not None
        )
        if self.use_adams_encoder and not use_adams_temporal_context:
            raise ValueError(
                "use_adams_encoder=True requires global_clip or sparse_sequence_clip, "
                "plus global_anchor_indices or sparse_sequence_anchor_indices in the batch."
            )
        pair_global_clip = global_clip if (self.use_pair_global_memory and use_global_temporal_context) else None
        use_teacher_branch = self.use_teacher_branch and (self.training or run_with_gt)
        N, _, H, W = img0.shape
        if adams_tau_list is None:
            adams_tau_list = time_step.reshape(N, 1)
        flowt0_pred_list = []
        flowt0_res_list = []
        flowt1_pred_list = []
        flowt1_res_list = []
        flow0t_tea_list = []
        flowt1_tea_list = []
        flowt0_pred_tea_list = []
        flowt0_res_tea_list = []
        flowt1_pred_tea_list = []
        flowt1_res_tea_list = []
        refine_mask_tea_list = []
        interp_imgs = []
        interp_imgs_tea = []
        bim_r_gt_list = []
        bim_phi_gt_list = []
        bim_r_pred_list = []
        bim_phi_pred_list = []

        padder = InputPadder(
            img0.shape,
            divisor=int(2 ** (pyr_level - 1 + getattr(self.sn, "num_downsamples", 2)))
        )  # pyr_level=3

        ### Normalize input images
        with torch.set_grad_enabled(False):
            tenStats = [img0, img1]
            if self.training or run_with_gt:
                tenStats.append(imgt)
            tenMean_ = sum([tenIn.mean([1, 2, 3], True) for tenIn in tenStats]) / len(tenStats)
            tenStd_ = (sum([tenIn.std([1, 2, 3], False, True).square() + (
                    tenMean_ - tenIn.mean([1, 2, 3], True)).square() for tenIn in tenStats]) / len(tenStats)).sqrt()

            img0 = (img0 - tenMean_) / (tenStd_ + 0.0000001)
            img1 = (img1 - tenMean_) / (tenStd_ + 0.0000001)
            if self.training or run_with_gt:
                imgt = (imgt - tenMean_) / (tenStd_ + 0.0000001)
            if use_video_sequence:
                clip = (clip - tenMean_.unsqueeze(1)) / (tenStd_.unsqueeze(1) + 0.0000001)
            if use_local_temporal_context:
                local_clip = (local_clip - tenMean_.unsqueeze(1)) / (tenStd_.unsqueeze(1) + 0.0000001)
            if use_adams_temporal_context:
                adams_clip = (adams_clip - tenMean_.unsqueeze(1)) / (tenStd_.unsqueeze(1) + 0.0000001)
            if use_global_temporal_context and not self.use_pair_global_memory:
                global_clip = (global_clip - tenMean_.unsqueeze(1)) / (tenStd_.unsqueeze(1) + 0.0000001)

        ### Pad images for downsampling
        img0, img1 = padder.pad(img0, img1)
        if self.training or run_with_gt:
            imgt = padder.pad(imgt)
        if use_video_sequence:
            clip = self._pad_sequence_clip(padder, clip)
        if use_local_temporal_context:
            local_clip = self._pad_sequence_clip(padder, local_clip)
        if use_adams_temporal_context:
            adams_clip = self._pad_sequence_clip(padder, adams_clip)
        if use_global_temporal_context and not self.use_pair_global_memory:
            global_clip = self._pad_sequence_clip(padder, global_clip)

        N, _, H, W = img0.shape
        teacher_input_dict = {"use_teacher_branch": use_teacher_branch}
        self.pair_global_memory_output = None
        if self.use_pair_global_memory and use_global_temporal_context:
            pair_k = global_anchor_indices[:, 0]
            pair_tau = time_step.reshape(N)
            self.pair_global_memory_output = self.pair_global_memory(
                pair_global_clip,
                pair_k,
                pair_tau,
                frame_mask=global_frame_mask,
                frame_positions=global_frame_positions,
                anchor_indices=global_anchor_indices,
            )

        for level in list(range(pyr_level))[::-1]:
            ### Downsample images if needed
            if level != 0:
                scale_factor = 1 / 2 ** level
                img0_this_lvl = F.interpolate(
                    input=img0, scale_factor=scale_factor,
                    mode="bilinear", align_corners=False, antialias=True)
                img1_this_lvl = F.interpolate(
                    input=img1, scale_factor=scale_factor,
                    mode="bilinear", align_corners=False, antialias=True)
                if use_teacher_branch:
                    imgt_this_lvl = F.interpolate(
                        input=imgt, scale_factor=scale_factor,
                        mode="bilinear", align_corners=False, antialias=True)
                    teacher_input_dict['imgt_this_lvl'] = imgt_this_lvl
            else:
                img0_this_lvl = img0
                img1_this_lvl = img1
                if use_teacher_branch:
                    imgt_this_lvl = imgt
                    teacher_input_dict['imgt_this_lvl'] = imgt_this_lvl

            if use_video_sequence:
                if level != 0:
                    B_clip, T_clip, C_clip, H_clip, W_clip = clip.shape
                    clip_this_lvl = clip.reshape(B_clip * T_clip, C_clip, H_clip, W_clip)
                    clip_this_lvl = F.interpolate(
                        input=clip_this_lvl, scale_factor=scale_factor,
                        mode="bilinear", align_corners=False, antialias=True)
                    clip_this_lvl = clip_this_lvl.reshape(
                        B_clip, T_clip, C_clip, clip_this_lvl.shape[-2], clip_this_lvl.shape[-1])
                else:
                    clip_this_lvl = clip
                self.video_sequence_input = {
                    "clip": clip_this_lvl,
                    "frame_mask": frame_mask,
                    "anchor_indices": anchor_indices,
                }
            else:
                self.video_sequence_input = None
            if use_local_temporal_context or use_global_temporal_context or use_adams_temporal_context:
                scale_factor = 1.0 if level == 0 else 1 / 2 ** level
                self.temporal_context_input = {}
                if use_adams_temporal_context:
                    self.temporal_context_input.update(
                        {
                            "adams_clip": self._resize_sequence_clip(adams_clip, scale_factor),
                            "adams_frame_mask": adams_frame_mask,
                            "adams_anchor_indices": adams_anchor_indices,
                            "adams_tau_list": adams_tau_list,
                        }
                    )
                if use_local_temporal_context:
                    self.temporal_context_input.update(
                        {
                            "local_clip": self._resize_sequence_clip(local_clip, scale_factor),
                            "local_frame_mask": local_frame_mask,
                            "local_anchor_indices": local_anchor_indices,
                            "local_frame_relative_positions": local_frame_relative_positions,
                            "target_local_position": target_local_position,
                        }
                    )
                if use_global_temporal_context:
                    if self.use_pair_global_memory:
                        self.temporal_context_input.update(
                            {
                                "global_frame_mask": global_frame_mask,
                                "global_frame_positions": global_frame_positions,
                                "global_anchor_indices": global_anchor_indices,
                                "target_global_position": target_global_position,
                            }
                        )
                    else:
                        self.temporal_context_input.update(
                            {
                                "global_clip": self._resize_sequence_clip(global_clip, scale_factor),
                                "global_frame_mask": global_frame_mask,
                                "global_frame_positions": global_frame_positions,
                                "global_anchor_indices": global_anchor_indices,
                                "target_global_position": target_global_position,
                            }
                        )
            else:
                self.temporal_context_input = None

            ### Initialize zero flows for lowest pyramid level
            if level == pyr_level - 1:
                last_flow = torch.zeros(
                    (N, 4, H // (2 ** (level + 1)), W // (2 ** (level + 1))), device=img0.device
                )
                last_occ = torch.zeros(N, 1, H // (2 ** (level + 1)), W // (2 ** (level + 1)), device=img0.device)
            else:
                last_flow = flow
                last_occ = occ

            ### Single pyramid level run
            flow, occ, interp_img, extra_dict, teacher_dict = self.forward_one_lvl(
                img0_this_lvl, img1_this_lvl, last_flow, last_occ, teacher_input_dict, time_step)

            flowt0_pred_list.append((flow[:, :2]))
            flowt1_pred_list.append((flow[:, 2:]))
            flowt0_res_list.append(extra_dict['flow_res'][:, :2])
            flowt1_res_list.append(extra_dict['flow_res'][:, 2:])
            interp_imgs.append((interp_img) * (tenStd_ + 0.0000001) + tenMean_)
            if use_teacher_branch:
                flowt0_pred_tea_list.append(
                    (teacher_dict['flow_t0_tea']))
                flowt1_pred_tea_list.append(
                    (teacher_dict['flow_t1_tea']))
                flowt0_res_tea_list.append(teacher_dict['flow_t0_res_tea'])
                flowt1_res_tea_list.append(teacher_dict['flow_t1_res_tea'])
                interp_imgs_tea.append((teacher_dict['interp_img_tea']) * (tenStd_ + 0.0000001) + tenMean_)
                flow0t_tea_list.append(teacher_dict['flow_0t_res'][:, 2:])
                flowt1_tea_list.append(teacher_dict['flow_t1_res'][:, :2])
                if 'bim_r_gt' in teacher_dict:
                    bim_r_gt_list.append(teacher_dict['bim_r_gt'])
                    bim_phi_gt_list.append(teacher_dict['bim_phi_gt'])
                if 'bim_r_pred' in teacher_dict:
                    bim_r_pred_list.append(teacher_dict['bim_r_pred'])
                    bim_phi_pred_list.append(teacher_dict['bim_phi_pred'])

        self.video_sequence_input = None
        self.temporal_context_input = None
        self.pair_global_memory_output = None
        self.adams_output = None

        result_dict = {
            "imgt_preds": interp_imgs, "flowt0_pred_list": flowt0_pred_list[::-1],
            "flowt1_pred_list": flowt1_pred_list[::-1],
            'imgt_pred': padder.unpad(interp_imgs[-1].contiguous()),
            'flowt0_pred_tea_list': flowt0_pred_tea_list[::-1], 'flowt1_pred_tea_list': flowt1_pred_tea_list[::-1],
            'interp_imgs_tea': interp_imgs_tea, 'refine_mask_tea': refine_mask_tea_list,
            'flowt0_res_list': flowt0_res_list[::-1], 'flowt1_res_list': flowt1_res_list[::-1],
            'flowt0_res_tea_list': flowt0_res_tea_list[::-1], 'flowt1_res_tea_list': flowt1_res_tea_list[::-1],
            'flow0t_tea_list': flow0t_tea_list[::-1], 'flowt1_tea_list': flowt1_tea_list[::-1],
            'bim_r_gt_list': bim_r_gt_list[::-1], 'bim_phi_gt_list': bim_phi_gt_list[::-1],
            'bim_r_pred_list': bim_r_pred_list[::-1], 'bim_phi_pred_list': bim_phi_pred_list[::-1],
        }
        if (self.use_tcar or self.use_asd) and "warped_img0" in extra_dict and "warped_img1" in extra_dict:
            warped_img0 = extra_dict["warped_img0"] * (tenStd_ + 0.0000001) + tenMean_
            warped_img1 = extra_dict["warped_img1"] * (tenStd_ + 0.0000001) + tenMean_
            result_dict["warped_img0"] = padder.unpad(warped_img0.contiguous())
            result_dict["warped_img1"] = padder.unpad(warped_img1.contiguous())
            if "refine_res" in extra_dict:
                refine_res = extra_dict["refine_res"] * (tenStd_ + 0.0000001)
                result_dict["refine_res"] = padder.unpad(refine_res.contiguous())

        return result_dict
