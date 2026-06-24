from modules.models.base_model import BaseModel
import json
import math
import os
import time
from typing import Iterable
from pathlib import Path
import wandb

import torch
from torchvision.utils import save_image

import utils.misc
import utils.misc as misc
from utils.plot import plot_samples_per_epoch
from utils.metrics import (
    calculate_batch_psnr,
    calculate_batch_ssim,
    calculate_batch_lpips,
    calculate_batch_stlpips,
    calculate_batch_niqe,
    calculate_batch_vessel_psnr,
)
from utils.flowvis import flow2img
from modules.models import make, register


@register('bim_vfi')
class BiMVFI(BaseModel):
    def __init__(self, cfg):
        super(BiMVFI, self).__init__(cfg)

    def _get_eval_metric_cfg(self):
        metric_cfg = self.cfgs.get("eval_metrics", self.cfgs.get("validation_metrics", {}))
        return metric_cfg if isinstance(metric_cfg, dict) else {}

    def _get_std_metric_names(self, metric_cfg):
        std_cfg = metric_cfg.get("std", {})
        record_std = bool(metric_cfg.get("record_std", False))
        if isinstance(std_cfg, dict):
            record_std = bool(std_cfg.get("enabled", record_std))
            std_names = std_cfg.get("metrics", metric_cfg.get("std_metrics", None))
        else:
            record_std = bool(std_cfg or record_std)
            std_names = metric_cfg.get("std_metrics", None)

        if std_names is None:
            std_names = ["psnr", "ssim", "lpips", "vessel_psnr"]
        if isinstance(std_names, str):
            std_names = [std_names]
        return record_std, set(str(name) for name in std_names)

    def _get_vessel_psnr_cfg(self, metric_cfg):
        vessel_cfg = metric_cfg.get("vessel_psnr", metric_cfg.get("vessel", {}))
        vessel_cfg = vessel_cfg if isinstance(vessel_cfg, dict) else {}
        return vessel_cfg, bool(vessel_cfg.get("enabled", False))

    @staticmethod
    def _metric_update_payload(value, n, values=None, use_exact_std=False):
        if use_exact_std and values is not None:
            finite_values = []
            for item in values:
                item = float(item)
                if math.isfinite(item):
                    finite_values.append(item)
            if finite_values:
                return {
                    "value": sum(finite_values) / len(finite_values),
                    "n": len(finite_values),
                    "sum_sq": sum(item * item for item in finite_values),
                }
        return {"value": float(value), "n": int(n)}

    def _sync_validation_metrics_by_file(self, test_indicator):
        world_size = misc.get_world_size()
        if world_size <= 1:
            return

        rank = misc.get_rank()
        run_id = (
            os.environ.get("BATCH_JOB_ID")
            or os.environ.get("SLURM_JOB_ID")
            or os.environ.get("TORCHELASTIC_RUN_ID")
            or os.environ.get("MASTER_PORT")
            or self.cfgs.get("env", {}).get("port", "default")
        )
        safe_indicator = str(test_indicator).replace("\\", "_").replace("/", "_")
        sync_dir = (
            Path(self.cfgs["output_dir"])
            / "dist_validate_metrics"
            / f"run_{run_id}_{safe_indicator}_epoch{self.current_epoch:06d}_iter{self.current_iteration:09d}"
        )
        sync_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "rank": rank,
            "world_size": world_size,
            "meters": {
                name: {"count": meter.count, "total": meter.total, "total_sq": meter.total_sq}
                for name, meter in self.metric_logger.meters.items()
            },
        }
        rank_file = sync_dir / f"rank_{rank}.json"
        tmp_file = sync_dir / f"rank_{rank}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_file, rank_file)

        expected_files = [sync_dir / f"rank_{idx}.json" for idx in range(world_size)]
        timeout = float(self.cfgs.get("validation_file_sync_timeout_seconds", 86400))
        start_time = time.time()
        while not all(path.exists() for path in expected_files):
            if time.time() - start_time > timeout:
                missing = [str(path.name) for path in expected_files if not path.exists()]
                raise TimeoutError(
                    f"Timed out waiting for validation metric files in {sync_dir}: {missing}"
                )
            time.sleep(5)

        totals = {
            name: {"count": 0, "total": 0.0, "total_sq": 0.0}
            for name in self.metric_logger.meters.keys()
        }
        for path in expected_files:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, stats in data.get("meters", {}).items():
                if name not in totals:
                    continue
                totals[name]["count"] += int(stats["count"])
                totals[name]["total"] += float(stats["total"])
                totals[name]["total_sq"] += float(stats.get("total_sq", 0.0))

        for name, stats in totals.items():
            self.metric_logger.meters[name].count = stats["count"]
            self.metric_logger.meters[name].total = stats["total"]
            self.metric_logger.meters[name].total_sq = stats["total_sq"]

    def train_one_epoch(self, train_loader: Iterable, epoch: int, max_norm: float = 0):
        """
        Training of one epoch
        """
        self.current_epoch = epoch

        print("Epoch {} training starts:".format(epoch))  # debug用的

        self._reset_metric()

        self.model.train()

        header = 'Epoch: [{}]'.format(epoch)
        print("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        print_freq = 100
        for input_dict in self.metric_logger.log_every(train_loader, print_freq, header):
            input_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}

            # ### 测试内容 ###
            # print("这是第{}个epoch的input_dict————————————".format(epoch), input_dict)
            # break

            loss, losses, result_dict = self.train_one_step_opt(input_dict)
            self.lr_scheduler.step()
            imgt_pred = result_dict['imgt_pred']
            imgt_pred = torch.clamp(imgt_pred, 0, 1)

            self.metric_logger.update(loss=loss, **losses)
            self.metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            # 这里是debug用的
            if self.current_iteration == 0:
                print("len(train_loader) =", len(train_loader))

            if misc.is_main_process() and self.current_iteration % print_freq == 0:
                nsample = 4
                patch_size = self.cfgs['train_dataset']['args']['patch_size']

                img0_p, img1_p = input_dict['img0'][:nsample].detach(), input_dict['img1'][:nsample].detach()
                gt_p, imgt_pred_p = input_dict['imgt'][:nsample].detach(), imgt_pred[:nsample].detach()
                overlapped_img = img0_p * 0.5 + img1_p * 0.5

                if self.cfgs.get('enable_wandb', False):
                    wandb.log({"loss": loss}, step=self.current_iteration)
                    for k, v in losses.items():
                        wandb.log({f'loss_{k}': v}, step=self.current_iteration)
                    self._log_learning_rates(self.current_iteration)
                    if self.current_iteration % (print_freq * 10) == 0:
                        flowfwd = flow2img(result_dict['flowt0_pred_list'][0][:nsample].detach())

                        figure = torch.stack(
                            [overlapped_img, imgt_pred_p, flowfwd, gt_p])
                        figure = torch.transpose(figure, 0, 1).reshape(-1, 3, patch_size, patch_size)
                        image = plot_samples_per_epoch(
                            figure, os.path.join(self.cfgs['output_dir'], "imgs_train"),
                            self.current_epoch, self.current_iteration, nsample, 'train'
                        )
                        wandb.log({"Image": wandb.Image(image, file_type="jpg")}, step=self.current_iteration)
                else:
                    self.summary_writer.add_scalar("Train/loss", loss, self.current_iteration)
                    for k, v in losses.items():
                        self.summary_writer.add_scalar(f'Train/loss_{k}', v, self.current_iteration)
                    self._log_learning_rates(self.current_iteration)
                    if self.current_iteration % (print_freq * 10) == 0:
                        flowfwd = flow2img(result_dict['flowt0_pred_list'][0][:nsample].detach())

                        figure = torch.stack(
                            [overlapped_img, imgt_pred_p, flowfwd, gt_p])
                        figure = torch.transpose(figure, 0, 1).reshape(-1, 3, patch_size, patch_size)
                        image = plot_samples_per_epoch(
                            figure, os.path.join(self.cfgs['output_dir'], "imgs_train"),
                            self.current_epoch, self.current_iteration, nsample, 'train'
                        )
                        self.summary_writer.add_image("Train/image", image, self.current_iteration)

            self.current_iteration += 1

            # if self.current_iteration == 0:
            # break  # 这个如果break了，那么就是一个epoch只train一个batch，用于测试；注释掉break就可以正常训练

            # gather the stats from all processes
        self.metric_logger.synchronize_between_processes()
        self.current_epoch += 1
        if utils.misc.is_main_process():
            self.logger.info(f"Averaged training stats: {self.metric_logger}")

    @torch.no_grad()
    def validate(self, val_loader, test_indicator=None):
        """
        Validation step for each mini-batch
        """
        self.model.eval()

        self.metric_logger = misc.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('psnr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.metric_logger.add_meter('ssim', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.metric_logger.add_meter('lpips', misc.SmoothedValue(window_size=1))
        self.metric_logger.add_meter('stlpips', misc.SmoothedValue(window_size=1))
        self.metric_logger.add_meter('niqe', misc.SmoothedValue(window_size=1))
        metric_cfg = self._get_eval_metric_cfg()
        record_std, std_metric_names = self._get_std_metric_names(metric_cfg)
        vessel_psnr_cfg, vessel_psnr_enabled = self._get_vessel_psnr_cfg(metric_cfg)
        if vessel_psnr_enabled:
            self.metric_logger.add_meter('vessel_psnr', misc.SmoothedValue(window_size=1))
        header = 'Test:'
        psnr_dict = {}
        lpips_dict = {}
        stlpips_dict = {}

        print_freq = 10
        if test_indicator is None:
            test_indicator = f"{self.cfgs['test_dataset']['name']}_{self.cfgs['test_dataset']['args']['split']}"

        use_tcar = bool(getattr(self.model_without_ddp, "use_tcar", False))
        test_save_imgs = bool(self.cfgs['test_dataset'].get('save_imgs', False))
        should_save_imgs = test_save_imgs
        for input_dict in self.metric_logger.log_every(val_loader, print_freq, header):
            input_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
            img0 = input_dict['img0']
            imgt = input_dict['imgt']
            img1 = input_dict['img1']
            result_dict = self.model(**input_dict, run_with_gt=False)  ############################################################ bim测试：打开后，会使用gt计算真实bim的值；True

            scene_names = input_dict['scene_names']
            save_subdirs = input_dict.get('save_subdir')
            save_stems = input_dict.get('save_stem')
            if save_subdirs is None and all(
                key in input_dict
                for key in ("sample_name", "phase_index", "target_frame_index", "target_offset")
            ):
                save_subdirs = [
                    os.path.join(sample_name, f"phase_{int(phase_index):02d}")
                    for sample_name, phase_index in zip(
                        input_dict["sample_name"], input_dict["phase_index"].detach().cpu().tolist()
                    )
                ]
                save_stems = [
                    f"t{int(target_frame):04d}_off{int(target_offset):02d}"
                    for target_frame, target_offset in zip(
                        input_dict["target_frame_index"].detach().cpu().tolist(),
                        input_dict["target_offset"].detach().cpu().tolist(),
                    )
                ]
            use_sample_save_names = save_subdirs is not None and save_stems is not None

            imgt_pred = result_dict['imgt_pred']
            if use_tcar:
                warped_img0 = result_dict['warped_img0']
                warped_img1 = result_dict['warped_img1']
                refine_res = result_dict['refine_res']
            psnr, psnr_list = calculate_batch_psnr(imgt, result_dict['imgt_pred'])
            ssim_return_list = record_std and "ssim" in std_metric_names
            lpips_return_list = record_std and "lpips" in std_metric_names
            ssim, ssim_values = calculate_batch_ssim(
                imgt, result_dict['imgt_pred'], return_list=ssim_return_list
            )
            lpips, lpips_values = calculate_batch_lpips(
                imgt, imgt_pred, self.lpips_metric, return_list=lpips_return_list
            )
            stlpips, bs = calculate_batch_stlpips(imgt, imgt_pred, self.stlpips_metric)
            niqe, bs = calculate_batch_niqe(imgt, imgt_pred, self.niqe_metric)
            metric_updates = {
                'psnr': self._metric_update_payload(
                    psnr,
                    len(psnr_list),
                    values=psnr_list,
                    use_exact_std=record_std and "psnr" in std_metric_names,
                ),
                'ssim': self._metric_update_payload(
                    ssim,
                    len(psnr_list),
                    values=ssim_values if ssim_return_list else None,
                    use_exact_std=ssim_return_list,
                ),
                'lpips': self._metric_update_payload(
                    lpips,
                    len(psnr_list),
                    values=lpips_values if lpips_return_list else None,
                    use_exact_std=lpips_return_list,
                ),
                'stlpips': {'value': stlpips, 'n': len(psnr_list)},
                'niqe': {'value': niqe, 'n': len(psnr_list)},
            }
            if vessel_psnr_enabled:
                vessel_psnr, vessel_psnr_list = calculate_batch_vessel_psnr(
                    imgt, result_dict['imgt_pred'], vessel_cfg=vessel_psnr_cfg
                )
                if vessel_psnr_list:
                    metric_updates['vessel_psnr'] = self._metric_update_payload(
                        vessel_psnr,
                        len(vessel_psnr_list),
                        values=vessel_psnr_list,
                        use_exact_std=record_std and "vessel_psnr" in std_metric_names,
                    )
            self.metric_logger.update(**metric_updates)
            if should_save_imgs:
                flow_h, flow_w = imgt.shape[-2:]
                flow_t0_vis = flow2img(
                    result_dict['flowt0_pred_list'][0].detach().clone()[..., :flow_h, :flow_w]
                )
                flow_t1_vis = flow2img(
                    result_dict['flowt1_pred_list'][0].detach().clone()[..., :flow_h, :flow_w]
                )
                for i in range(len(scene_names[1])):
                    if self.cfgs['mode'] == "test":
                        save_root = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator)
                    else:
                        save_root = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator)
                    if use_sample_save_names:
                        save_dir = os.path.join(save_root, save_subdirs[i])
                        save_stem = save_stems[i]
                        result_key = os.path.join(save_subdirs[i], f"{save_stem}.png")
                        scene0_path = os.path.join(save_dir, f"{save_stem}_img0.png")
                        scenet_path = os.path.join(save_dir, f"{save_stem}.png")
                        scene1_path = os.path.join(save_dir, f"{save_stem}_img1.png")
                        pred_path = os.path.join(save_dir, f"{save_stem}_pred.png")
                        warped0_path = os.path.join(save_dir, f"{save_stem}_warped_0.png")
                        warped1_path = os.path.join(save_dir, f"{save_stem}_warped_1.png")
                        refine_res_path = os.path.join(save_dir, f"{save_stem}_refine_res.png")
                        overlayed_path = os.path.join(save_dir, f"{save_stem}_overlayed.png")
                        residual_path = os.path.join(save_dir, f"{save_stem}_residual_x10.png")
                        flow_t0_path = os.path.join(save_dir, f"{save_stem}_flow_t0.png")
                        flow_t1_path = os.path.join(save_dir, f"{save_stem}_flow_t1.png")
                    else:
                        result_key = scene_names[1][i]
                        scene0_path = os.path.join(save_root, scene_names[0][i])
                        scenet_path = os.path.join(save_root, scene_names[1][i])
                        scene1_path = os.path.join(save_root, scene_names[2][i])
                        pred_path = scenet_path.replace('.', '_pred.')
                        warped0_path = scenet_path.replace('.', '_warped_0.')
                        warped1_path = scenet_path.replace('.', '_warped_1.')
                        refine_res_path = scenet_path.replace('.', '_refine_res.')
                        overlayed_path = scenet_path.replace('.', '_overlayed.')
                        residual_path = scenet_path.replace('.', '_residual_x10.')
                        flow_t0_path = scenet_path.replace('.', '_flow_t0.')
                        flow_t1_path = scenet_path.replace('.', '_flow_t1.')
                    psnr_dict[result_key] = float(psnr_list[i])
                    lpips_dict[result_key] = float(lpips)
                    stlpips_dict[result_key] = float(stlpips)
                    Path(scene0_path).parent.mkdir(exist_ok=True, parents=True)
                    save_image(img0[i], scene0_path)
                    save_image(imgt_pred[i], pred_path)
                    save_image(imgt[i], scenet_path)
                    save_image(img1[i], scene1_path)
                    if use_tcar:
                        save_image(warped_img0[i], warped0_path)
                        save_image(warped_img1[i], warped1_path)
                        save_image(
                            (refine_res[i] * 0.5 + 0.5).clamp(0, 1),
                            refine_res_path
                        )
                    save_image((img1[i] + img0[i]) / 2, overlayed_path)
                    save_image(torch.abs(imgt[i] - imgt_pred[i]) * 10, residual_path)
                    save_image(flow_t0_vis[i], flow_t0_path)
                    save_image(flow_t1_vis[i], flow_t1_path)

        metric_sync = self.cfgs.get("validation_metric_sync", "file")
        if metric_sync == "file":
            self._sync_validation_metrics_by_file(test_indicator)
        elif metric_sync != "none":
            self.metric_logger.synchronize_between_processes()
        available_std_names = std_metric_names.intersection(self.metric_logger.meters.keys())
        self.logger.info(
            f"Averaged validate stats:{self.metric_logger.print_avg(include_std=record_std, std_names=available_std_names)}"
        )
        if should_save_imgs:
            psnr_str = []
            psnr_dict = sorted(psnr_dict.items(), key=lambda item: item[1])
            for key, val in psnr_dict:
                psnr_str.append("{}: {}".format(key, val))
            psnr_str = "\n".join(psnr_str)
            if self.cfgs['mode'] == "test":
                outdir = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator)
            else:
                outdir = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator)
            with open(os.path.join(outdir, "results.txt"), "w") as f:
                f.write(psnr_str)
        if misc.is_main_process() and self.cfgs['mode'] == 'train':
            self.summary_writer.add_scalar("Val/psnr", self.metric_logger.psnr.global_avg, self.current_epoch)
            self.summary_writer.add_scalar("Val/ssim", self.metric_logger.ssim.global_avg, self.current_epoch)
            if vessel_psnr_enabled and 'vessel_psnr' in self.metric_logger.meters:
                self.summary_writer.add_scalar(
                    "Val/vessel_psnr", self.metric_logger.vessel_psnr.global_avg, self.current_epoch
                )
            if record_std:
                for metric_name in sorted(available_std_names):
                    self.summary_writer.add_scalar(
                        f"Val/{metric_name}_std",
                        self.metric_logger.meters[metric_name].global_std,
                        self.current_epoch,
                    )
        if self.cfgs.get('enable_wandb', False):
            wandb_metrics = {
                'val_psnr': self.metric_logger.psnr.global_avg,
                'val_ssim': self.metric_logger.ssim.global_avg,
                'val_lpips': self.metric_logger.lpips.global_avg,
                'val_stlpips': self.metric_logger.stlpips.global_avg,
                'val_niqe': self.metric_logger.niqe.global_avg,
            }
            if vessel_psnr_enabled and 'vessel_psnr' in self.metric_logger.meters:
                wandb_metrics['val_vessel_psnr'] = self.metric_logger.vessel_psnr.global_avg
            if record_std:
                for metric_name in sorted(available_std_names):
                    wandb_metrics[f'val_{metric_name}_std'] = self.metric_logger.meters[metric_name].global_std
            wandb.log(wandb_metrics, step=self.current_iteration)
        return self.metric_logger.psnr.global_avg
