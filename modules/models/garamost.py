import sys
import os
import logging
import torch
from pathlib import Path
from torchvision.utils import save_image

sys.path.insert(0, '/home/jiaxuan/tmp_model/GaraMoSt')
import config as garamost_cfg
from Trainer import Model as GaraMoStModel

from modules.models.base_model import BaseModel
from modules.models import register
from utils.padder import InputPadder
from utils.metrics import calculate_batch_psnr, calculate_batch_ssim
import utils.misc as misc


@register('garamost')
class GaraMoSt(BaseModel):
    def __init__(self, cfgs):
        super(GaraMoSt, self).__init__(cfgs)
        
        model_args = cfgs.get('model', {}).get('args', {})
        
        context_aware_granularity = model_args.get('lambda_r', [7, 7])
        
        self.most_model = GaraMoStModel(-1, context_aware_granularity)
        
        resume_path = cfgs.get('resume')
        if resume_path and os.path.exists(resume_path):
            self.most_model.load_model(full_path=resume_path)
        
        self.most_model.eval()
        self.most_model.net.to(self.device)
        
        self.model_without_ddp = self.most_model.net
        self.model = self.most_model.net
        
        self.use_tta = model_args.get('TTA', False)
        self.weight_loaded = True

    def forward(self, img0, img1, time_step=None, imgt=None, run_with_gt=False, **kwargs):
        if time_step is None:
            time_step = 0.5
        
        img0 = img0[:, [2, 1, 0], :, :]
        img1 = img1[:, [2, 1, 0], :, :]
        
        if isinstance(time_step, torch.Tensor):
            time_step = time_step.squeeze()
            if time_step.dim() == 0:
                time_step = time_step.unsqueeze(0)
        
        with torch.no_grad():
            if isinstance(time_step, (int, float)) or (isinstance(time_step, torch.Tensor) and time_step.numel() == 1):
                padder = InputPadder(img0.shape, divisor=32)
                img0_pad, img1_pad = padder.pad(img0, img1)
                timestep_val = time_step.item() if isinstance(time_step, torch.Tensor) else time_step
                pred = self.most_model.inference(img0_pad, img1_pad, timestep=timestep_val, TTA=self.use_tta)
                pred = padder.unpad(pred)
            else:
                B = img0.shape[0]
                pred_list = []
                padder = InputPadder(img0.shape, divisor=32)
                
                for i in range(B):
                    img0_i = img0[i:i+1]
                    img1_i = img1[i:i+1]
                    t_i = time_step[i].item() if isinstance(time_step, torch.Tensor) else time_step[i]
                    
                    img0_pad_i, img1_pad_i = padder.pad(img0_i, img1_i)
                    pred_i = self.most_model.inference(img0_pad_i, img1_pad_i, timestep=t_i, TTA=self.use_tta)
                    pred_i = padder.unpad(pred_i)
                    pred_list.append(pred_i)
                
                pred = torch.cat(pred_list, dim=0)
            
            pred = torch.clamp(pred, 0, 1)
            pred = pred[:, [2, 1, 0], :, :]
            
        return {'imgt_pred': pred}

    def train_one_step(self, input_dict):
        raise NotImplementedError("GaraMoSt does not support training")

    def validate(self, val_loader, test_indicator=None):
        self.model_without_ddp.eval()
        
        self.metric_logger = misc.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('psnr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.metric_logger.add_meter('ssim', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Test:'
        
        if test_indicator is None:
            split = self.cfgs['test_dataset'].get('args', {}).get('split', '')
            test_indicator = f"{self.cfgs['test_dataset']['name']}_{split}" if split else self.cfgs['test_dataset']['name']
        
        print_freq = 10
        for input_dict in self.metric_logger.log_every(val_loader, print_freq, header):
            input_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
            
            img0 = input_dict['img0']
            imgt = input_dict['imgt']
            img1 = input_dict['img1']
            time_step = input_dict.get('time_step', 0.5)
            
            result_dict = self.forward(img0, img1, time_step=time_step)
            
            imgt_pred = result_dict['imgt_pred']
            
            scene_names = input_dict['scene_names']
            
            psnr, psnr_list = calculate_batch_psnr(imgt, imgt_pred)
            ssim, bs = calculate_batch_ssim(imgt, imgt_pred)
            self.metric_logger.update(psnr={'value': psnr, 'n': len(psnr_list)},
                                      ssim={'value': ssim, 'n': len(psnr_list)})
            
            if self.cfgs['test_dataset']['save_imgs']:
                # for i in range(len(scene_names)):
                #     scene_path = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator, scene_names[i])
                #     Path(scene_path).parent.mkdir(exist_ok=True, parents=True)
                #     save_image(img0[i], scene_path.replace('.', '_img0.'))
                #     save_image(imgt_pred[i], scene_path.replace('.', '_pred.'))
                #     save_image(imgt[i], scene_path)
                #     save_image(img1[i], scene_path.replace('.', '_img1.'))
                for i in range(len(scene_names[1])):
                    scene_path = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator, scene_names[1][i])
                    Path(scene_path).parent.mkdir(exist_ok=True, parents=True)
                    save_image(img0[i], scene_path.replace('.', '_img0.'))
                    save_image(imgt_pred[i], scene_path.replace('.', '_pred.'))
                    save_image(imgt[i], scene_path)
                    save_image(img1[i], scene_path.replace('.', '_img1.'))
                    save_image(torch.abs(imgt[i] - imgt_pred[i]) * 10, scene_path.replace('.', '_residual_x10.'))
        
        self.logger.info(f"Averaged validate stats:{self.metric_logger.print_avg()}")
        
        if misc.is_main_process() and self.cfgs.get('enable_wandb', False):
            import wandb
            wandb.run.summary[f"{test_indicator}_psnr"] = self.metric_logger.psnr.global_avg
            wandb.run.summary[f"{test_indicator}_ssim"] = self.metric_logger.ssim.global_avg
        
        return self.metric_logger.psnr.global_avg