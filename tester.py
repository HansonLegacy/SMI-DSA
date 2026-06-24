import os
import argparse
import yaml
import json
import datetime
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import time as time_module

from utils.metrics import calculate_batch_psnr, calculate_batch_ssim
import datasets
import modules.models as models


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    return rank, world_size, local_rank


class UnifiedTester:
    def __init__(self, cfgs, rank=0, world_size=1):
        self.cfgs = cfgs
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank}')
        else:
            self.device = torch.device('cpu')
        
        self.models = {}
        self.results = {}
    
    def register_model(self, name, model_cfg):
        """注册模型: name为模型标识, model_cfg包含模型配置"""
        if self.is_main_process:
            print(f"Loading model: {name}")
        model = models.make(model_cfg)
        
        # 加载预训练权重（如果模型没有自行加载）
        if not getattr(model, 'weight_loaded', False):
            if 'resume' in model_cfg and model_cfg['resume']:
                if self.is_main_process:
                    print(f"Loading checkpoint from: {model_cfg['resume']}")
                checkpoint = torch.load(model_cfg['resume'], map_location='cpu')
                state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
                missing, unexpected = model.model_without_ddp.load_state_dict(state, strict=False)
                if self.is_main_process:
                    print(f"Loaded pretrained weights")
                    print(f"Missing keys: {missing}")
                    print(f"Unexpected keys: {unexpected}")
        
        self.models[name] = model
        self.results[name] = {
            'psnr': [], 
            'ssim': [], 
        }
        self.output_dir = model_cfg.get('output_dir')
    
    def load_test_dataset(self, dataset_cfg):
        """统一数据集加载，支持数据分片"""
        if self.is_main_process:
            print(f"Loading dataset: {dataset_cfg['name']}")
        
        dataset = datasets.make(dataset_cfg)
        
        # 与trainer保持一致：从test_dataset.loader中读取batch_size和num_workers
        # 与trainer的make_distributed_loader一致，batch_size需要除以world_size
        loader_cfg = dataset_cfg.get('loader', {})
        batch_size = loader_cfg.get('batch_size', 1)
        num_workers = loader_cfg.get('num_workers', 4)
        
        # 与trainer保持一致：分布式测试时，每个GPU的batch_size要除以GPU数量
        if self.world_size > 1:
            batch_size = batch_size // self.world_size
            num_workers = num_workers // max(1, self.world_size)
        
        # 数据分片：每个GPU处理一部分
        if self.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
            )
            collate_fn = getattr(dataset, "collate_fn", None)
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                              num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
        else:
            collate_fn = getattr(dataset, "collate_fn", None)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
        
        return loader
    
    def inference_single_model(self, model_name, img0, img1, time_step, input_dict):
        """推理单个模型 - 子类实现具体的模型接口"""
        raise NotImplementedError("子类必须实现inference_single_model方法")
    
    # 移除自定义的calculate_metrics方法，使用model.validate()中的指标计算
    pass
    
    def run(self, dataset_cfg, target_model_name=None):
        """在统一数据集上运行指定的模型"""
        test_loader = self.load_test_dataset(dataset_cfg)
        
        # 确定要测试的模型
        if target_model_name is not None:
            if target_model_name not in self.models:
                raise ValueError(f"Model '{target_model_name}' not found. Available: {list(self.models.keys())}")
            test_models = {target_model_name: self.models[target_model_name]}
        else:
            test_models = self.models
        
        # 初始化结果存储
        for model_name in test_models:
            self.results[model_name] = {
                'psnr': [], 'ssim': [], 'scene_names': []
            }
        
        # 显示总样本数
        # 优先使用sampler.num_samples，这是每个GPU实际处理的样本数（不含padding）
        if hasattr(test_loader, 'sampler') and hasattr(test_loader.sampler, 'num_samples'):
            total_samples = test_loader.sampler.num_samples
        elif hasattr(test_loader, 'sampler') and hasattr(test_loader.sampler, 'dataset'):
            total_samples = len(test_loader.sampler.dataset)
        elif hasattr(test_loader, 'dataset'):
            total_samples = len(test_loader.dataset)
        else:
            total_samples = len(test_loader) * test_loader.batch_size
        
        if self.is_main_process:
            print(f"\nRunning test on {total_samples} samples...")
        

        
        # 复用model.validate()流程，确保与训练时验证逻辑一致
        # 注意：validate方法会遍历整个test_loader，每个GPU只处理自己的分片数据
        for model_name, model in test_models.items():
            # 初始化测试日志（与trainer.py中test_before_train一致）
            model.init_validation_logger()
            
            # 调用validate方法，与trainer.py中的流程一致
            # validate方法会遍历整个test_loader，每个GPU只处理自己的分片数据
            model.validate(test_loader)
            
            psnr = model.metric_logger.psnr.global_avg
            ssim = model.metric_logger.ssim.global_avg
            
            self.results[model_name]['psnr'].append(psnr)
            self.results[model_name]['ssim'].append(ssim)
        
        # 收集各GPU的结果并聚合
        if self.world_size > 1:
            import torch.distributed as dist
            dist.barrier()  # 同步所有 GPU
            self._aggregate_results(target_model_name)
        
        self._print_summary(target_model_name)
        self._save_results(target_model_name)
    
    def _aggregate_results(self, model_name=None):
        """聚合各GPU的结果"""
        if model_name is None:
            model_names = list(self.results.keys())
        else:
            model_names = [model_name]
        
        for name in model_names:
            psnr_list = self.results[name]['psnr']
            ssim_list = self.results[name]['ssim']
            count = len(psnr_list)
            
            # 计算当前GPU的平均值
            avg_psnr = sum(psnr_list) / count if count > 0 else 0
            avg_ssim = sum(ssim_list) / count if count > 0 else 0
            
            # 转换为tensor
            psnr_tensor = torch.tensor([avg_psnr], dtype=torch.float64, device=self.device)
            ssim_tensor = torch.tensor([avg_ssim], dtype=torch.float64, device=self.device)
            count_tensor = torch.tensor([count], dtype=torch.long, device=self.device)
            
            # 汇总到主进程
            psnr_gathered = [torch.zeros_like(psnr_tensor) for _ in range(self.world_size)]
            ssim_gathered = [torch.zeros_like(ssim_tensor) for _ in range(self.world_size)]
            count_gathered = [torch.zeros_like(count_tensor) for _ in range(self.world_size)]
            
            dist.all_gather(psnr_gathered, psnr_tensor)
            dist.all_gather(ssim_gathered, ssim_tensor)
            dist.all_gather(count_gathered, count_tensor)
            
            # 主进程加权平均
            if self.is_main_process:
                psnr_vals = torch.cat(psnr_gathered).cpu().numpy()
                ssim_vals = torch.cat(ssim_gathered).cpu().numpy()
                count_vals = torch.cat(count_gathered).cpu().numpy().tolist()
                
                total_count = sum(count_vals)
                weighted_psnr = sum(psnr_vals[i] * count_vals[i] for i in range(len(count_vals))) / total_count
                weighted_ssim = sum(ssim_vals[i] * count_vals[i] for i in range(len(count_vals))) / total_count
                
                self.results[name]['psnr'] = [weighted_psnr]
                self.results[name]['ssim'] = [weighted_ssim]
                self.results[name]['count'] = total_count
                print(f"[Rank {self.rank}] Aggregated {total_count} samples from {self.world_size} GPUs")
    
    def _print_summary(self, model_name):
        if not self.is_main_process:
            return
            
        print("\n" + "="*60)
        print("Test Results Summary")
        print("="*60)
        
        name = model_name
        res = self.results[name]
        if 'psnr' in res and len(res['psnr']) > 0:
            avg_psnr = sum(res['psnr']) / len(res['psnr'])
        else:
            avg_psnr = 0
        
        if 'ssim' in res and len(res['ssim']) > 0:
            avg_ssim = sum(res['ssim']) / len(res['ssim'])
        else:
            avg_ssim = 0
        
        count = self.cfgs.get('test_dataset', {}).get('args', {}).get('num_samples', 
                     len(res.get('psnr', [])) * self.cfgs.get('test_dataset', {}).get('loader', {}).get('batch_size', 32))
        print(f"\nModel: {name}")
        print(f"  Samples: {count}")
        print(f"  PSNR:   {avg_psnr:.4f}")
        print(f"  SSIM:   {avg_ssim:.4f}")
    
    def _save_results(self, model_name, save_path=None):
        if not self.is_main_process:
            return
            
        if save_path is None:
            save_path = self.output_dir
        os.makedirs(save_path, exist_ok=True)
        
        name = model_name
        res = self.results[name]
        
        if 'psnr' in res and len(res['psnr']) > 0:
            avg_psnr = sum(res['psnr']) / len(res['psnr'])
        else:
            avg_psnr = 0
        
        if 'ssim' in res and len(res['ssim']) > 0:
            avg_ssim = sum(res['ssim']) / len(res['ssim'])
        else:
            avg_ssim = 0
        
        count = self.cfgs.get('test_dataset', {}).get('args', {}).get('num_samples', 
                     len(res.get('psnr', [])) * self.cfgs.get('test_dataset', {}).get('loader', {}).get('batch_size', 32))
        
        summary = {
            'model': name,
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'sample_count': count,
        }
        
        filename = f"{name}_results.json"
        save_file = os.path.join(save_path, filename)
        with open(save_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {save_file}")


class BiMVFITester(UnifiedTester):
    """BiM-VFI模型测试器"""
    
    def inference_single_model(self, model_name, img0, img1, time_step, input_dict):
        model_wrapper = self.models[model_name]
        pyr_level = input_dict.get('pyr_level', 3)
        
        # 构建输入字典
        model_input = {
            'img0': img0,
            'img1': img1,
            'time_step': time_step,
            'pyr_level': pyr_level,
            'imgt': None,
            'run_with_gt': False,
        }
        
        with torch.no_grad():
            # 注意：model_wrapper.model 才是真正的神经网络组件
            result_dict = model_wrapper.model(**model_input)
        return result_dict['imgt_pred']


class MostDSATester(UnifiedTester):
    """MoStDSA模型测试器"""
    
    def inference_single_model(self, model_name, img0, img1, time_step, input_dict):
        model_wrapper = self.models[model_name]
        
        with torch.no_grad():
            if time_step.dim() == 0:
                time_step = time_step.unsqueeze(0)
            
            batch_size = img0.shape[0]
            preds = []
            for i in range(batch_size):
                t = time_step[i].item() if time_step.numel() > 1 else time_step.item()
                img0_i = img0[i:i+1]
                img1_i = img1[i:i+1]
                result_dict = model_wrapper(img0_i, img1_i, t=t)
                preds.append(result_dict['pred'])
            
            imgt_pred = torch.cat(preds, dim=0)
        return imgt_pred


class GaraMoStTester(UnifiedTester):
    """GaraMoSt模型测试器 - 根据实际接口调整"""
    
    def inference_single_model(self, model_name, img0, img1, time_step, input_dict):
        model_wrapper = self.models[model_name]
        
        with torch.no_grad():
            if time_step.dim() == 0:
                time_step = time_step.unsqueeze(0)
            
            batch_size = img0.shape[0]
            preds = []
            for i in range(batch_size):
                t = time_step[i].item() if time_step.numel() > 1 else time_step.item()
                img0_i = img0[i:i+1]
                img1_i = img1[i:i+1]
                result_dict = model_wrapper(img0_i, img1_i, time_step=t)
                preds.append(result_dict['imgt_pred'])
            
            imgt_pred = torch.cat(preds, dim=0)
        return imgt_pred


def create_tester(cfgs, model_type, rank=0, world_size=1):
    """根据模型类型创建对应的测试器"""
    testers = {
        'bimvfi': BiMVFITester,
        'mostdsa': MostDSATester,
        'garamost': GaraMoStTester,
    }
    
    if model_type not in testers:
        raise ValueError(f"Unknown model_type: '{model_type}'. Available: {list(testers.keys())}")
    
    tester_class = testers[model_type]
    return tester_class(cfgs, rank=rank, world_size=world_size)


def main():
    parser = argparse.ArgumentParser(description="Unified Test Script for VFI Models")
    parser.add_argument("--config", type=str, default="cfgs/tester.yaml", 
                       help="Path to config file")
    parser.add_argument("--model", type=str, required=True,
                       help="Specify model name to test (must be in config)")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    
    # 检测是否使用多GPU
    num_gpus = cfgs.get('num_gpus', 1)
    use_distributed = num_gpus > 1
    
    if use_distributed:
        rank, world_size, local_rank = setup_distributed()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank, timeout=datetime.timedelta(hours=24))
        
        if rank == 0:
            print(f"Running distributed test with {world_size} GPUs")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        print("Running single GPU test")
    
    # 获取指定模型的信息
    target_model_name = args.model
    model_type = None
    target_model_info = None
    
    if 'models' in cfgs:
        for model_info in cfgs['models']:
            if model_info.get('name') == target_model_name:
                target_model_info = model_info
                model_type = model_info.get('type', 'bimvfi')
                break
        
        if target_model_info is None:
            available_models = [m.get('name') for m in cfgs['models']]
            raise ValueError(f"Model '{target_model_name}' not found in config. Available: {available_models}")
    else:
        raise ValueError("'models' key not found in config file.")
    
    # 根据模型类型创建测试器
    tester = create_tester(cfgs, model_type, rank=rank, world_size=world_size)
    
    # 注册指定模型
    exp_name = cfgs.get('exp_name', 'test')
    save_root = cfgs.get('save_root', 'test_outputs')
    save_dir = os.path.join(save_root, exp_name)
    is_distributed = world_size > 1
    
    model_cfg = {
        'model': {
            'name': target_model_info.get('name', target_model_name),
            'args': target_model_info.get('args', {}),
        },
        'mode': 'test',
        'resume': target_model_info.get('resume'),
        'distributed': is_distributed,
        'gpu': rank,
        'world_size': world_size,
        'rank': rank,
        'test_dataset': cfgs.get('test_dataset', cfgs.get('dataset')),
        'output_dir': os.path.join(save_dir, 'output'),
        'summary_dir': os.path.join(save_dir, 'summaries'),
    }
    tester.register_model(target_model_name, model_cfg)
    
    # 准备数据集配置
    dataset_cfg = cfgs.get('test_dataset', cfgs.get('dataset', {'name': 'xiehe_3_datasplit', 'args': {}}))
    
    # 运行测试
    tester.run(dataset_cfg, target_model_name=target_model_name)
    
    # 清理分布式环境
    if use_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
