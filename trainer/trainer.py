import datetime
import heapq
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader, DistributedSampler, Sampler

import datasets
import modules.models as models
import utils
from utils.misc import get_rank, get_world_size, is_main_process


def synchronize():
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    torch.distributed.barrier()


def _estimate_validation_cost(dataset, index):
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return 1.0

    sample = samples[index]
    frame_items = sample.get("frame_items")
    sparse_anchor_positions = sample.get("sparse_anchor_positions")
    if frame_items is None or sparse_anchor_positions is None:
        return 1.0

    source = getattr(dataset, "global_context_source", None)
    if source == "sparse_anchors":
        global_len = len(sparse_anchor_positions)
    elif source == "dense_without_target":
        global_len = max(1, len(frame_items) - 1)
    elif source == "dense_all":
        global_len = len(frame_items)
    else:
        global_len = getattr(dataset, "context_length", 1)

    max_frames = getattr(dataset, "global_context_max_frames", None)
    if max_frames is not None:
        global_len = min(global_len, int(max_frames))

    local_len = getattr(dataset, "context_length", 0)
    return float(max(1, global_len + local_len + 3))


class CostBalancedDistributedSampler(Sampler):
    """
    Assign validation samples by estimated frame cost instead of equal count.

    Video-sequence samples can vary widely in global context length, so equal-count
    DistributedSampler shards may leave one rank processing long sequences while
    other ranks wait at the final metric sync.
    """

    def __init__(self, dataset, num_replicas, rank, shuffle=False, seed=0):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._cached_epoch = None
        self._cached_indices = None

        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"Invalid rank {rank} for num_replicas={num_replicas}.")

    def _rank_indices(self):
        if self._cached_epoch == self.epoch and self._cached_indices is not None:
            return self._cached_indices

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            indices = list(range(len(self.dataset)))
            rng.shuffle(indices)
        else:
            indices = list(range(len(self.dataset)))

        weighted = [
            (-_estimate_validation_cost(self.dataset, index), index)
            for index in indices
        ]
        weighted.sort()

        assignments = [[] for _ in range(self.num_replicas)]
        heap = [(0.0, rank) for rank in range(self.num_replicas)]
        heapq.heapify(heap)

        for neg_cost, index in weighted:
            load, rank = heapq.heappop(heap)
            assignments[rank].append(index)
            heapq.heappush(heap, (load - neg_cost, rank))

        self._cached_epoch = self.epoch
        self._cached_indices = assignments[self.rank]
        return self._cached_indices

    def __iter__(self):
        return iter(self._rank_indices())

    def __len__(self):
        return len(self._rank_indices())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._cached_epoch = None
        self._cached_indices = None


class Trainer(object):
    """
    Wrapper for training, more related to engineering than research code.
    """

    def __init__(self, cfgs):
        self.rank = get_rank()
        self.cfgs = cfgs
        self.is_master = (self.rank == 0)
        self.is_train = False

        env = cfgs["env"]
        self.tot_gpus = get_world_size()
        self.distributed = (get_world_size() > 1)

        if self.is_master:
            logger = utils.misc.set_save_dir(cfgs["log_dir"], cfgs["run_description"], replace=False)
            with open(osp.join(cfgs["cfg_dir"], f'cfg_{cfgs["run_description"]}.yaml'), "w") as f:
                yaml.dump(cfgs, f, sort_keys=False)
            self.log = logger.info
            self.enable_tb = True
        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False
            self.enable_wandb = False

        self.make_datasets()
        self.model = models.make(cfgs)

        if "resume" not in self.cfgs and "pretrained" in self.cfgs:
            ckpt = torch.load(self.cfgs["pretrained"], map_location="cpu")
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model_state = self.model.model_without_ddp.state_dict()
            matched_state = {}
            mismatched = []
            unexpected = []
            for key, value in state.items():
                if key not in model_state:
                    unexpected.append(key)
                    continue
                if model_state[key].shape != value.shape:
                    mismatched.append(
                        f"{key}: ckpt{tuple(value.shape)} != model{tuple(model_state[key].shape)}"
                    )
                    continue
                matched_state[key] = value

            load_result = self.model.model_without_ddp.load_state_dict(matched_state, strict=False)
            missing = list(load_result.missing_keys)
            self.log(f'Loaded pretrained weights from {self.cfgs["pretrained"]}')
            self.log(f"Missing keys: {missing}")
            self.log(f"Unexpected keys: {unexpected}")
            if mismatched:
                self.log(f"Skipped mismatched keys ({len(mismatched)}): {mismatched}")

        for mod_name in cfgs["model"].get("freeze_modules", []):
            self._set_module_requires_grad(mod_name, False)
        for prefix in cfgs["model"].get("freeze_param_prefixes", []):
            self._set_prefix_requires_grad(prefix, False)

        self.start_epoch = 0
        self.end_epoch = self.cfgs["max_epoch"]
        if "resume" in self.cfgs:
            run_id = self.model.load_checkpoint(self.cfgs["resume"])
            self.start_epoch = self.model.current_epoch
        else:
            run_id = wandb.util.generate_id()

        if self.is_master and env["wandb_upload"]:
            self.enable_wandb = True
            self.cfgs["enable_wandb"] = True
            with open("wandb.yaml", "r") as f:
                wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
            os.environ["WANDB_DIR"] = env["save_dir"]
            os.environ["WANDB_NAME"] = env["exp_name"]
            os.environ["WANDB_API_KEY"] = wandb_cfg["api_key"]
            wandb.init(
                project=wandb_cfg["project"],
                entity=wandb_cfg["entity"],
                config=cfgs,
                id=run_id,
                name=env["exp_name"],
                resume="allow",
            )
        else:
            self.enable_wandb = False
            self.cfgs["enable_wandb"] = False

    def _resolve_module(self, module_path):
        module = self.model.model_without_ddp
        for attr in module_path.split("."):
            module = getattr(module, attr, None)
            if module is None:
                return None
        return module

    def _set_module_requires_grad(self, module_path, requires_grad):
        module = self._resolve_module(module_path)
        if module is None:
            self.log(f"Warning: module '{module_path}' not found, skip.")
            return
        for param in module.parameters():
            param.requires_grad = requires_grad
        action = "unfrozen" if requires_grad else "frozen"
        self.log(f"Module '{module_path}' has been {action} (requires_grad={requires_grad}).")

    def _set_prefix_requires_grad(self, prefixes, requires_grad):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        prefixes = tuple(prefixes)
        matched = 0
        for name, param in self.model.model_without_ddp.named_parameters():
            if name.startswith(prefixes):
                param.requires_grad = requires_grad
                matched += 1
        action = "unfrozen" if requires_grad else "frozen"
        if matched == 0:
            self.log(f"Warning: parameter prefixes {list(prefixes)} not found, skip.")
        else:
            self.log(
                f"Parameter prefixes {list(prefixes)} have been {action} "
                f"(matched {matched} tensors)."
            )

    def make_datasets(self):
        """
        By default, train dataset performs shuffle and drop_last.
        Distributed sampler will extend the dataset with a prefix to make the
        length divisible by tot_gpus, samplers are stored in .dist_samplers.
        """
        cfgs = self.cfgs
        self.dist_samplers = []

        def make_distributed_loader(
            dataset,
            batch_size,
            num_workers,
            shuffle=False,
            drop_last=False,
            balance_by_cost=False,
        ):
            if self.distributed and balance_by_cost:
                sampler = CostBalancedDistributedSampler(
                    dataset,
                    num_replicas=self.tot_gpus,
                    rank=self.rank,
                    shuffle=shuffle,
                    seed=self.cfgs["seed"],
                )
            else:
                sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
            collate_fn = getattr(dataset, "collate_fn", None)
            per_rank_batch_size = batch_size // self.tot_gpus
            per_rank_num_workers = num_workers // self.tot_gpus
            if per_rank_batch_size < 1:
                raise ValueError(
                    f"Global batch_size ({batch_size}) is smaller than the number of GPUs ({self.tot_gpus}). "
                    f"Please set loader.batch_size >= {self.tot_gpus} for distributed runs."
                )
            loader = DataLoader(
                dataset,
                per_rank_batch_size,
                drop_last=drop_last,
                sampler=sampler,
                shuffle=(shuffle and (sampler is None)),
                num_workers=per_rank_num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            return loader, sampler

        if cfgs.get("mode") == "train" and cfgs.get("train_dataset") is not None:
            train_dataset = datasets.make(cfgs["train_dataset"])
            self.log(f"Train dataset: len={len(train_dataset)}")
            loader_cfg = cfgs["train_dataset"]["loader"]
            self.train_loader, train_sampler = make_distributed_loader(
                train_dataset,
                loader_cfg["batch_size"],
                loader_cfg["num_workers"],
                shuffle=loader_cfg.get("shuffle", True),
                drop_last=True,
            )
            self.dist_samplers.append(train_sampler)
            if self.cfgs["lr_scheduler"]["name"] == "one_cycle_lr":
                self.cfgs["lr_scheduler"]["args"]["total_steps"] = (
                    len(self.train_loader) * self.cfgs["max_epoch"]
                )

        if cfgs.get("test_dataset") is not None:
            test_dataset = datasets.make(cfgs["test_dataset"])
            self.log(f"Test dataset: len={len(test_dataset)}")
            loader_cfg = cfgs["test_dataset"]["loader"]
            self.test_loader, test_sampler = make_distributed_loader(
                test_dataset,
                loader_cfg["batch_size"],
                loader_cfg["num_workers"],
                shuffle=loader_cfg.get("shuffle", False),
                drop_last=False,
                balance_by_cost=loader_cfg.get("balance_by_cost", False),
            )
            self.dist_samplers.append(test_sampler)

        if cfgs.get("demo_dataset") is not None:
            self.demo_root = self.cfgs["demo_dataset"]["args"]["root_path"]

    def train(self):
        print("Start training")
        start_time = time.time()
        self.is_train = True
        self.model.init_training_logger()
        self.best_performance = 0

        if self.cfgs.get("test_before_train", False) and getattr(self, "test_loader", None) is not None:
            self.log("Running initial test before training...")
            self.model.init_validation_logger()
            self.validate()

        for epoch in range(self.start_epoch, self.end_epoch):
            if self.cfgs["distributed"]:
                self.train_loader.batch_sampler.sampler.set_epoch(epoch)

            random.seed(self.cfgs["seed"] + epoch)
            np.random.seed(self.cfgs["seed"] + epoch)
            torch.random.manual_seed(self.cfgs["seed"] + epoch)
            torch.manual_seed(self.cfgs["seed"] + epoch)
            torch.cuda.manual_seed_all(self.cfgs["seed"] + epoch)

            for mod_name, target_epoch in self.cfgs["model"].get("unfreeze_epoch", {}).items():
                if epoch + 1 == target_epoch:
                    self._set_module_requires_grad(mod_name, True)

            for prefix, target_epoch in self.cfgs["model"].get("unfreeze_param_prefixes", {}).items():
                if epoch + 1 == target_epoch:
                    self._set_prefix_requires_grad(prefix, True)

            self.model.train_one_epoch(self.train_loader, epoch)

            if is_main_process():
                self.model.save_checkpoint(f"model_{epoch + 1}_before_test.pth")
            synchronize()

            if ((epoch + 1) % self.cfgs["validate_every"]) == 0:
                performance = self.validate()
                if is_main_process() and performance > self.best_performance:
                    self.best_performance = performance
                    self.model.save_checkpoint(f"model_{epoch + 1}.pth", is_best=1)
                    self.log(
                        "best performance achieved at epoch {} with performance of {}".format(
                            epoch, self.best_performance
                        )
                    )
                synchronize()

            if ((epoch + 1) % self.cfgs["save_every"]) == 0 and is_main_process():
                self.model.save_checkpoint(f"model_{epoch + 1}.pth")
            synchronize()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")
        if is_main_process():
            self.finalize_training()

    def validate(self):
        if not self.is_train:
            self.model.init_validation_logger()
        performance = self.model.validate(self.test_loader)
        synchronize()
        return performance

    def benchmark(self):
        self.model.init_testing_logger()
        self.model.benchmark()

    def demo(self):
        self.model.init_demo_logger()
        self.model.demo(self.demo_root)

    def finalize_training(self):
        self.model.finalize_training()
