import os
from glob import glob

from datasets import register
from .data_utils import *

import torch
import torchvision.transforms.v2.functional as TF
from torchvision.io import read_image
from torch.utils.data import Dataset


@register("aaa_2d_triplet_3frame")
class AAA2DTriplet3Frame(Dataset):
    """
    AAA-2D-三元组-插3帧:
      frame_1.png, frame_2_gt.png, frame_3_gt.png, frame_4_gt.png, frame_5.png
    训练/验证样本展开为 3 个 time_step: [0.25, 0.50, 0.75]
    """
    # T_LIST = [0.25, 0.50, 0.75]  # [0.25]
    T_LIST = [0.25]

    # GT_NAME = {0.25: "frame_2_gt.png", 0.50: "frame_3_gt.png", 0.75: "frame_4_gt.png"}  # {0.25: "frame_2_gt.png"}
    GT_NAME = {0.25: "frame_2_gt.png"}


    END0 = "frame_1.png"
    END1 = "frame_5.png"

    def __init__(self, root_path, patch_size=(224, 224), split="train", **kwargs):
        super().__init__()
        self.data_root = root_path
        self.mode = split

        # # patch_size 兼容 int / list / tuple
        # if isinstance(patch_size, int):
        #     self.patch_size = (patch_size, patch_size)
        # else:
        #     self.patch_size = tuple(patch_size)

        self.patch_size = patch_size

        # 找到所有包含 frame_1.png 的目录
        f1_list = glob(os.path.join(self.data_root, "**", self.END0), recursive=True)
        dirs = sorted(set(os.path.dirname(p) for p in f1_list))

        # 过滤出文件齐全的目录
        # need = [self.END0, self.END1, "frame_2_gt.png", "frame_3_gt.png", "frame_4_gt.png"]  # need = [self.END0, self.END1, "frame_2_gt.png"]
        need = [self.END0, self.END1, "frame_2_gt.png"]

        valid = []
        for d in dirs:
            if all(os.path.exists(os.path.join(d, fn)) for fn in need):
                valid.append(d)

        # 95/5 切分（跟 vimeo 一样）
        cnt = int(len(valid) * 0.95)
        if self.mode == "train":
            self.sample_dirs = valid[:cnt]
        else:  # val/test 都走后 5%
            self.sample_dirs = valid[cnt:]

        # 展开成 (dir, t)
        self.items = [(d, t) for d in self.sample_dirs for t in self.T_LIST]

        if len(self.items) == 0:
            raise RuntimeError(f"No valid samples found under {root_path}")

    def __len__(self):
        return len(self.items)

    def get_img(self, index):
        d, t = self.items[index]
        img0 = read_image(os.path.join(d, self.END0))
        img1 = read_image(os.path.join(d, self.END1))
        imgt = read_image(os.path.join(d, self.GT_NAME[t]))

        rel = os.path.relpath(d, self.data_root)
        scene_names = [
            os.path.join(rel, self.END0),
            os.path.join(rel, self.GT_NAME[t]),
            os.path.join(rel, self.END1),
        ]
        return img0, imgt, img1, float(t), scene_names

    def __getitem__(self, item):
        img0, imgt, img1, embt, scene_names = self.get_img(item)
        time_step = torch.tensor([embt], dtype=torch.float32).reshape(1, 1, 1)

        if self.mode == "train":
            img0, imgt, img1 = random_crop(img0, imgt, img1, self.patch_size)
            # if random.random() > 0.5:
            #     img0, imgt, img1 = random_hor_flip(img0, imgt, img1)
            # if random.random() > 0.5:
            #     img0, imgt, img1 = random_ver_flip(img0, imgt, img1)
            # 医学图可能是 1 通道，跳过颜色置换
            # if img0.shape[0] == 3 and random.random() > 0.5:
            #     img0, imgt, img1 = random_color_permutation(img0, imgt, img1)
            if random.random() > 0.5:
                img0, imgt, img1, time_step = random_temporal_flip(img0, imgt, img1, time_step)
            # degree = random.randint(0, 3)
            # img0, imgt, img1 = random_rotation(img0, imgt, img1, degree)

        return {
            "img0": TF.to_dtype(img0, torch.float32, scale=True),
            "imgt": TF.to_dtype(imgt, torch.float32, scale=True),
            "img1": TF.to_dtype(img1, torch.float32, scale=True),
            "time_step": time_step,
            "scene_names": scene_names,
        }
