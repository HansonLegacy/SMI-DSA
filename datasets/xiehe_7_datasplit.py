# datasets/xiehe_2d_3_brain.py
# 这是补充的：协和脑2d插3帧数据集
# AAA-2D-三元组-插3帧
import os
from glob import glob

from datasets import register
from .data_utils import *

import torch
import torchvision.transforms.v2.functional as TF
from torchvision.io import read_image
from torch.utils.data import Dataset

# 固定七帧插值位置
T_LIST = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8]
# T_LIST = [1/4, 2/4, 3/4]  # 这是插3帧
GT_NAME = {1/8: "frame_2_gt.png", 2/8: "frame_3_gt.png", 3/8: "frame_4_gt.png", 4/8: "frame_5_gt.png", 5/8: "frame_6_gt.png", 6/8: "frame_7_gt.png", 7/8: "frame_8_gt.png"}
# GT_NAME = {1/4: "frame_2_gt.png", 2/4: "frame_3_gt.png", 3/4: "frame_4_gt.png"}

@register("xiehe_7_datasplit")
class XHTrans7Frame(Dataset):
    """
    协和脑血管插7帧数据集：
    每个样本目录包含 frame_1.png / frame_9.png / frame_2_gt.png ~ frame_8_gt.png
    训练样本展开为 7 个 item：(t=1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8)
    """
    def __init__(self, root_path, patch_size=(224, 224), split="train", **kwargs):
        super(XHTrans7Frame, self).__init__()
        self.data_root = root_path
        self.mode = split
        self.patch_size = patch_size

        # 扫描所有包含 frame_1.png 的目录
        f1_list = glob(os.path.join(self.data_root, "**", "frame_1.png"), recursive=True)
        dirs = sorted(set(os.path.dirname(p) for p in f1_list))

        # 过滤：必须包含所有需要的文件
        valid = []
        need = ["frame_1.png", "frame_9.png", "frame_2_gt.png", "frame_3_gt.png", "frame_4_gt.png", "frame_5_gt.png", "frame_6_gt.png", "frame_7_gt.png", "frame_8_gt.png"]
        for d in dirs:
            if all(os.path.exists(os.path.join(d, fn)) for fn in need):
                valid.append(d)

        # train/val 切分：95%/5%（跟 vimeo 一样）
        cnt = int(len(valid) * 0.95)
        if self.mode == "train":
            self.sample_dirs = valid[:cnt]
        elif self.mode == "val":  # val 走后 5%
            self.sample_dirs = valid[cnt:]
        elif self.mode == "test":  # test 用全数据
            self.sample_dirs = valid

        # 展开成 (dir, t) 七倍数据
        self.items = []
        for d in self.sample_dirs:
            for t in T_LIST:
                self.items.append((d, t))

    def __len__(self):
        return len(self.items)

    def get_img(self, index):
        d, t = self.items[index]

        img0 = read_image(os.path.join(d, "frame_1.png"))  # uint8, [C,H,W]
        img1 = read_image(os.path.join(d, "frame_9.png"))
        imgt = read_image(os.path.join(d, GT_NAME[t]))

        # scene_names 用于日志/保存，不影响训练
        rel = os.path.relpath(d, self.data_root)
        scene_names = [
            os.path.join(rel, "frame_1.png"),
            os.path.join(rel, GT_NAME[t]),
            os.path.join(rel, "frame_9.png"),
        ]
        return img0, imgt, img1, float(t), scene_names

    def __getitem__(self, item):
        img0, imgt, img1, embt, scene_names = self.get_img(item)
        time_step = torch.Tensor([embt]).reshape(1, 1, 1)

        if self.mode == "train":
            # ====== 数据增强（对齐 VimeoSeptuplet 风格）======（MoStDSA里，只有水平、垂直、宽度旋转以及90、180、270度旋转；剪切尺寸默认是320*320）（师兄说“旋转”要谨慎）

            # resize（补充）
            # img0, imgt, img1 = fixed_resize(img0, imgt, img1, self.patch_size)  # resize 到286x286，再crop到256x256

            # 随机裁剪
            img0, imgt, img1 = random_crop(img0, imgt, img1, self.patch_size)  # 除了224x224还可以改成其他尺寸吗？

            # # 随机水平翻转
            # if random.random() > 0.5:
            #     img0, imgt, img1 = random_hor_flip(img0, imgt, img1)

            # # 随机垂直翻转
            # if random.random() > 0.5:
            #     img0, imgt, img1 = random_ver_flip(img0, imgt, img1)

            # 随机调整RGB通道的顺序
            # if img0.shape[0] == 3 and random.random() > 0.5:
            #     img0, imgt, img1 = random_color_permutation(img0, imgt, img1)

            # # 随机时间翻转
            # if random.random() > 0.5:
            #     img0, imgt, img1, time_step = random_temporal_flip(img0, imgt, img1, time_step)

            # # 随机旋转角度：0, 90, 180, 270度
            # degree = random.randint(0, 3)
            # img0, imgt, img1 = random_rotation(img0, imgt, img1, degree)

            # 凑数
            # aaaaaa=1

        elif self.mode in ["val", "test"]:
            # resize 到指定尺寸512x512做测试
            img0 = TF.resize(img0, (512, 512), antialias=True)
            imgt = TF.resize(imgt, (512, 512), antialias=True)
            img1 = TF.resize(img1, (512, 512), antialias=True)

        # 转 float32 到 [0,1]（跟 vimeo 一样在增强后做）
        input_dict = {
            "img0": TF.to_dtype(img0, torch.float32, scale=True),
            "imgt": TF.to_dtype(imgt, torch.float32, scale=True),
            "img1": TF.to_dtype(img1, torch.float32, scale=True),
            "time_step": time_step,
            "scene_names": scene_names,
        }
        return input_dict
