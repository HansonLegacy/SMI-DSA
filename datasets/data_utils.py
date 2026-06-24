import cv2
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
import torch
import numpy as np
import random
import torch.nn.functional as F

perm = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
rotate = [90, 180, 270]


def random_crop(img0, imgt, img1, crop_size):
    im_h, im_w = img0.shape[-2:]
    crop_h, crop_w = crop_size, crop_size
    i = random.randint(0, im_h - crop_h)
    j = random.randint(0, im_w - crop_w)
    img0 = img0[:, i:i + crop_h, j:j + crop_w]
    imgt = imgt[:, i:i + crop_h, j:j + crop_w]
    img1 = img1[:, i:i + crop_h, j:j + crop_w]
    return img0, imgt, img1


def random_hor_flip(img0, imgt, img1):
    img0, imgt, img1 = TF.horizontal_flip(img0), TF.horizontal_flip(imgt), TF.horizontal_flip(img1)
    return img0, imgt, img1


def random_ver_flip(img0, imgt, img1):
    img0, imgt, img1 = TF.vertical_flip(img0), TF.vertical_flip(imgt), TF.vertical_flip(img1)
    return img0, imgt, img1


def random_color_permutation(img0, imgt, img1):
    perm_idx = random.randint(0, 5)
    img0, imgt, img1 = TF.permute_channels(img0, perm[perm_idx]), TF.permute_channels(imgt, perm[perm_idx]), TF.permute_channels(img1, perm[perm_idx])
    return img0, imgt, img1


def random_temporal_flip(img0, imgt, img1, time_step):
    tmp = img1
    img1 = img0
    img0 = tmp
    time_step = 1 - time_step
    return img0, imgt, img1, time_step


def random_rotation(img0, imgt, img1, degree):
    if degree != 3:
        img0 = TF.rotate(img0, rotate[degree])
        imgt = TF.rotate(imgt, rotate[degree])
        img1 = TF.rotate(img1, rotate[degree])
    return img0, imgt, img1


def random_resize(img0, imgt, img1):
    h, w = img0.shape[-2:]
    img0 = TF.resize(img0, [2*h, 2*w])
    imgt = TF.resize(imgt, [2*h, 2*w])
    img1 = TF.resize(img1, [2*h, 2*w])
    return img0, imgt, img1


def read_flow(name):
    with open(name, "rb") as f:
        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def fixed_resize(img0, imgt, img1, target_size):  # 补充，不crop而是resize；和random_crop形式对齐
    # target_size 可以是 (224, 224)
    # 输入通常是 [C, H, W]，interpolate 需要 [B, C, H, W]
    # 所以先 unsqueeze(0) 增加 batch 维度，处理完再 squeeze(0)
    
    def _resize(img):
        img = img.unsqueeze(0) 
        img = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
        return img.squeeze(0)

    img0 = _resize(img0)
    imgt = _resize(imgt)
    img1 = _resize(img1)
    
    return img0, imgt, img1


def pad_img_to_square_1024(img):
    _, h, w = img.shape
    if h == 1024 and w == 1024:
        return img

    if max(h, w) > 1024:
        scale = 1024.0 / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = TF.resize(img, [new_h, new_w], antialias=True)

    pad_h = (1024 - img.shape[-2]) // 2
    pad_w = (1024 - img.shape[-1]) // 2
    img = F.pad(img, [pad_w, pad_w, pad_h, pad_h], mode='constant', value=0)

    return img