import numpy as np
import math
import cv2
import torch
import torch.nn.functional as F
from math import exp
from stlpips_pytorch import stlpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from basicsr.metrics.niqe import calculate_niqe
import lpips

try:
    from skimage.filters import frangi, threshold_otsu
except Exception:
    frangi = None
    threshold_otsu = None


def calculate_batch_psnr(gt_tensor, output_tensor, mode='avg'):
    # both parameters are in the form of tensor of size: BS, C, H, W

    if mode == 'avg':
        gt_np = gt_tensor.cpu().numpy().astype(np.float32)
        output_np = output_tensor.cpu().numpy().astype(np.float32)

        bs = gt_np.shape[0]
        psnr_list = []
        psnr = 0
        for i in range(bs):
            gt_im = gt_np[i, :, :, :]
            output_im = output_np[i, :, :, :]

            gt_im = gt_im.transpose((1, 2, 0))
            output_im = output_im.transpose((1, 2, 0))
            output_im = np.round(output_im * 255) / 255.
            # output_im = np.round(output_im * 255).astype('uint8') / 255
            mse = float(((gt_im - output_im) ** 2).mean())
            psnr_it = -10 * np.log10(max(mse, 1e-12))
            psnr_list.append(psnr_it)
            psnr += psnr_it

            # psnr_list.append(-10 * math.log10(((gt_im - output_im) * (gt_im - output_im)).mean()))
            # psnr += -10 * math.log10(((gt_im - output_im) * (gt_im - output_im)).mean())
            # psnr_list.append(peak_signal_noise_ratio(gt_im, output_im, data_range=1.))
            # psnr += peak_signal_noise_ratio(gt_im, output_im, data_range=1.)
        return float(psnr / bs), psnr_list
    else:
        raise NotImplementedError


def calculate_batch_ssim(gt_tensor, output_tensor, mode='avg', return_list=False):
    if mode == 'avg':
        output_tensor = torch.round(output_tensor * 255) / 255.

        bs = gt_tensor.shape[0]
        ssim_list = []
        # for i in range(bs):
        #     gt_im = gt_np[i, :, :, :]
        #     output_im = output_np[i, :, :, :]
        #     gt_im = gt_im.transpose((1, 2, 0))
        #     output_im = output_im.transpose((1, 2, 0))
        #     output_np = np.round(output_im * 255) / 255.
        #
        #     ssim += ssim_matlab(gt_im, output_im)
        for i in range(bs):
            ssim_list.append(float(ssim_matlab(gt_tensor[i:i+1], output_tensor[i:i+1]).detach().cpu()))

        ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
        if return_list:
            return ssim, ssim_list
        return ssim, bs
    else:
        raise NotImplementedError


def calculate_batch_lpips(gt_tensor, output_tensor, lpips_metric, mode='avg', return_list=False):
    if mode == 'avg':
        gt_tensor = gt_tensor * 2 - 1
        output_tensor = output_tensor * 2 - 1

        bs = gt_tensor.shape[0]
        lpips_list = []
        for i in range(bs):
            score = lpips_metric(gt_tensor[i:i+1], output_tensor[i:i+1])
            lpips_list.append(float(score.detach().cpu()))

        lpips_value = float(np.mean(lpips_list)) if lpips_list else 0.0
        if return_list:
            return lpips_value, lpips_list
        return lpips_value, bs
    else:
        raise NotImplementedError


def calculate_batch_stlpips(gt_tensor, output_tensor, stlpips_metric, mode='avg', return_list=False):
    if mode == 'avg':
        gt_tensor = gt_tensor * 2 - 1
        output_tensor = output_tensor * 2 - 1

        bs = gt_tensor.shape[0]
        stlpips_list = []
        for i in range(bs):
            score = stlpips_metric(gt_tensor[i:i+1], output_tensor[i:i+1])
            stlpips_list.append(float(score.detach().cpu()))

        stlpips_value = float(np.mean(stlpips_list)) if stlpips_list else 0.0
        if return_list:
            return stlpips_value, stlpips_list
        return stlpips_value, bs
    else:
        raise NotImplementedError


def calculate_batch_niqe(gt_tensor, output_tensor, niqe_metric, mode='avg'):
    if mode == 'avg':
        bs = gt_tensor.shape[0]
        niqe_sum = 0
        for i in range(bs):
            gt_np = (gt_tensor[i].cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
            output_np = (output_tensor[i].cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
            niqe_sum += calculate_niqe(output_np, crop_border=0)

        return float(niqe_sum), 1
    else:
        raise NotImplementedError


def _tensor_image_to_gray_np(image):
    if image.shape[0] == 1:
        return image[0].astype(np.float32)
    return image.astype(np.float32).mean(axis=0)


def _normalize_np(image, eps=1e-12):
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value <= eps:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def _threshold_score_map(score_map, cfg):
    threshold_mode = str(cfg.get("threshold_mode", cfg.get("threshold", "percentile"))).lower()
    if threshold_mode == "otsu":
        if threshold_otsu is None:
            raise ImportError("skimage.filters.threshold_otsu is required for vessel threshold_mode='otsu'.")
        return float(threshold_otsu(score_map))
    if threshold_mode == "absolute":
        return float(cfg.get("threshold_value", 0.5))

    percentile = float(cfg.get("threshold_percentile", cfg.get("percentile", 85.0)))
    percentile = min(100.0, max(0.0, percentile))
    return float(np.percentile(score_map, percentile))


def _fallback_topk_mask(score_map, min_pixels):
    flat = score_map.reshape(-1)
    if flat.size == 0:
        return np.zeros_like(score_map, dtype=bool)
    k = min(max(int(min_pixels), 1), flat.size)
    topk_indices = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(flat.shape, dtype=bool)
    mask[topk_indices] = True
    return mask.reshape(score_map.shape)


def build_vessel_mask_from_gt(gt_gray, vessel_cfg=None):
    cfg = {} if vessel_cfg is None else vessel_cfg
    method = str(cfg.get("method", "frangi")).lower()
    polarity = str(cfg.get("polarity", "bright")).lower()
    gt_norm = _normalize_np(gt_gray)

    if method == "frangi":
        if frangi is None:
            raise ImportError("skimage.filters.frangi is required for vessel method='frangi'.")
        sigmas = cfg.get("sigmas", (1.0, 2.0, 3.0))
        if polarity == "dark":
            score_map = frangi(gt_norm, sigmas=sigmas, black_ridges=True)
        elif polarity == "both":
            bright_score = frangi(gt_norm, sigmas=sigmas, black_ridges=False)
            dark_score = frangi(gt_norm, sigmas=sigmas, black_ridges=True)
            score_map = np.maximum(bright_score, dark_score)
        else:
            score_map = frangi(gt_norm, sigmas=sigmas, black_ridges=False)
    elif method in {"intensity", "threshold"}:
        if polarity == "dark":
            score_map = 1.0 - gt_norm
        elif polarity == "both":
            score_map = np.abs(gt_norm - float(np.median(gt_norm)))
        else:
            score_map = gt_norm
    else:
        raise ValueError(f"Unsupported vessel mask method: {method}")

    score_map = _normalize_np(score_map)
    threshold_value = _threshold_score_map(score_map, cfg)
    mask = score_map >= threshold_value

    dilation_radius = int(cfg.get("dilation_radius", cfg.get("dilate_radius", 0)))
    if dilation_radius > 0 and mask.any():
        kernel_size = dilation_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    min_pixels = int(cfg.get("min_pixels", 16))
    if int(mask.sum()) < min_pixels:
        empty_policy = str(cfg.get("empty_policy", "fallback_topk")).lower()
        if empty_policy == "fallback_topk":
            mask = _fallback_topk_mask(score_map, min_pixels)
        elif empty_policy == "full":
            mask = np.ones_like(score_map, dtype=bool)
        else:
            mask = np.zeros_like(score_map, dtype=bool)

    return mask


def calculate_batch_vessel_psnr(gt_tensor, output_tensor, vessel_cfg=None, mode='avg'):
    if mode != 'avg':
        raise NotImplementedError

    cfg = {} if vessel_cfg is None else vessel_cfg
    eps = float(cfg.get("eps", 1e-12))
    quantize_output = bool(cfg.get("quantize_output", True))
    gt_np = gt_tensor.detach().cpu().numpy().astype(np.float32)
    output_np = output_tensor.detach().cpu().numpy().astype(np.float32)

    if quantize_output:
        output_np = np.round(output_np * 255.0) / 255.0

    vessel_psnr_list = []
    for i in range(gt_np.shape[0]):
        gt_gray = _tensor_image_to_gray_np(gt_np[i])
        output_gray = _tensor_image_to_gray_np(output_np[i])
        mask = build_vessel_mask_from_gt(gt_gray, cfg)
        if not mask.any():
            continue
        mse = float(((gt_gray - output_gray) ** 2)[mask].mean())
        vessel_psnr_list.append(float(-10.0 * np.log10(max(mse, eps))))

    if not vessel_psnr_list:
        return 0.0, []
    return float(np.mean(vessel_psnr_list)), vessel_psnr_list


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).cuda()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().cuda()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    # mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    # mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1 = F.conv2d(F.pad(img1, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)
    mu2 = F.conv2d(F.pad(img2, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    # sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    # sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    sigma1_sq = F.conv2d(F.pad(img1 * img1, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(img2 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(F.pad(img1 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def ssim_matlab(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=1):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.shape
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 3 channel for SSIM
        self.channel = 3
        self.window = create_window(window_size, channel=self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        _ssim = ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        dssim = (1 - _ssim) / 2
        return dssim


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
