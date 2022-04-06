import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from expert.models import LaplacianPyramid



class NLPD_Loss(nn.Module):
    def __init__(self, stages=6, dims=3, device=None):
        super().__init__()
        self.pyr = LaplacianPyramid(stages, dims=dims)
        if device:
            self.pyr = self.pyr.to(device)
        self.pyr.eval()

    def forward(self, output, target):
        out = self.pyr.compare(output, target)
        return out


class MSE_Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mse = F.mse_loss
    
    def forward(self, output, target):
        out = F.mse_loss(output, target, reduction="sum") / (target.size(0) * 2)
        return out


def ssim(clean, noisy, normalized=True):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.] else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean = clean.cpu().detach().numpy().astype(np.float32)
    noisy = noisy.cpu().detach().numpy().astype(np.float32)
    return np.array([structural_similarity(c[0], n[0], data_range=255) for c, n in zip(clean, noisy)]).mean()


def psnr(clean, noisy, normalized=True):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean = clean.cpu().detach().numpy().astype(np.float32)
    noisy = noisy.cpu().detach().numpy().astype(np.float32)
    return np.array([peak_signal_noise_ratio(c[0], n[0], data_range=255) for c, n in zip(clean, noisy)]).mean()