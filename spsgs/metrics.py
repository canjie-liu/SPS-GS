import math

import torch
import torch.nn.functional as F

_Y = (65.481, 128.553, 24.966)


def _to_y(x):
    w = torch.tensor(_Y, device=x.device, dtype=torch.float64).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True) + 16.0


def _window(device):
    ax = torch.arange(11, device=device, dtype=torch.float64) - 5
    g = torch.exp(-ax ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, 11, 11)


def _ssim(a, b):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    w = _window(a.device)
    mu1, mu2 = F.conv2d(a, w), F.conv2d(b, w)
    m11, m22, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s11 = F.conv2d(a * a, w) - m11
    s22 = F.conv2d(b * b, w) - m22
    s12 = F.conv2d(a * b, w) - m12
    return float((((2 * m12 + c1) * (2 * s12 + c2)) / ((m11 + m22 + c1) * (s11 + s22 + c2))).mean())


def psnr_ssim(sr, hr, scale):
    cb = int(math.ceil(scale))
    sr = sr.clamp(0, 1).double() * 255.0
    hr = hr.clamp(0, 1).double() * 255.0
    if cb:
        sr, hr = sr[:, :, cb:-cb, cb:-cb], hr[:, :, cb:-cb, cb:-cb]
    psnr = float(10 * torch.log10(255.0 ** 2 / ((sr - hr) ** 2).mean().clamp(min=1e-12)))
    return psnr, _ssim(_to_y(sr / 255.0), _to_y(hr / 255.0))
