import math
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def degrade(img, size):
    if not isinstance(img, Image.Image):
        img = transforms.ToPILImage()(img)
    img = transforms.Resize(size, InterpolationMode.BICUBIC)(img)
    return transforms.ToTensor()(img)


def shift_hr(x, dx, dy):
    b, _, h, w = x.shape
    ys = torch.linspace(-1, 1, h, device=x.device)
    xs = torch.linspace(-1, 1, w, device=x.device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    gx = gx + (2.0 * dx / max(w - 1, 1))
    gy = gy + (2.0 * dy / max(h - 1, 1))
    grid = torch.stack([gx, gy], -1).unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)


def crop_to_scale(hr, scale):
    lr_h = int(hr.shape[-2] // scale)
    lr_w = int(hr.shape[-1] // scale)
    return hr[..., :int(lr_h * scale), :int(lr_w * scale)], (lr_h, lr_w)


def make_frames(hr, lr_size, scale, frames=8, seed=0):
    rng = random.Random(seed * 100003 + int(math.ceil(scale)))
    out, shifts = [], []
    for k in range(frames):
        if k == 0:
            dx = dy = 0.0
        else:
            dx, dy = rng.uniform(0, scale), rng.uniform(0, scale)
        h2 = shift_hr(hr, dx, dy).clamp(0, 1)
        out.append(degrade(h2[0].cpu(), lr_size).unsqueeze(0).to(hr.device))
        shifts.append((dx * 4.0 / max(scale, 1e-9), dy * 4.0 / max(scale, 1e-9)))
    return torch.stack(out, 1), shifts


def simulate(hr, scale, frames=8, seed=0):
    hr, lr_size = crop_to_scale(hr, scale)
    lr_frames, shifts = make_frames(hr, lr_size, scale, frames, seed)
    lr = degrade(hr[0].cpu(), lr_size).unsqueeze(0).to(hr.device)
    return hr, lr, lr_frames, shifts


def bicubic(lr, size):
    return F.interpolate(lr, size=size, mode='bicubic', align_corners=False)
