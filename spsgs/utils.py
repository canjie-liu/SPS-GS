import os

import torch
from PIL import Image
from torchvision import transforms

from .frames import bicubic, simulate
from .metrics import psnr_ssim
from .model import SPSGS

WEIGHTS = {
    'aid': 'weights/SPS-GS_AID.pth',
    'ucmerced': 'weights/SPS-GS_UCMerced.pth',
    'whu-rs19': 'weights/SPS-GS_WHU-RS19.pth',
    'rsscn7': 'weights/SPS-GS_RSSCN7.pth',
    'nwpu-resisc45': 'weights/SPS-GS_NWPU-RESISC45.pth',
    'div2k': 'weights/SPS-GS_DIV2K.pth',
}


def setup(device='cuda'):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        print('CUDA is not available, running on the CPU (slow).')
        device = 'cpu'
    return torch.device(device)


def weight_path(name):
    return WEIGHTS.get(str(name).lower(), name)


def available_models():
    return [k for k, v in WEIGHTS.items() if os.path.isfile(v)]


def load_model(name='aid', device='cuda'):
    path = weight_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found. Download the pretrained weights into weights/ (see README.md).')
    model = SPSGS().to(device).eval()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return model


def load_image(path, device='cuda'):
    return transforms.ToTensor()(Image.open(path).convert('RGB')).unsqueeze(0).to(device)


def pil_to_tensor(img, device='cuda'):
    return transforms.ToTensor()(img.convert('RGB')).unsqueeze(0).to(device)


def to_pil(x):
    a = (x[0].clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def _starts(n, t, ov):
    if n <= t:
        return [0]
    v = list(range(0, n - t + 1, t - 2 * ov))
    if v[-1] + t < n:
        v.append(n - t)
    return v


def _tiled(model, frames, scale, shifts, t=384, ov=32):
    si = int(round(scale))
    h, w = frames.shape[-2:]
    out = None
    for y0 in _starts(h, t, ov):
        for x0 in _starts(w, t, ov):
            y1, x1 = min(y0 + t, h), min(x0 + t, w)
            o = model(frames[..., y0:y1, x0:x1].contiguous(), scale, shifts)
            if out is None:
                out = o.new_zeros(o.shape[0], o.shape[1], h * si, w * si)
            vy0, vy1 = y0 + (ov if y0 > 0 else 0), y1 - (ov if y1 < h else 0)
            vx0, vx1 = x0 + (ov if x0 > 0 else 0), x1 - (ov if x1 < w else 0)
            out[..., vy0 * si:vy1 * si, vx0 * si:vx1 * si] = \
                o[..., (vy0 - y0) * si:(vy1 - y0) * si, (vx0 - x0) * si:(vx1 - x0) * si]
    return out


@torch.no_grad()
def run(model, frames, scale, shifts, tile=360000, size=None):
    h, w = frames.shape[-2:]
    if tile and h * w > tile and abs(scale - round(scale)) < 1e-9:
        return _tiled(model, frames, scale, shifts)
    return model(frames, scale, shifts, size)


@torch.no_grad()
def super_resolve(model, hr, scale, frames=8, seed=0, tile=360000):
    hr, lr, lr_frames, shifts = simulate(hr, scale, frames, seed)
    h, w = hr.shape[-2:]
    sr = run(model, lr_frames, scale, shifts, tile, (h, w))[..., :h, :w]
    bi = bicubic(lr, (h, w))
    hc, wc = min(h, sr.shape[-2], bi.shape[-2]), min(w, sr.shape[-1], bi.shape[-1])
    hr, sr, bi = hr[..., :hc, :wc], sr[..., :hc, :wc], bi[..., :hc, :wc]
    return dict(hr=hr, lr=lr, frames=lr_frames, shifts=shifts, bicubic=bi, sr=sr,
                metrics=dict(bicubic=psnr_ssim(bi, hr, scale), sr=psnr_ssim(sr, hr, scale)))
