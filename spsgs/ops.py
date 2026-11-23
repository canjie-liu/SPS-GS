import math

import torch
import torch.nn.functional as F


def _grid(shifts, h, w, device, dtype):
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([torch.stack([gx + 2.0 * dx / max(w - 1, 1), gy + 2.0 * dy / max(h - 1, 1)], -1)
                        for dx, dy in shifts])


def shift_frames(x, shifts):
    b, c, h, w = x.shape
    g = _grid(shifts, h, w, x.device, x.dtype)
    k = g.shape[0]
    g = g.unsqueeze(1).expand(k, b, h, w, 2).reshape(k * b, h, w, 2)
    xr = x.unsqueeze(0).expand(k, b, c, h, w).reshape(k * b, c, h, w)
    return F.grid_sample(xr, g, mode='bilinear', padding_mode='reflection', align_corners=True)


def shift_each(x, shifts, mode='bilinear'):
    kb, c, h, w = x.shape
    k = len(shifts)
    g = _grid(shifts, h, w, x.device, x.dtype)
    g = g.unsqueeze(1).expand(k, kb // k, h, w, 2).reshape(kb, h, w, 2)
    return F.grid_sample(x, g, mode=mode, padding_mode='reflection', align_corners=True)


_FFT_OK = {}
_DCT_CACHE = {}


def _fft_available(device):
    key = str(device)
    if key not in _FFT_OK:
        try:
            torch.fft.rfft2(torch.zeros(1, 1, 8, 8, device=device))
            _FFT_OK[key] = True
        except RuntimeError:
            _FFT_OK[key] = False
    return _FFT_OK[key]


def _dct_shift_mats(n, ds, device):
    key = (n, tuple(float(v) for v in ds), str(device))
    a = _DCT_CACHE.get(key)
    if a is None:
        if len(_DCT_CACHE) >= 16:
            _DCT_CACHE.clear()
        k = torch.arange(n, device=device, dtype=torch.float64)
        j = k + 0.5
        cf = torch.cos((math.pi / n) * k[:, None] * j[None, :]).float()
        wk = torch.full((n,), 2.0 / n, device=device, dtype=torch.float64)
        wk[0] = 1.0 / n
        d = torch.tensor(list(key[1]), device=device, dtype=torch.float64).view(-1, 1, 1)
        ci = (torch.cos((math.pi / n) * (j.view(1, n, 1) + d) * k.view(1, 1, n)) * wk).float()
        a = _DCT_CACHE[key] = ci @ cf
    return a


def _fourier_shift_mm(x, shifts):
    kb, c, h, w = x.shape
    k = len(shifts)
    tf = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast('cuda', enabled=False):
            ay = _dct_shift_mats(h, [b for _, b in shifts], x.device)
            ax = _dct_shift_mats(w, [a for a, _ in shifts], x.device)
            xv = x.float().reshape(k, (kb // k) * c, h, w)
            y = ay.unsqueeze(1) @ xv @ ax.transpose(1, 2).unsqueeze(1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf
    return y.reshape(kb, c, h, w).to(x.dtype)


def fourier_shift(x, shifts):
    if x.is_cuda and not _fft_available(x.device):
        return _fourier_shift_mm(x, shifts)
    kb, c, h, w = x.shape
    k = len(shifts)
    xs = torch.cat([x, x.flip(-1)], -1)
    xs = torch.cat([xs, xs.flip(-2)], -2)
    hp, wp = 2 * h, 2 * w
    spec = torch.fft.rfft2(xs)
    fy = torch.fft.fftfreq(hp, device=x.device, dtype=x.dtype).view(1, hp, 1)
    fx = torch.fft.rfftfreq(wp, device=x.device, dtype=x.dtype).view(1, 1, wp // 2 + 1)
    d = torch.tensor([[float(a), float(b)] for a, b in shifts], device=x.device, dtype=x.dtype)
    ph = 2 * math.pi * (fx * d[:, 0].view(k, 1, 1) + fy * d[:, 1].view(k, 1, 1))
    p = torch.polar(torch.ones_like(ph), ph)
    p = p.unsqueeze(1).expand(k, kb // k, hp, wp // 2 + 1).reshape(kb, 1, hp, wp // 2 + 1)
    return torch.fft.irfft2(spec * p, s=(hp, wp))[..., :h, :w]


def down_aa(x, size, mode, max_f=40.0):
    f = max(x.shape[-2] / size[0], x.shape[-1] / size[1])
    if f <= max_f:
        return F.interpolate(x, size=size, mode=mode, align_corners=False, antialias=True)
    r = f ** 0.5
    mid = (max(int(round(x.shape[-2] / r)), size[0]), max(int(round(x.shape[-1] / r)), size[1]))
    x = F.interpolate(x, size=mid, mode=mode, align_corners=False, antialias=True)
    return F.interpolate(x, size=size, mode=mode, align_corners=False, antialias=True)


def haar_dwt(x):
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (a + b + c + d) / 4, (a - b + c - d) / 4, (a + b - c - d) / 4, (a - b - c + d) / 4


_COORDS = {}


def gaussian_grid(fh, fw, device):
    key = (fh, fw, str(device))
    t = _COORDS.get(key)
    if t is None:
        if len(_COORDS) >= 32:
            _COORDS.pop(next(iter(_COORDS)))
        xg, yg = torch.meshgrid(torch.arange(fh), torch.arange(fw), indexing='ij')
        xg = 2 * (xg / fh) - 1
        yg = 2 * (yg / fw) - 1
        t = _COORDS[key] = torch.stack((yg, xg), dim=-1).reshape(-1, 2).to(device)
    return t
