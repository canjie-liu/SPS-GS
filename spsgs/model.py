import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import down_aa, fourier_shift, gaussian_grid, haar_dwt, shift_each, shift_frames
from .render import rasterize
from .swinir import SwinIR


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        return self.layers(x.contiguous().view(-1, x.shape[-1])).view(*shape, -1)


class EarlyFusion(nn.Conv2d):
    def __init__(self, frames, embed_dim, hidden=128, n_map=5, n_dm=12):
        super().__init__(3 * frames, embed_dim, 3, 1, 1)
        self.k = frames
        self.n_map = n_map
        self.n_dm = n_dm
        self.body = nn.Sequential(nn.Conv2d(6 + n_map, hidden, 3, 1, 1), nn.GELU(),
                                  nn.Conv2d(hidden, hidden, 3, 1, 1), nn.GELU())
        self.head = nn.Conv2d(hidden, embed_dim, 3, 1, 1)
        self.dm = nn.Conv2d(n_dm, hidden, 3, 1, 1)

    def forward(self, x):
        n, nl = self.k - 1, 3 * self.k
        b, _, h, w = x.shape
        nq = nl + n * self.n_map
        ref = x[:, :3].unsqueeze(1).expand(b, n, 3, h, w)
        dif = x[:, 3:nl].reshape(b, n, 3, h, w)
        mp = x[:, nl:nq].reshape(b, n, self.n_map, h, w)
        z = self.body(torch.cat([ref, dif, mp], 2).reshape(b * n, 6 + self.n_map, h, w))
        z = (z.reshape(b, n, -1, h, w) * mp[:, :, -1:]).sum(1)
        z = z + self.dm(x[:, nq:nq + self.n_dm])
        return super().forward(x[:, :nl]) + self.head(z)


class Step(nn.Module):
    def __init__(self, cin, hidden, cout):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, hidden, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                                  nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                                  nn.Conv2d(hidden, cout, 3, padding=1))

    def forward(self, x):
        return self.body(x)


class SPSGS(nn.Module):
    def __init__(self, frames=8, steps=7, gain=10.0, ridge=0.1):
        super().__init__()
        self.frames = frames
        self.gain = gain
        self.ridge = ridge
        self.encoder = SwinIR()
        self.encoder.conv_first = EarlyFusion(frames, self.encoder.embed_dim)
        self.unshuffle = nn.PixelUnshuffle(2)
        c = 256
        self.color_head = MLP(c, 3, [256, 128, 64])
        self.offset_head = MLP(c, 2, [256, 128, 64])
        self.cov_head = nn.Conv2d(c, 3, 1)
        self.step_in = nn.Conv2d(c, 32, 1)
        self.steps = nn.ModuleList([Step(32 + 12 + 3 + 2, 64, 5) for _ in range(steps)])

    def demodulate(self, d, shifts, b, h, w):
        n, nv = self.frames - 1, len(shifts)
        rows = []
        for dd in shifts:
            ax, ay = 2.0 * math.pi * float(dd[0]) / 4.0, 2.0 * math.pi * float(dd[1]) / 4.0
            r = []
            for mx, my in ((1, 0), (0, 1)):
                t = mx * ax + my * ay
                r += [math.cos(t) - 1.0, math.sin(t)]
            rows.append(r)
        m = torch.tensor(rows, dtype=torch.float64)
        g = m.T @ m
        lam = self.ridge * float(g.diagonal().mean()) + 1e-6
        a = torch.linalg.solve(g + lam * torch.eye(4, dtype=torch.float64), m.T)
        if nv < n:
            a = torch.cat([a, a.new_zeros(4, n - nv)], 1)
        a = a.to(device=d.device, dtype=d.dtype)
        return torch.einsum('jk,bkchw->bjchw', a, d.reshape(b, n, 3, h, w)).reshape(b, 12, h, w)

    def fusion_input(self, frames, shifts):
        ref = frames[:, 0]
        b, _, h, w = ref.shape
        n = self.frames - 1
        k = min(frames.shape[1], self.frames)
        back = [(-float(d[0]) / 4.0, -float(d[1]) / 4.0) for d in shifts[1:k]]
        oth = frames[:, 1:k].transpose(0, 1).reshape((k - 1) * b, 3, h, w)
        al = fourier_shift(oth, back).reshape(k - 1, b, 3, h, w).transpose(0, 1).reshape(b, 3 * (k - 1), h, w)
        d = (al - ref.repeat(1, k - 1, 1, 1)) * self.gain
        if k - 1 < n:
            d = torch.cat([d, ref.new_zeros(b, 3 * (n - k + 1), h, w)], 1)
        v = []
        for dd in shifts[1:k]:
            ax, ay = 2.0 * math.pi * float(dd[0]) / 4.0, 2.0 * math.pi * float(dd[1]) / 4.0
            v += [math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), 1.0]
        nm = self.encoder.conv_first.n_map * n
        v += [0.0] * (nm - len(v))
        mp = torch.tensor(v, device=ref.device, dtype=ref.dtype).view(1, nm, 1, 1).expand(b, nm, h, w)
        return torch.cat([ref, d, mp, self.demodulate(d, shifts[1:k], b, h, w)], 1)

    def gaussians(self, feat):
        b, _, fh, fw = feat.shape
        ng = fh * fw
        pc = F.leaky_relu(feat, 0.01)
        x = feat.reshape(b, -1, ng).permute(1, 0, 2).reshape(-1, b * ng).permute(1, 0)
        p = pc.reshape(b, -1, ng).permute(1, 0, 2).reshape(-1, b * ng).permute(1, 0)
        color = self.color_head(x).reshape(b, ng, -1)
        offset = self.offset_head(p).reshape(b, ng, -1)
        z = self.cov_head(pc).reshape(b, 3, ng).permute(0, 2, 1)
        chol = torch.stack((F.softplus(z[..., 0]), z[..., 1], F.softplus(z[..., 2])), -1)
        return color, offset, chol

    def render(self, color, offset, chol, size, scale, lr_size, grid_size):
        hh, ww = size
        s1, s2 = scale
        lr_h, lr_w = lr_size
        base = gaussian_grid(grid_size[0], grid_size[1], color.device)
        out = []
        for i in range(color.shape[0]):
            o = offset[i]
            xy = torch.cat((base[:, 0:1] + 2 * o[:, 0:1] / lr_w - 1 / ww,
                            base[:, 1:2] + 2 * o[:, 1:2] / lr_h - 1 / hh), dim=1)
            wc = chol[i] / 4
            wc = torch.stack((wc[:, 0] * s2, wc[:, 1] * s1, wc[:, 2] * s1), dim=1)
            out.append(rasterize(xy.contiguous(), wc.contiguous(), color[i].contiguous(), (hh, ww)).unsqueeze(0))
        return torch.cat(out)

    def forward(self, frames, scale, shifts, size=None):
        s1 = s2 = float(torch.tensor(float(scale), dtype=torch.float32))
        feat = self.unshuffle(self.encoder(self.fusion_input(frames, shifts)))
        lr_h, lr_w = int(frames.shape[-2]), int(frames.shape[-1])
        size = (round(lr_h * s1), round(lr_w * s2)) if size is None else (int(size[0]), int(size[1]))
        grid_size = (feat.shape[-2], feat.shape[-1])
        fh, fw = grid_size
        k = frames.shape[1]
        sh = [(float(d[0]) * s2 / 4.0, float(d[1]) * s1 / 4.0) for d in shifts[:k]]
        back = [(-dx, -dy) for dx, dy in sh]
        c, o, chol = self.gaussians(feat)
        b = c.shape[0]
        fz = self.step_in(feat)
        frk = frames[:, :k].transpose(0, 1).reshape(k * b, *frames.shape[2:])
        for step in self.steps:
            x = self.render(c, torch.tanh(o), chol, size, (s1, s2), (lr_h, lr_w), grid_size)
            res = frk - down_aa(shift_frames(x, sh), (lr_h, lr_w), 'bicubic')
            rk = F.interpolate(res, size=size, mode='bicubic', align_corners=False)
            rk = shift_each(rk, back).reshape(k, b, 3, size[0], size[1])
            e = rk[0]
            for j in range(1, k):
                e = e + rk[j]
            e = e / k
            eg = down_aa(e, (fh, fw), 'bilinear')
            _, lh, hl, hh = haar_dwt(down_aa(e, (2 * fh, 2 * fw), 'bilinear'))
            cg = c.permute(0, 2, 1).reshape(b, 3, fh, fw)
            og = o.permute(0, 2, 1).reshape(b, 2, fh, fw)
            d = step(torch.cat([fz, eg * self.gain, lh * self.gain, hl * self.gain, hh * self.gain, cg, og], 1))
            c = c + d[:, :3].reshape(b, 3, fh * fw).permute(0, 2, 1)
            o = o + d[:, 3:5].reshape(b, 2, fh * fw).permute(0, 2, 1)
        return self.render(c, torch.tanh(o), chol, size, (s1, s2), (lr_h, lr_w), grid_size)
