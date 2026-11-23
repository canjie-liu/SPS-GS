import torch


def rasterize(xy, chol, color, size, tile=16, block=2, chunk=1 << 23):
    h, w = size
    dev = xy.device
    cx = 0.5 * w * xy[:, 0] + 0.5 * w
    cy = 0.5 * h * xy[:, 1] + 0.5 * h
    l11, l21, l22 = chol.unbind(1)
    a = l11 * l11
    b = l11 * l21
    c = l21 * l21 + l22 * l22
    det = a * c - b * b
    inv = 1.0 / det
    qa, qb, qc = c * inv, -b * inv, a * inv
    m = 0.5 * (a + c)
    r = torch.ceil(3.0 * torch.sqrt(m + torch.sqrt(torch.clamp(m * m - det, min=0.0))))
    ok = (det != 0) & torch.isfinite(r + cx + cy + qa + qb + qc)
    px, py, pr = torch.where(ok, cx, -1e9), torch.where(ok, cy, -1e9), torch.where(ok, r, 0.0)
    ex = 3.5 * torch.sqrt(torch.where(ok, a, 0.0))
    ey = 3.5 * torch.sqrt(torch.where(ok, c, 0.0))
    tw, th = (w + tile - 1) // tile, (h + tile - 1) // tile
    x0 = torch.maximum(torch.trunc(px / tile - pr / tile).clamp(0, tw) * tile, torch.ceil(px - ex))
    x1 = torch.minimum((torch.trunc(px / tile + pr / tile + 1).clamp(0, tw) * tile).clamp(max=w), torch.floor(px + ex) + 1)
    y0 = torch.maximum(torch.trunc(py / tile - pr / tile).clamp(0, th) * tile, torch.ceil(py - ey))
    y1 = torch.minimum((torch.trunc(py / tile + pr / tile + 1).clamp(0, th) * tile).clamp(max=h), torch.floor(py + ey) + 1)
    bx0, by0 = torch.floor(x0 / block).long(), torch.floor(y0 / block).long()
    nx = torch.where(x1 > x0, torch.floor((x1 - 1) / block).long() + 1 - bx0, torch.zeros_like(bx0))
    ny = torch.where(y1 > y0, torch.floor((y1 - 1) / block).long() + 1 - by0, torch.zeros_like(by0))
    cnt = nx * ny
    bw, bh = (w + block - 1) // block, (h + block - 1) // block
    out = color.new_zeros(bh * bw, block * block, color.shape[1])
    g = torch.nonzero(cnt > 0).squeeze(1)
    if g.numel() > 0:
        cnt = cnt[g]
        cum = torch.cumsum(cnt, 0)
        step = max(1, chunk // (block * block))
        total = int(cum[-1])
        cuts = torch.searchsorted(cum, torch.arange(step, total, step, device=dev), right=True).tolist() if total > step else []
        edges = [0] + cuts + [g.numel()]
        oy, ox = torch.meshgrid(torch.arange(block, device=dev), torch.arange(block, device=dev), indexing='ij')
        ox, oy = ox.reshape(1, -1).float(), oy.reshape(1, -1).float()
        q = torch.stack((cx, cy, qa, qb, qc), 1)
        for e0, e1 in zip(edges[:-1], edges[1:]):
            if e1 <= e0:
                continue
            cc = cnt[e0:e1]
            gi = torch.repeat_interleave(g[e0:e1], cc)
            k = torch.arange(gi.numel(), device=dev) - torch.repeat_interleave(torch.cumsum(cc, 0) - cc, cc)
            nxg = nx[gi]
            ix = bx0[gi] + k % nxg
            iy = by0[gi] + k // nxg
            p = q[gi]
            u = (ix * block).float().unsqueeze(1) + ox
            v = (iy * block).float().unsqueeze(1) + oy
            dx = p[:, 0:1] - u
            dy = p[:, 1:2] - v
            sig = 0.5 * (p[:, 2:3] * dx * dx + p[:, 4:5] * dy * dy) + p[:, 3:4] * dx * dy
            al = torch.exp(-sig)
            keep = (sig >= 0) & (al >= 1.0 / 255.0) & (u < w) & (v < h)
            al = torch.where(keep, al, torch.zeros_like(al))
            out.index_add_(0, iy * bw + ix, al.unsqueeze(2) * color[gi].unsqueeze(1))
    out = out.view(bh, bw, block, block, -1).permute(4, 0, 2, 1, 3).reshape(-1, bh * block, bw * block)
    return out[:, :h, :w]
