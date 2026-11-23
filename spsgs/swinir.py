import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def window_partition(x, ws):
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c)


def window_reverse(windows, ws, h, w):
    b = int(windows.shape[0] / (h * w / ws / ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, ws, num_heads):
        super().__init__()
        self.ws = ws
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads))
        coords = torch.stack(torch.meshgrid([torch.arange(ws), torch.arange(ws)], indexing='ij'))
        coords = torch.flatten(coords, 1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer('relative_position_index', rel.sum(-1), persistent=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.ws * self.ws, self.ws * self.ws, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    _masks = {}

    def __init__(self, dim, num_heads, ws, shift, mlp_ratio):
        super().__init__()
        self.ws = ws
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def mask(self, h, w, device):
        key = (h, w, self.ws, self.shift, str(device))
        m = SwinTransformerBlock._masks.get(key)
        if m is None:
            if len(SwinTransformerBlock._masks) >= 16:
                SwinTransformerBlock._masks.pop(next(iter(SwinTransformerBlock._masks)))
            img = torch.zeros((1, h, w, 1))
            sl = (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None))
            cnt = 0
            for a in sl:
                for b in sl:
                    img[:, a, b, :] = cnt
                    cnt += 1
            mw = window_partition(img, self.ws).view(-1, self.ws * self.ws)
            m = mw.unsqueeze(1) - mw.unsqueeze(2)
            m = m.masked_fill(m != 0, float(-100.0)).masked_fill(m == 0, float(0.0)).to(device)
            SwinTransformerBlock._masks[key] = m
        return m

    def forward(self, x, size):
        h, w = size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        win = window_partition(x, self.ws).view(-1, self.ws * self.ws, c)
        win = self.attn(win, mask=self.mask(h, w, x.device) if self.shift > 0 else None)
        x = window_reverse(win.view(-1, self.ws, self.ws, c), self.ws, h, w)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + x.view(b, h * w, c)
        return x + self.mlp(self.norm2(x))


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, ws, mlp_ratio):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, ws, 0 if i % 2 == 0 else ws // 2, mlp_ratio) for i in range(depth)])

    def forward(self, x, size):
        for blk in self.blocks:
            x = blk(x, size)
        return x


class RSTB(nn.Module):
    def __init__(self, dim, depth, num_heads, ws, mlp_ratio):
        super().__init__()
        self.dim = dim
        self.residual_group = BasicLayer(dim, depth, num_heads, ws, mlp_ratio)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, size):
        y = self.residual_group(x, size)
        y = y.transpose(1, 2).view(x.shape[0], self.dim, size[0], size[1])
        return self.conv(y).flatten(2).transpose(1, 2) + x


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.flatten(2).transpose(1, 2))


class SwinIR(nn.Module):
    def __init__(self, in_chans=3, embed_dim=180, depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6),
                 window_size=8, mlp_ratio=2.0, num_feat=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(embed_dim)
        self.layers = nn.ModuleList([RSTB(embed_dim, d, h, window_size, mlp_ratio) for d, h in zip(depths, num_heads)])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = nn.Sequential(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2),
                                      nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2))

    def forward_features(self, x):
        size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, size)
        x = self.norm(x)
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, size[0], size[1])

    def forward(self, x):
        _, _, h, w = x.shape
        ph = (self.window_size - h % self.window_size) % self.window_size
        pw = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, pw, 0, ph), 'reflect')
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.upsample(self.conv_before_upsample(x))
        return x[:, :, :4 * h, :4 * w]
