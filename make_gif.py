import argparse
import os

import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from spsgs import load_image, load_model, setup, to_pil
from spsgs.frames import crop_to_scale, degrade, make_frames
from spsgs.utils import run


def font(size):
    for name in ('DejaVuSans-Bold.ttf', 'arialbd.ttf', 'Arial Bold.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def window(img, cx, cy, size):
    h, w = img.shape[-2:]
    x0 = int(round(min(max(cx - size / 2, 0), max(w - size, 0))))
    y0 = int(round(min(max(cy - size / 2, 0), max(h - size, 0))))
    return img[..., y0:y0 + size, x0:x0 + size]


def panel(img, label, size, fnt):
    p = to_pil(img)
    if p.size != (size, size):
        p = p.resize((size, size), Image.BICUBIC)
    d = ImageDraw.Draw(p)
    x0, y0, x1, y1 = d.textbbox((0, 0), label, font=fnt)
    d.rectangle((6, 6, 18 + x1 - x0, 14 + y1 - y0), fill=(0, 0, 0))
    d.text((12 - x0, 10 - y0), label, font=fnt, fill=(255, 255, 255))
    return p


def main():
    ap = argparse.ArgumentParser(description='Continuous zoom GIF: bicubic vs. SPS-GS from the same LR frames')
    ap.add_argument('-i', '--input', default='assets/examples/center_246.jpg')
    ap.add_argument('-o', '--output', default='results/demo.gif')
    ap.add_argument('-m', '--model', default='aid')
    ap.add_argument('--base', type=float, default=4.0, help='downsampling factor of the simulated LR frames')
    ap.add_argument('--max-scale', type=float, default=8.0, help='largest output scale (relative to the LR frames)')
    ap.add_argument('--center', type=float, nargs=2, default=[0.632, 0.172], help='zoom center (x, y) in [0, 1]')
    ap.add_argument('--size', type=int, default=300, help='side length of each panel in pixels')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--duration', type=int, default=90, help='ms per frame')
    ap.add_argument('--hold', type=int, default=12, help='extra frames at the start and the end')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    dev = setup(args.device)
    model = load_model(args.model, dev)
    hr, lr_size = crop_to_scale(load_image(args.input, dev), args.base)
    frames, shifts = make_frames(hr, lr_size, args.base, seed=args.seed)
    lr = degrade(hr[0].cpu(), lr_size).unsqueeze(0).to(dev)
    lh, lw = lr_size
    s0 = max(args.size / min(lh, lw), 1.0)
    scales = np.geomspace(s0, max(args.max_scale, s0), args.steps)
    cx, cy = args.center[0] * lw, args.center[1] * lh
    fnt = font(max(12, args.size // 18))
    out = []
    for s in scales:
        s = float(s)
        sr = run(model, frames, s, shifts)
        bi = F.interpolate(lr, size=sr.shape[-2:], mode='bicubic', align_corners=False)
        a = panel(window(bi, cx * s, cy * s, args.size), f'Bicubic  x{s:.1f}', args.size, fnt)
        b = panel(window(sr, cx * s, cy * s, args.size), f'SPS-GS  x{s:.1f}', args.size, fnt)
        img = Image.new('RGB', (2 * args.size + 6, args.size), (255, 255, 255))
        img.paste(a, (0, 0))
        img.paste(b, (args.size + 6, 0))
        out.append(img)
        print(f'x{s:.2f}', flush=True)
    out = [out[0]] * args.hold + out + [out[-1]] * args.hold
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out[0].save(args.output, save_all=True, append_images=out[1:], duration=args.duration, loop=0, optimize=True)
    print(f'saved {args.output} ({len(out)} frames)')


if __name__ == '__main__':
    main()
