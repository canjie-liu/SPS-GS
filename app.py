import argparse
import glob
import os
import tempfile

import gradio as gr
from PIL import Image

from spsgs import available_models, load_model, setup, super_resolve, to_pil
from spsgs.utils import pil_to_tensor

try:
    from gradio_imageslider import ImageSlider
except Exception:
    ImageSlider = None

MAX_SIDE = 1024
EXAMPLE_SCALES = [4.0, 8.0, 2.5]
NAMES = {'aid': 'AID', 'ucmerced': 'UCMerced', 'whu-rs19': 'WHU-RS19', 'rsscn7': 'RSSCN7',
         'nwpu-resisc45': 'NWPU-RESISC45', 'div2k': 'DIV2K (natural images)'}
DEV = None
_cache = {}

DESCRIPTION = """
# SPS-GS: arbitrary-scale super-resolution from sub-pixel-shifted frames

Upload one image. It is treated as the high-resolution scene: eight low-resolution frames with random sub-pixel
shifts are simulated from it (sub-pixel shift, then downsampling by the chosen scale factor), and SPS-GS encodes the
frames once and reconstructs the image at the original resolution with 2D Gaussian splatting. Bicubic interpolation of
the reference frame and the original image are shown for comparison. Images larger than 1024 px are resized first.
"""


def get_model(name):
    if name not in _cache:
        _cache.clear()
        _cache[name] = load_model(name, DEV)
    return _cache[name]


def predict(img, name, scale, seed):
    if img is None:
        raise gr.Error('Please upload an image.')
    img = img.convert('RGB')
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        r = MAX_SIDE / max(w, h)
        img = img.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
    scale = float(scale)
    if min(img.size) // scale < 8:
        raise gr.Error('The image is too small for this scale factor.')
    r = super_resolve(get_model(name), pil_to_tensor(img, DEV), scale, seed=int(seed or 0))
    bi, sr, gt = to_pil(r['bicubic']), to_pil(r['sr']), to_pil(r['hr'])
    lr = to_pil(r['lr']).resize(gt.size, Image.NEAREST)
    (pb, sb), (ps, ss) = r['metrics']['bicubic'], r['metrics']['sr']
    info = (f'| | PSNR (dB) | SSIM |\n|---|---|---|\n| Bicubic | {pb:.2f} | {sb:.4f} |\n| SPS-GS | {ps:.2f} | {ss:.4f} |\n\n'
            f'LR frames: 8 x {r["lr"].shape[-1]}x{r["lr"].shape[-2]} px, output: {sr.width}x{sr.height} px (x{scale:g})')
    path = os.path.join(tempfile.mkdtemp(), f'SPS-GS_x{scale:g}.png')
    sr.save(path)
    pair = (bi, sr) if ImageSlider is not None else [(bi, 'Bicubic'), (sr, 'SPS-GS')]
    return pair, lr, gt, info, path


def build():
    models = available_models()
    if not models:
        raise SystemExit('No weights found in weights/. Download the pretrained weights first (see README.md).')
    examples = sorted(glob.glob(os.path.join('assets', 'examples', '*')))
    with gr.Blocks(title='SPS-GS') as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type='pil', label='Input image (treated as the high-resolution scene)')
                model = gr.Dropdown([(NAMES.get(m, m), m) for m in models], value=models[0], label='Model')
                scale = gr.Slider(2.0, 16.0, value=4.0, step=0.1, label='Scale factor')
                seed = gr.Number(value=0, precision=0, label='Seed of the sub-pixel shifts')
                btn = gr.Button('Super-resolve', variant='primary')
                if examples:
                    gr.Examples([[e, models[0], EXAMPLE_SCALES[i % len(EXAMPLE_SCALES)], 0] for i, e in enumerate(examples)],
                                [inp, model, scale, seed])
            with gr.Column(scale=1):
                if ImageSlider is not None:
                    cmp = ImageSlider(label='Bicubic (left) | SPS-GS (right), drag to compare', type='pil')
                else:
                    cmp = gr.Gallery(label='Bicubic | SPS-GS', columns=2)
                with gr.Row():
                    lr = gr.Image(type='pil', label='LR reference frame (nearest)')
                    gt = gr.Image(type='pil', label='Original')
                info = gr.Markdown()
                out = gr.File(label='Download SPS-GS result')
        btn.click(predict, [inp, model, scale, seed], [cmp, lr, gt, info, out])
    return demo


def main():
    global DEV
    ap = argparse.ArgumentParser(description='SPS-GS web demo')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    DEV = setup(args.device)
    build().queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
