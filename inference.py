import argparse
import glob
import os

from spsgs import load_image, load_model, setup, super_resolve, to_pil

EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def main():
    ap = argparse.ArgumentParser(description='SPS-GS inference from single images (simulation mode)')
    ap.add_argument('-i', '--input', default='assets/examples', help='image file or folder')
    ap.add_argument('-o', '--output', default='results', help='output folder')
    ap.add_argument('-m', '--model', default='aid', help='aid | ucmerced | whu-rs19 | rsscn7 | nwpu-resisc45 | div2k | path')
    ap.add_argument('-s', '--scale', type=float, nargs='+', default=[4.0], help='scale factors, e.g. 2 3.5 8')
    ap.add_argument('--seed', type=int, default=0, help='seed of the sub-pixel shifts')
    ap.add_argument('--tile', type=int, default=360000, help='tile LR inputs with more pixels than this (integer scales)')
    ap.add_argument('--save-all', action='store_true', help='also save the cropped original, bicubic and LR frames')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    dev = setup(args.device)
    model = load_model(args.model, dev)
    if os.path.isdir(args.input):
        files = sorted(f for f in glob.glob(os.path.join(args.input, '*')) if f.lower().endswith(EXTS))
    else:
        files = [args.input]
    os.makedirs(args.output, exist_ok=True)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        hr = load_image(f, dev)
        for s in args.scale:
            r = super_resolve(model, hr, s, seed=args.seed, tile=args.tile)
            tag = f'{name}_x{s:g}'
            to_pil(r['sr']).save(os.path.join(args.output, f'{tag}_SPS-GS.png'))
            if args.save_all:
                to_pil(r['bicubic']).save(os.path.join(args.output, f'{tag}_bicubic.png'))
                to_pil(r['hr']).save(os.path.join(args.output, f'{tag}_original.png'))
                for k in range(r['frames'].shape[1]):
                    to_pil(r['frames'][:, k]).save(os.path.join(args.output, f'{tag}_LR{k}.png'))
            (pb, sb), (ps, ss) = r['metrics']['bicubic'], r['metrics']['sr']
            print(f'{tag}: bicubic {pb:.2f} dB / {sb:.4f}   SPS-GS {ps:.2f} dB / {ss:.4f}', flush=True)


if __name__ == '__main__':
    main()
