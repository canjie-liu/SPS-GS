import argparse
import glob
import os

from spsgs import load_image, load_model, setup, super_resolve, to_pil

EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def main():
    ap = argparse.ArgumentParser(description='PSNR / SSIM of SPS-GS on a folder of high-resolution images')
    ap.add_argument('-i', '--input', required=True, help='dataset root or folder of high-resolution test images')
    ap.add_argument('-l', '--list', default='', help='text file of image paths relative to --input, e.g. splits/AID_test.txt')
    ap.add_argument('-m', '--model', default='aid')
    ap.add_argument('-s', '--scale', type=float, nargs='+', default=[2.0, 4.0, 8.0])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tile', type=int, default=360000)
    ap.add_argument('--save', default='', help='folder to save the SPS-GS outputs (optional)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    dev = setup(args.device)
    model = load_model(args.model, dev)
    if args.list:
        files = [os.path.join(args.input, ln.strip()) for ln in open(args.list, encoding='utf-8') if ln.strip()]
        missing = [f for f in files if not os.path.isfile(f)]
        assert not missing, f'{len(missing)} listed images not found, e.g. {missing[0]}'
    else:
        files = sorted(f for f in glob.glob(os.path.join(args.input, '**', '*'), recursive=True)
                       if f.lower().endswith(EXTS))
    assert files, f'no images found in {args.input}'
    for s in args.scale:
        acc = dict(bicubic=[0.0, 0.0], sr=[0.0, 0.0])
        n = 0
        for f in files:
            hr = load_image(f, dev)
            if min(hr.shape[-2:]) // s < 8:
                continue
            r = super_resolve(model, hr, s, seed=args.seed, tile=args.tile)
            for k in acc:
                acc[k][0] += r['metrics'][k][0]
                acc[k][1] += r['metrics'][k][1]
            n += 1
            if args.save:
                d = os.path.join(args.save, f'x{s:g}')
                os.makedirs(d, exist_ok=True)
                to_pil(r['sr']).save(os.path.join(d, os.path.splitext(os.path.basename(f))[0] + '.png'))
        if n:
            print(f'x{s:g} ({n} images): bicubic {acc["bicubic"][0] / n:.3f} dB / {acc["bicubic"][1] / n:.4f}   '
                  f'SPS-GS {acc["sr"][0] / n:.3f} dB / {acc["sr"][1] / n:.4f}', flush=True)


if __name__ == '__main__':
    main()
