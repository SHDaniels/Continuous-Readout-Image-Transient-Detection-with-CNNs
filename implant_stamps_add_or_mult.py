#!/usr/bin/env python
"""
implant_stamps_v2.py

Clean rewrite of the stamp-implantation script, with switches so that
additive vs. multiplicative (and normalized vs. not) is the ONLY thing that
changes between runs.

    # generate stamps
    python implant_stamps_v2.py run --mode add  --norm renorm --out /lustre/lrspec/ZTF_data/training_add_renorm/
    python implant_stamps_v2.py run --mode mult --norm renorm --out /lustre/lrspec/ZTF_data/training_mult_renorm/

    # sanity checks on one image (do this first)
    python implant_stamps_v2.py selftest --image /lustre/lrspec/ZTF_data/training_aug24/implant_train/<some file>

--mode
    mult : v' = v * ((A-1)*g(y) + 1)              (what the old code does, on background-subtracted flux)
    add  : v' = v + (A-1)*g(y)*ref_col            ref_col = mean streak flux in that column over the
                                                  same +-10 row window the old code uses to find columns.
                                                  So at the transient peak the injected flux equals the
                                                  multiplicative case on average, but it no longer scales
                                                  with the individual pixel value (noise, structure).
--norm
    renorm : legacy. min-max normalize the image before injection AND after every injection
    once   : min-max normalize before injection only; fixed scale, no renormalizing afterwards
    none   : raw science pixels, never normalized (see note in load_image)

Reuses from implantation_functions: gaussian_event, get_image_background, get_stamp,
get_intensities_{pos,neg}_feb, get_10_streak_stamps_w_offcenter.
Output layout / filenames match the old training_feb26 layout, so the assembly
script only needs its root path changed.
"""
import argparse
import os

import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks

import implantation_functions as imp

DEFAULT_IMAGES = "/lustre/lrspec/ZTF_data/training_aug24/implant_train/"
N_PER_IMAGE = 10
LEGACY_SIZE = 287 * 4 * 10 + 1   # keep so center_values / sigmas draw from the same RNG stream as before
OFFSETS_POS = np.concatenate(([1, 2], np.negative([1, 2])))        # labelled as transient
OFFSETS_BG = np.concatenate((np.arange(4, 25), -np.arange(4, 25)))  # labelled as background


def load_image(path, hdu, norm):
    with fits.open(path) as hdul:
        arr = np.array(hdul[hdu].data, dtype=np.float64)
    if norm in ("renorm", "once"):
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    # norm == "none": get_image_background clips at 0 from below, so this only makes sense if
    # the raw pixels are non-negative. Also, stamps will be on a totally different scale than the
    # CR / satellite stamps, so normalize every class per-stamp downstream if you use this.
    return arr


def find_streak_columns(col_mean, streak_x):
    """Verbatim column-selection logic from implant_transient_full_image."""
    maxcol = np.argmax(col_mean[streak_x - 5: streak_x + 5]) + streak_x - 5
    maxval = col_mean[maxcol]
    cols = [maxcol, maxcol + 1]
    val = col_mean[maxcol + 1]
    for i in range(2, 6):
        idx = maxcol + i
        newval = col_mean[idx]
        if newval > val or newval < 0.1 * maxval:
            break
        val = newval
        cols.append(idx)
    cols.append(maxcol - 1)
    val = col_mean[maxcol - 1]
    for i in range(2, 6):
        idx = maxcol - i
        newval = col_mean[idx]
        if newval > val or newval < 0.1 * maxval:
            break
        val = newval
        cols.append(idx)
    return maxcol, cols


def inject(image, streak_x, center_y, A, sigma, mode, norm):
    """Inject one gaussian brightening/dimming into the streak near streak_x. Returns (image, maxcol)."""
    center_y = int(center_y)
    background = imp.get_image_background(image)
    excess = image - background
    col_mean = excess[center_y - 10: center_y + 10].mean(axis=0)
    maxcol, cols = find_streak_columns(col_mean, int(streak_x))

    ys = np.arange(image.shape[0], dtype=float)
    out = excess.copy()
    if mode == "mult":
        for c in cols:
            out[:, c] = imp.gaussian_event(ys, center_y, sigma, A, excess[:, c])
    elif mode == "add":
        g = np.exp(-((ys - center_y) ** 2) / (2.0 * sigma ** 2))
        for c in cols:
            ref = max(col_mean[c], 0.0)   # guard: a column that is below background shouldn't flip the sign
            out[:, c] = excess[:, c] + (A - 1.0) * g * ref
    else:
        raise ValueError(mode)
    out += background

    if norm == "renorm":
        out = (out - out.min()) / (out.max() - out.min())
    return out, maxcol


def make_stamps(image, sigmas, intensities, center_values, counter, mode, norm):
    """Replacement for get_10_trans_stamps_w_offcenter, same outputs, with a mode switch."""
    averages = np.average(image[50:-50, 50:-50], axis=0)
    peaks, heights = find_peaks(averages, height=(None, None))
    min_height = np.sort(heights["peak_heights"])[-N_PER_IMAGE]
    tall_peaks = peaks[heights["peak_heights"] >= min_height] + 50

    implanted = image
    stamps, info, bg_stamps, bg_info = [], [], [], []
    for k, x in enumerate(tall_peaks):
        cy = center_values[counter + k]
        A = intensities[counter + k]
        sig = sigmas[counter + k]
        implanted, maxcol = inject(implanted, x, cy, A, sig, mode, norm)

        stamps.append(imp.get_stamp(implanted, maxcol, cy))
        info.append((A, sig, maxcol, cy, 0))
        for off in OFFSETS_POS:
            stamps.append(imp.get_stamp(implanted, maxcol + off, cy))
            info.append((A, sig, maxcol, cy, off))
        for off in OFFSETS_BG:
            bg_stamps.append(imp.get_stamp(implanted, maxcol + off, cy))
            bg_info.append((A, sig, maxcol, cy, off))
    return stamps, info, bg_stamps, bg_info


def save(out, kind, start, hdu, stamps, info, bgs, bg_info):
    tag = f"{start}_{hdu}_stamps"
    np.save(f"{out[kind]}/images/{tag}", np.array(stamps))
    np.save(f"{out[kind]}/info/{tag}_info", np.array(info))
    np.save(f"{out['bg']}/images/{tag}_{kind}", np.array(bgs))
    np.save(f"{out['bg']}/info/{tag}_info_{kind}", np.array(bg_info))


def run(a):
    files = sorted(os.listdir(a.images))   # sorted: file order (and so the counters) is reproducible
    assert a.size >= len(files) * N_PER_IMAGE, "--size too small for this many images"
    for f in files:
        # assembly script pairs bg image/info files by slicing the name at character 50
        assert len(f[:40]) == 40, f"filename shorter than 40 chars: {f}"

    out = {k: os.path.join(a.out, d) for k, d in dict(
        pos="offcenter_stamps_pos", neg="offcenter_stamps_neg",
        streak="offcenter_stamps_streaks", bg="offcenter_stamps_background").items()}
    for d in out.values():
        for sub in ("images", "info"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)

    for hdu in a.hdus:
        print(f"=== HDU {hdu} | mode={a.mode} norm={a.norm} ===", flush=True)
        np.random.seed(20 + hdu)
        center_values = np.random.randint(30, high=3050, size=a.size, dtype=int)
        sigmas = np.random.randint(3, high=10, size=a.size, dtype=int)
        # EXPLICIT integer seeds. The old script passed seed=None, which makes
        # np.random.seed(None) reseed from OS entropy -> intensities were not reproducible.
        intensities = {
            "pos": imp.get_intensities_pos_feb(a.size, 1000 + hdu),
            "neg": imp.get_intensities_neg_feb(a.size, 2000 + hdu),
        }
        counter = 0
        for fname in files:
            print(fname, flush=True)
            start = fname[:40]
            image = load_image(os.path.join(a.images, fname), hdu, a.norm)
            for kind in ("pos", "neg", "streak"):
                if kind == "streak":   # no injection, so unaffected by --mode
                    res = imp.get_10_streak_stamps_w_offcenter(image, center_values, counter)
                else:
                    res = make_stamps(image, sigmas, intensities[kind], center_values,
                                      counter, a.mode, a.norm)
                save(out, kind, start, hdu, *res)
            counter += N_PER_IMAGE


def selftest(a):
    img = load_image(a.image, 1, "renorm")
    cy, sig, A = 1500, 5, 1.5
    x = int(np.argmax(img[50:-50, 50:-50].mean(axis=0))) + 50

    legacy, mc_old = imp.implant_transient_full_image(img, x, cy, A, sig, True)
    new, mc_new = inject(img, x, cy, A, sig, "mult", "renorm")
    print("1) mult/renorm vs legacy: maxcol equal =", mc_old == mc_new,
          "| max abs diff =", np.abs(legacy - new).max(), "(should be ~1e-12)")

    m, _ = inject(img, x, cy, A, sig, "mult", "once")
    ad, _ = inject(img, x, cy, A, sig, "add", "once")
    dm, da = m - img, ad - img
    print("2) injected flux, same A, no renorm:")
    print("   peak pixel delta   mult %.4g   add %.4g" % (dm.max(), da.max()))
    print("   summed delta       mult %.4g   add %.4g" % (dm.sum(), da.sum()))
    print("   (these should be comparable; the matching is on the mean, so not identical)")

    for A2 in (0.5, 0.9):
        m2, _ = inject(img, x, cy, A2, sig, "mult", "once")
        a2, _ = inject(img, x, cy, A2, sig, "add", "once")
        print(f"3) dimming A={A2}: min delta mult {np.min(m2 - img):.4g}  add {np.min(a2 - img):.4g}"
              f" | pixels pushed below image min: mult {(m2 < img.min()).sum()}  add {(a2 < img.min()).sum()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=["mult", "add"], required=True)
    r.add_argument("--norm", choices=["renorm", "once", "none"], default="renorm")
    r.add_argument("--out", required=True)
    r.add_argument("--images", default=DEFAULT_IMAGES)
    r.add_argument("--hdus", type=int, nargs="+", default=[1, 2, 3, 4])
    r.add_argument("--size", type=int, default=LEGACY_SIZE)
    s = sub.add_parser("selftest")
    s.add_argument("--image", required=True)
    args = p.parse_args()
    run(args) if args.cmd == "run" else selftest(args)