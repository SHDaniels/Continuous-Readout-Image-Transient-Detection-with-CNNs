#!/usr/bin/env python
"""
make_dim_to_zero_stamps.py

~200 stamps of dimming transients that go all the way to zero streak flux at the peak
(A = 0), drawn from the same original images as the main injections.

Reuses implant_stamps_v2 (inject / make_stamps), so durations (sigma 3-9), the
5 stamps per transient (offsets 0, +-1, +-2) and the info format
(A, sigma, maxcol, center_y, offset) are identical to the other injections.
Default: 4 images x 10 transients x 5 stamps = 200 stamps.

    python make_dim_to_zero_stamps.py --mode mult --norm renorm --out /lustre/lrspec/ZTF_data/dim0_mult_renorm/
    python make_dim_to_zero_stamps.py --mode add  --norm renorm --out /lustre/lrspec/ZTF_data/dim0_add_renorm/
"""
import argparse
import os

import numpy as np

import implant_stamps_add_or_mult as v2

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["mult", "add"], required=True)
ap.add_argument("--norm", choices=["renorm", "once", "none"], default="renorm")
ap.add_argument("--out", required=True)
ap.add_argument("--images", default=v2.DEFAULT_IMAGES)
ap.add_argument("--n-images", type=int, default=4)   # x10 transients x5 stamps = 200
ap.add_argument("--seed", type=int, default=9000)    # differs from the main script's seeds
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
files = sorted(os.listdir(a.images))
combos = [(f, h) for f in files for h in (1, 2, 3, 4)]
picks = rng.choice(len(combos), size=min(a.n_images, len(combos)), replace=False)

for sub in ("images", "info"):
    os.makedirs(os.path.join(a.out, sub), exist_ok=True)

n_total = 0
for i in picks:
    fname, hdu = combos[i]
    image = v2.load_image(os.path.join(a.images, fname), hdu, a.norm)
    n = 2 * v2.N_PER_IMAGE   # a little slack in case of tied peak heights
    centers = rng.integers(30, 3050, size=n)
    sigmas = rng.integers(3, 10, size=n)
    stamps, info, _, _ = v2.make_stamps(image, sigmas, np.zeros(n), centers, 0, a.mode, a.norm)

    tag = f"{fname[:40]}_{hdu}_stamps_dim0"
    np.save(os.path.join(a.out, "images", tag), np.array(stamps))
    np.save(os.path.join(a.out, "info", tag + "_info"), np.array(info))
    n_total += len(stamps)
    print(fname, "hdu", hdu, "->", len(stamps), "stamps", flush=True)

print("total stamps:", n_total)