#!/usr/bin/env python3
"""Converts satellite images in "pan" format (separate r, g, b, p, nir bands)
into a regular RGB image, using the Brovey pansharpening algorithm.

Usage:
    python generate_rgb.py                # asks for the prefix interactively
    python generate_rgb.py sat1           # processes data/sat1_{r,g,b,p,nir}.tif
    python generate_rgb.py sat1 --scale 0.25   # output at 1/4 of the Pan resolution

Assumes the files follow the pattern <prefix>_r.tif, <prefix>_g.tif,
<prefix>_b.tif, <prefix>_p.tif and <prefix>_nir.tif inside the 'data' folder.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import tifffile

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SUFFIXES = ["r", "g", "b", "p", "nir"]


def read_band(prefix, suffix):
    """Reads a band from its prefix and suffix."""
    path = os.path.join(DATA_DIR, f"{prefix}_{suffix}.tif")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    print(f"  reading {os.path.basename(path)} ...")
    band = tifffile.imread(path)
    # If the band comes with a channel axis (e.g. HxWx1), flatten it to 2D.
    if band.ndim == 3 and band.shape[2] == 1:
        band = band[:, :, 0]
    return band


def brovey(prefix, scale=1.0, output=None):
    print(f"Processing prefix '{prefix}' ...")

    R = read_band(prefix, "r")
    G = read_band(prefix, "g")
    B = read_band(prefix, "b")
    Pan = read_band(prefix, "p")

    original_dtype = Pan.dtype
    pan_rows, pan_cols = Pan.shape[:2]

    width = max(1, int(round(pan_cols * scale)))
    height = max(1, int(round(pan_rows * scale)))
    print(f"Pan resolution: {pan_cols}x{pan_rows} -> output: {width}x{height}")

    # Resize in the native dtype (uint8/uint16) and only then convert to
    # float64. This avoids creating a huge full-resolution float64 copy (for the
    # Pan band that would be tens of GB of RAM). cv2.resize takes (width, height).
    print("Resizing the bands...")

    def resize(band):
        # INTER_AREA when shrinking (better quality); INTER_CUBIC when enlarging.
        h, w = band.shape[:2]
        interp = cv2.INTER_AREA if (width < w or height < h) else cv2.INTER_CUBIC
        return cv2.resize(band, (width, height), interpolation=interp).astype(np.float64)

    R = resize(R)
    G = resize(G)
    B = resize(B)
    Pan = resize(Pan)

    print("Applying pansharpening (Brovey)...")
    intensity = (R + G + B) / 3.0
    intensity[intensity == 0] = np.finfo(np.float64).eps  # avoid division by zero

    factor = Pan / intensity
    R *= factor
    G *= factor
    B *= factor

    if original_dtype == np.uint8:
        max_value = 255
    elif original_dtype == np.uint16:
        max_value = 65535
    else:
        max_value = float(Pan.max())

    rgb = np.stack([R, G, B], axis=-1)
    np.clip(rgb, 0, max_value, out=rgb)

    # Scale [0, max_value] to [0, 255] for an 8-bit viewable image.
    rgb8 = (rgb / max_value * 255.0).astype(np.uint8)

    if output is None:
        output = os.path.join(DATA_DIR, f"{prefix}_rgb.png")
    # cv2 expects BGR when writing.
    cv2.imwrite(output, cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))
    print(f"Done! Image saved to: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Generates RGB (Brovey) from satellite bands.")
    parser.add_argument("prefix", nargs="?", help="Image prefix (e.g. sat1)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Output resolution scale factor (e.g. 0.25). Default: 1.0")
    parser.add_argument("--output", help="Output file path (default: data/<prefix>_rgb.png)")
    args = parser.parse_args()

    prefix = args.prefix or input("Enter the image prefix (e.g. sat1): ").strip()
    if not prefix:
        print("Empty prefix. Aborting.")
        sys.exit(1)

    try:
        brovey(prefix, scale=args.scale, output=args.output)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
