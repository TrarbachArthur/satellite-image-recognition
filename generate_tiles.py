#!/usr/bin/env python3
"""Converts satellite images in "pan" format (separate r, g, b, p, nir bands)
into a GRID OF RGB TILES (regular images), using Brovey pansharpening.

The final image (RGB + Panchromatic fusion at full resolution) is NEVER built
whole in memory: the Pan band is read in strips of rows (streaming) and the
output is cut into smaller tiles, well suited to object detection.

Usage:
    python generate_tiles.py                     # asks for the prefix
    python generate_tiles.py sat2                # 1024 tiles, scale 1.0, PNG
    python generate_tiles.py sat2 --tile 640     # 640x640 tiles
    python generate_tiles.py sat2 --overlap 64   # 64px overlap between tiles
    python generate_tiles.py sat2 --format jpg   # saves JPG instead of PNG
    python generate_tiles.py sat2 --skip-empty   # drops nearly uniform tiles (open sea)

Assumes the <prefix>_r/g/b/p/nir.tif suffixes inside the 'data' folder.
Brovey RGB only uses r, g, b, p (nir does not go into the visible RGB image).
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import tifffile

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def open_page(prefix, suffix):
    """Opens the TIFF page of a band (without loading the data)."""
    path = os.path.join(DATA_DIR, f"{prefix}_{suffix}.tif")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return tifffile.TiffFile(path).pages[0]


def read_strip(page, r0, r1):
    """Reads rows [r0:r1) of a TIFF page, decoding only the strips needed
    (does not load the whole file)."""
    H, W = page.shape[0], page.shape[1]
    r0, r1 = max(0, r0), min(H, r1)
    rps = page.rowsperstrip or 1
    offs, cnts = page.dataoffsets, page.databytecounts
    fh = page.parent.filehandle
    s0, s1 = r0 // rps, (r1 - 1) // rps
    parts = []
    for s in range(s0, s1 + 1):
        fh.seek(offs[s])
        seg, _, shp = page.decode(fh.read(cnts[s]), s)
        parts.append(np.asarray(seg).reshape(shp[-3], shp[-2]))
    block = np.concatenate(parts, axis=0)
    start = r0 - s0 * rps
    return block[start:start + (r1 - r0)]


def read_full(page):
    """Reads a whole band (only use on small bands: R, G, B)."""
    arr = page.asarray()
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def brovey_bands(R, G, B, Pan):
    """Applies Brovey fusion to already aligned float32 arrays (same shape)."""
    I = (R + G + B) / 3.0
    I[I == 0] = np.finfo(np.float32).eps  # avoid division by zero
    f = Pan / I
    return R * f, G * f, B * f


def compute_limits(pan_page, R, G, B, target=1600):
    """Estimates contrast limits (2nd and 98th percentiles) per channel from a
    downscaled version of the image, so that ALL tiles share the same 8-bit scale."""
    print("Estimating global contrast (downscaled sample)...")
    H = pan_page.shape[0]
    step = max(1, H // target)
    rows = [read_strip(pan_page, i, i + 1)[0] for i in range(0, H, step)]
    pan_s = np.asarray(rows, dtype=np.float32)
    h2 = pan_s.shape[0]
    w2 = min(pan_page.shape[1], target)
    pan_s = cv2.resize(pan_s, (w2, h2), interpolation=cv2.INTER_AREA)
    Rr = cv2.resize(R.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
    Gr = cv2.resize(G.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
    Br = cv2.resize(B.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
    bR, bG, bB = brovey_bands(Rr, Gr, Br, pan_s)
    lo, hi = [], []
    for band in (bR, bG, bB):
        l, h = np.percentile(band, (2, 98))
        lo.append(l)
        hi.append(max(h, l + 1e-6))
    print(f"  limits per channel (R,G,B): lo={np.round(lo,1)} hi={np.round(hi,1)}")
    return np.array(lo, np.float32), np.array(hi, np.float32)


def process(prefix, tile=1024, scale=1.0, overlap=0,
            fmt="png", output_dir=None, skip_empty=False, empty_threshold=3.0):
    print(f"Processing '{prefix}' (tile={tile}, scale={scale}, overlap={overlap})")

    pan_page = open_page(prefix, "p")
    R = read_full(open_page(prefix, "r")).astype(np.float32)
    G = read_full(open_page(prefix, "g")).astype(np.float32)
    B = read_full(open_page(prefix, "b")).astype(np.float32)

    pan_h, pan_w = pan_page.shape[0], pan_page.shape[1]
    rgb_h, rgb_w = R.shape[:2]
    out_h = max(1, int(round(pan_h * scale)))
    out_w = max(1, int(round(pan_w * scale)))
    print(f"  Pan {pan_w}x{pan_h}  ->  output {out_w}x{out_h}")

    limits_lo, limits_hi = compute_limits(pan_page, R, G, B)
    span = limits_hi - limits_lo

    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, f"{prefix}_tiles")
    os.makedirs(output_dir, exist_ok=True)

    # Tile origin coordinates (with optional overlap).
    step = max(1, tile - overlap)
    tile_xs = list(range(0, out_w, step))
    tile_ys = list(range(0, out_h, step))
    print(f"  Grid: {len(tile_xs)} x {len(tile_ys)} = "
          f"{len(tile_xs) * len(tile_ys)} tiles (before filters)")

    # Resampling factors (output -> native band coordinates).
    sx_r, sy_r = rgb_w / out_w, rgb_h / out_h
    sx_p, sy_p = pan_w / out_w, pan_h / out_h

    manifest = []
    saved = discarded = 0

    # Processes one strip (row of tiles) at a time: reads the strip's Pan once
    # and cuts the tiles into columns. The remap is done PER TILE (small
    # dimensions) because cv2.remap only accepts up to 32767px per axis.
    for ty in tile_ys:
        bh = min(tile, out_h - ty)

        # Strip Pan: read directly at scale 1.0; resampled otherwise.
        if scale == 1.0:
            pan_strip = read_strip(pan_page, ty, ty + bh).astype(np.float32)
            pan_src_y0 = ty  # origin in Pan coordinates
        else:
            py0 = max(0, int(np.floor(ty * sy_p)) - 2)
            py1 = min(pan_h, int(np.ceil((ty + bh) * sy_p)) + 3)
            pan_strip = read_strip(pan_page, py0, py1).astype(np.float32)
            pan_src_y0 = py0

        # Source (global) coordinates of this strip's output rows.
        ys_rgb = (np.arange(ty, ty + bh, dtype=np.float32) + 0.5) * sy_r - 0.5

        for tx in tile_xs:
            bw = min(tile, out_w - tx)

            # RGB: direct remap from the native bands (source < 32767px)
            xs_rgb = (np.arange(tx, tx + bw, dtype=np.float32) + 0.5) * sx_r - 0.5
            mx = np.ascontiguousarray(np.broadcast_to(xs_rgb, (bh, bw)))
            my = np.ascontiguousarray(np.broadcast_to(ys_rgb[:, None], (bh, bw)))
            Ru = cv2.remap(R, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            Gu = cv2.remap(G, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            Bu = cv2.remap(B, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # Tile Pan
            if scale == 1.0:
                Pan = pan_strip[:, tx:tx + bw]
            else:
                cx0 = max(0, int(np.floor(tx * sx_p)) - 2)
                cx1 = min(pan_w, int(np.ceil((tx + bw) * sx_p)) + 3)
                sub = pan_strip[:, cx0:cx1]
                pmx = np.ascontiguousarray(np.broadcast_to(
                    ((np.arange(tx, tx + bw, dtype=np.float32) + 0.5) * sx_p - 0.5 - cx0), (bh, bw)))
                pmy = np.ascontiguousarray(np.broadcast_to(
                    (ys_rgb * 0 + (np.arange(ty, ty + bh, dtype=np.float32) + 0.5) * sy_p - 0.5 - pan_src_y0)[:, None],
                    (bh, bw)))
                Pan = cv2.remap(sub, pmx, pmy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # Brovey + 8-bit normalization with global limits
            bR, bG, bB = brovey_bands(Ru, Gu, Bu, Pan)
            chip = np.empty((bh, bw, 3), dtype=np.uint8)
            for c, band in enumerate((bR, bG, bB)):
                v = (band - limits_lo[c]) / span[c] * 255.0
                chip[:, :, c] = np.clip(v, 0, 255).astype(np.uint8)

            if skip_empty and float(chip.std()) < empty_threshold:
                discarded += 1
                continue

            name = f"{prefix}_y{ty:06d}_x{tx:06d}.{fmt}"
            cv2.imwrite(os.path.join(output_dir, name), cv2.cvtColor(chip, cv2.COLOR_RGB2BGR))
            manifest.append([name, tx, ty, bw, bh])
            saved += 1

        print(f"  strip y={ty:6d} ({bh}px) -> {saved} tiles saved so far")

    # Manifest mapping each tile back to the full image.
    with open(os.path.join(output_dir, f"{prefix}_tiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "x", "y", "width", "height"])
        w.writerows(manifest)

    print(f"Done! {saved} tiles saved to {output_dir}"
          + (f" ({discarded} empty tiles discarded)" if skip_empty else ""))
    return output_dir


def main():
    p = argparse.ArgumentParser(description="Generates RGB tiles (Brovey) from satellite bands.")
    p.add_argument("prefix", nargs="?", help="Image prefix (e.g. sat2)")
    p.add_argument("--tile", type=int, default=1024, help="Tile size in pixels (default 1024)")
    p.add_argument("--scale", type=float, default=1.0, help="Output resolution scale (default 1.0)")
    p.add_argument("--overlap", type=int, default=0, help="Overlap between tiles in pixels (default 0)")
    p.add_argument("--format", choices=["png", "jpg"], default="png", help="Output format (default png)")
    p.add_argument("--output-dir", help="Output folder (default data/<prefix>_tiles)")
    p.add_argument("--skip-empty", action="store_true", help="Drops nearly uniform tiles (open sea)")
    args = p.parse_args()

    prefix = args.prefix or input("Enter the image prefix (e.g. sat2): ").strip()
    if not prefix:
        print("Empty prefix. Aborting.")
        sys.exit(1)

    try:
        process(prefix, tile=args.tile, scale=args.scale, overlap=args.overlap,
                fmt=args.format, output_dir=args.output_dir, skip_empty=args.skip_empty)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
