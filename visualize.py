#!/usr/bin/env python3
"""Visualization of the labeled tiles.

From the labels of a scene, generates:
  - One COMPOSITE per class: the real image of each tile pasted at its correct
    position in the scene, with the rest black (one image per label).
  - A dedicated composite for the tiles marked as border.
  - An OVERVIEW: the downscaled real scene with each tile outlined by a square
    colored per class, and border tiles highlighted (magenta outline).

Since the real resolution is impractical, everything is generated at a
configurable scale.

Usage:
    python3 visualize.py sat1
    python3 visualize.py sat1 --scale 0.15
    python3 visualize.py --tiles data/sat1_tiles
"""

import argparse
import csv
import os
import sys

from PIL import Image, ImageDraw

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Per-class colors (RGB). Classes absent from the CSV are simply ignored.
COLORS = {
    "sea": (40, 110, 230),
    "land": (60, 160, 70),
    "cloud": (235, 235, 235),
    "object": (235, 45, 45),
    "uncertain": (240, 205, 40),
}
BORDER_COLOR = (235, 45, 235)  # magenta
UNKNOWN_COLOR = (150, 150, 150)


def load(tiles_dir, prefix):
    """Joins the manifest (positions) and the labels (label/border) on the 'file' column."""
    manifest = os.path.join(tiles_dir, f"{prefix}_tiles.csv")
    labels_csv = os.path.join(tiles_dir, f"{prefix}_labels.csv")
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Label CSV not found: {labels_csv}")

    labels = {}
    with open(labels_csv, newline="") as f:
        for r in csv.DictReader(f):
            labels[r["file"]] = (
                r.get("label", "") or "",
                1 if str(r.get("border", "0")) in ("1", "True", "true") else 0,
            )

    tiles = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            name = r["file"]
            label, border = labels.get(name, ("", 0))
            if not label:
                continue  # skip tiles not labeled yet
            tiles.append({
                "file": name,
                "x": int(r["x"]), "y": int(r["y"]),
                "w": int(r["width"]), "h": int(r["height"]),
                "label": label, "border": border,
            })
    return tiles


def generate(prefix, tiles_dir, scale, output_dir, extra_thickness=2):
    tiles = load(tiles_dir, prefix)
    if not tiles:
        raise RuntimeError("No labeled tiles found.")

    # Scene size (absolute positions preserved from 0,0).
    scene_w = max(t["x"] + t["w"] for t in tiles)
    scene_h = max(t["y"] + t["h"] for t in tiles)
    out_w = max(1, round(scene_w * scale))
    out_h = max(1, round(scene_h * scale))

    classes = sorted({t["label"] for t in tiles})
    print(f"{prefix}: {len(tiles)} labeled tiles | scene {scene_w}x{scene_h} "
          f"-> output {out_w}x{out_h} | classes: {classes}")

    overview = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    composites = {c: Image.new("RGB", (out_w, out_h), (0, 0, 0)) for c in classes}
    border_composite = Image.new("RGB", (out_w, out_h), (0, 0, 0))

    # Single I/O pass: opens each tile once and reuses the thumbnail.
    print("Pasting tiles (I/O pass)...")
    for i, t in enumerate(tiles):
        px, py = round(t["x"] * scale), round(t["y"] * scale)
        tw, th = max(1, round(t["w"] * scale)), max(1, round(t["h"] * scale))
        try:
            img = Image.open(os.path.join(tiles_dir, t["file"])).convert("RGB")
            if img.size != (tw, th):
                img = img.resize((tw, th))
        except Exception as e:
            print(f"  warning: failed to open {t['file']}: {e}")
            continue
        overview.paste(img, (px, py))
        composites[t["label"]].paste(img, (px, py))
        if t["border"]:
            border_composite.paste(img, (px, py))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(tiles)} tiles")

    print("Drawing outlines on the overview...")
    draw = ImageDraw.Draw(overview)
    for t in tiles:
        px, py = round(t["x"] * scale), round(t["y"] * scale)
        tw, th = max(1, round(t["w"] * scale)), max(1, round(t["h"] * scale))
        color = COLORS.get(t["label"], UNKNOWN_COLOR)
        # 'object' (rare) gets a thicker outline so it does not get lost in the grid.
        width = 1 + (extra_thickness if t["label"] == "object" else 0)
        draw.rectangle([px, py, px + tw - 1, py + th - 1], outline=color, width=width)

    # Magenta outline (on top) for border tiles.
    for t in tiles:
        if not t["border"]:
            continue
        px, py = round(t["x"] * scale), round(t["y"] * scale)
        tw, th = max(1, round(t["w"] * scale)), max(1, round(t["h"] * scale))
        draw.rectangle([px, py, px + tw - 1, py + th - 1], outline=BORDER_COLOR, width=1)

    _draw_legend(draw, classes)

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    def save(img, name):
        path = os.path.join(output_dir, name)
        img.save(path)
        saved.append(path)

    save(overview, f"{prefix}_overview.png")
    for c in classes:
        save(composites[c], f"{prefix}_composite_{c}.png")
    save(border_composite, f"{prefix}_composite_border.png")

    print(f"Done! {len(saved)} images saved to {output_dir}:")
    for p in saved:
        print("  " + os.path.basename(p))
    return output_dir


def _draw_legend(draw, classes):
    """Draws a simple legend in the top-left corner."""
    x0, y0, step, sw = 8, 8, 22, 16
    items = [(c, COLORS.get(c, UNKNOWN_COLOR)) for c in classes] + [("border", BORDER_COLOR)]
    draw.rectangle([x0 - 4, y0 - 4, x0 + 130, y0 + step * len(items) + 2], fill=(0, 0, 0))
    for i, (name, color) in enumerate(items):
        yy = y0 + i * step
        draw.rectangle([x0, yy, x0 + sw, yy + sw], fill=color, outline=(255, 255, 255))
        draw.text((x0 + sw + 6, yy + 2), name, fill=(255, 255, 255))


def resolve(args):
    if args.tiles:
        tiles_dir = os.path.abspath(args.tiles)
        prefix = args.prefix or os.path.basename(tiles_dir.rstrip("/")).replace("_tiles", "")
    else:
        if not args.prefix:
            args.prefix = input("Enter the image prefix (e.g. sat1): ").strip()
        prefix = args.prefix
        tiles_dir = os.path.join(DATA_DIR, f"{prefix}_tiles")
    return tiles_dir, prefix


def main():
    p = argparse.ArgumentParser(description="Visualization of the labeled tiles.")
    p.add_argument("prefix", nargs="?", help="Image prefix (e.g. sat1)")
    p.add_argument("--tiles", help="Path to the tiles folder (alternative to the prefix)")
    p.add_argument("--scale", type=float, default=0.1, help="Output scale (default 0.1)")
    p.add_argument("--output-dir", help="Output folder (default data/<prefix>_viz)")
    args = p.parse_args()

    tiles_dir, prefix = resolve(args)
    if not os.path.isdir(tiles_dir):
        print(f"ERROR: tiles folder not found: {tiles_dir}")
        sys.exit(1)
    output_dir = args.output_dir or os.path.join(DATA_DIR, f"{prefix}_viz")

    try:
        generate(prefix, tiles_dir, args.scale, output_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
