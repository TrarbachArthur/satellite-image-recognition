#!/usr/bin/env python3
"""Visual label correction.

Shows the whole downscaled scene (real image + per-class color overlay) on a
scrollable canvas. Click/drag to select tiles and reassign the label (or the
border flag) of all selected tiles at once, writing back to the same
<prefix>_labels.csv.

Usage:
    python3 fix_labels.py sat1                 # scale 0.25 (default)
    python3 fix_labels.py sat1 --scale 0.2
    python3 fix_labels.py --tiles data/sat1_tiles
"""

import argparse
import csv
import os
import sys

from PIL import Image

from label import Labeler, LABEL_KEYS, VALID_LABELS
from visualize import COLORS, UNKNOWN_COLOR

Image.MAX_IMAGE_PIXELS = None  # downscaled scenes are still large (>178 MP)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OVERLAY_ALPHA = 105  # 0-255: strength of the colored overlay


def read_manifest(tiles_dir, prefix):
    """Returns (coords, step, scene_w, scene_h).
    coords: {file: (x, y, w, h)}; step: (step_x, step_y) of the grid."""
    manifest = os.path.join(tiles_dir, f"{prefix}_tiles.csv")
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    coords = {}
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            coords[r["file"]] = (int(r["x"]), int(r["y"]),
                                 int(r["width"]), int(r["height"]))
    if not coords:
        raise RuntimeError("Empty manifest.")
    first = next(iter(coords.values()))
    step = (first[2], first[3])
    scene_w = max(x + w for x, y, w, h in coords.values())
    scene_h = max(y + h for x, y, w, h in coords.values())
    return coords, step, scene_w, scene_h


def build_base(coords, tiles_dir, scale, out_w, out_h):
    """Pastes all real tiles onto one downscaled image (the expensive I/O step)."""
    base = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    total = len(coords)
    for i, (file, (x, y, w, h)) in enumerate(coords.items()):
        px, py = round(x * scale), round(y * scale)
        tw, th = max(1, round(w * scale)), max(1, round(h * scale))
        try:
            img = Image.open(os.path.join(tiles_dir, file)).convert("RGB")
            if img.size != (tw, th):
                img = img.resize((tw, th))
        except Exception as e:
            print(f"  warning: failed to open {file}: {e}")
            continue
        base.paste(img, (px, py))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{total} tiles")
    return base


def get_base(prefix, tiles_dir, coords, scale, out_w, out_h):
    """Loads the base from the cache, or builds and saves it (cache next to the tiles folder)."""
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(tiles_dir)), f"{prefix}_viz")
    os.makedirs(viz_dir, exist_ok=True)
    cache = os.path.join(viz_dir, f"{prefix}_base_e{scale:g}.png")
    if os.path.exists(cache):
        img = Image.open(cache).convert("RGB")
        if img.size == (out_w, out_h):
            print(f"Base loaded from cache: {cache}")
            return img
    print(f"Building base (scale {scale}) — pasting {len(coords)} tiles...")
    base = build_base(coords, tiles_dir, scale, out_w, out_h)
    base.save(cache)
    print(f"Base saved to {cache}")
    return base


def build_overlay(base, labeler, coords, scale, step, class_filter=None):
    """Over the real base, blends a translucent color per class (small grid
    upscaled with NEAREST). Reflects the current state of the labels.

    If class_filter is given, tiles whose label is not in the filter get no
    tint (they show the raw image) -- useful for reviewing a single class."""
    px, py = step
    cols = max(x // px for x, y, w, h in coords.values()) + 1
    rows = max(y // py for x, y, w, h in coords.values()) + 1
    color = Image.new("RGB", (cols, rows), (0, 0, 0))
    alpha = Image.new("L", (cols, rows), 0)
    for file, (x, y, w, h) in coords.items():
        label, _ = labeler.state(file)
        if not label:
            continue
        if class_filter is not None and label not in class_filter:
            continue
        color.putpixel((x // px, y // py), COLORS.get(label, UNKNOWN_COLOR))
        alpha.putpixel((x // px, y // py), OVERLAY_ALPHA)
    color_up = color.resize(base.size, Image.NEAREST)
    alpha_up = alpha.resize(base.size, Image.NEAREST)
    over = base.copy()
    over.paste(color_up, (0, 0), alpha_up)
    return over


def run_gui(prefix, labeler, coords, scale, step, base_real, base_over, class_filter=None):
    """Viewer with zoom and pan over the downscaled base (fixed scale).

    Only the visible region is cropped from the base and enlarged, so it stays
    light and smooth at any zoom level. Internal coordinates are in "base px"
    (= scene * scale).

    If class_filter is given, only tiles whose current label is in the filter
    can be selected (single click or drag)."""
    import tkinter as tk
    from PIL import ImageTk

    step_x, step_y = step
    origin = {(x, y): a for a, (x, y, w, h) in coords.items()}
    base_w, base_h = base_real.size
    Z_MAX = 8.0  # max magnification of the base (it is already downscaled; beyond this it blurs)

    def passes_filter(name):
        return class_filter is None or labeler.state(name)[0] in class_filter

    window = tk.Tk()
    window.title(f"Label correction — {prefix}")

    top_bar = tk.Label(window, anchor="w", bg="#222", fg="#eee",
                       font=("TkDefaultFont", 11), padx=6, pady=4)
    top_bar.pack(fill="x")

    canvas = tk.Canvas(window, bg="#000", highlightthickness=0, width=1100, height=760)
    canvas.pack(fill="both", expand=True)

    legend = " | ".join([f"{k}={r}" for k, r in LABEL_KEYS] + [
        "b/n=border", "O=overlay", "wheel=zoom", "right-btn/arrows=pan", "0=fit",
        "Esc=clear", "q=quit"])
    footer = tk.Label(window, text=legend, anchor="w", bg="#222", fg="#aaa",
                      font=("TkDefaultFont", 9), padx=6, pady=3)
    footer.pack(fill="x")

    # z: screen px per base px; (ox, oy): top-left visible corner, in base px.
    state = {"overlay": True, "photo": None, "z": None, "ox": 0.0, "oy": 0.0,
             "sel": set(), "drag": None, "rubber": None, "pan": None}

    def z_fit(W, H):
        return min(W / base_w, H / base_h)

    def clamp_offsets(W, H):
        z = state["z"]
        vw, vh = W / z, H / z
        state["ox"] = (base_w - vw) / 2 if vw >= base_w else min(max(state["ox"], 0), base_w - vw)
        state["oy"] = (base_h - vh) / 2 if vh >= base_h else min(max(state["oy"], 0), base_h - vh)

    def display_to_base(dx, dy):
        return state["ox"] + dx / state["z"], state["oy"] + dy / state["z"]

    def base_to_display(bx, by):
        return (bx - state["ox"]) * state["z"], (by - state["oy"]) * state["z"]

    def display_bbox(file):
        x, y, w, h = coords[file]
        x0, y0 = base_to_display(x * scale, y * scale)
        x1, y1 = base_to_display((x + w) * scale, (y + h) * scale)
        return x0, y0, x1, y1

    def tile_at(dx, dy):
        bx, by = display_to_base(dx, dy)
        sx, sy = bx / scale, by / scale
        return origin.get((int(sx // step_x) * step_x, int(sy // step_y) * step_y))

    def render():
        W, H = canvas.winfo_width(), canvas.winfo_height()
        if W < 10 or H < 10:
            return
        if state["z"] is None:
            state["z"] = z_fit(W, H)
        state["z"] = max(z_fit(W, H), min(state["z"], Z_MAX))
        clamp_offsets(W, H)
        z, ox, oy = state["z"], state["ox"], state["oy"]

        src = base_over if state["overlay"] else base_real
        vw, vh = W / z, H / z
        region = src.crop((round(ox), round(oy), round(ox + vw), round(oy + vh)))
        region = region.resize((W, H))
        state["photo"] = ImageTk.PhotoImage(region)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=state["photo"])

        # outline the visible corrected/selected tiles (small sets)
        for a in state["sel"]:
            x0, y0, x1, y1 = display_bbox(a)
            if x1 >= 0 and y1 >= 0 and x0 <= W and y0 <= H:
                canvas.create_rectangle(x0, y0, x1, y1, outline="#fff", width=2)
        if state["rubber"]:
            canvas.coords(state["rubber"], *state["rubber_xy"])
        update_status()

    def update_status(msg=""):
        n = len(state["sel"])
        if n == 1:
            a = next(iter(state["sel"]))
            r, b = labeler.state(a)
            detail = f"  |  {a}: label={r or '-'} border={b}"
        else:
            detail = f"  |  {n} selected" if n else "  |  nothing selected"
        zoom_pct = (state["z"] / z_fit(canvas.winfo_width() or 1, canvas.winfo_height() or 1))
        filter_suffix = f"  [filter: {','.join(class_filter)}]" if class_filter else ""
        top_bar.config(text=f"{prefix} ({len(coords)} tiles)  zoom {zoom_pct:.1f}x{detail}{filter_suffix}   {msg}")

    def on_press(event):
        state["drag"] = (event.x, event.y, bool(event.state & 0x0001))
        state["rubber_xy"] = (event.x, event.y, event.x, event.y)
        state["rubber"] = canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                  outline="#ff0", dash=(4, 2))

    def on_drag(event):
        if not state["drag"]:
            return
        x0, y0, _ = state["drag"]
        state["rubber_xy"] = (x0, y0, event.x, event.y)
        canvas.coords(state["rubber"], x0, y0, event.x, event.y)

    def on_release(event):
        if not state["drag"]:
            return
        x0, y0, shift = state["drag"]
        state["drag"] = None
        if state["rubber"]:
            canvas.delete(state["rubber"])
            state["rubber"] = None
        if not shift:
            state["sel"] = set()
        if abs(event.x - x0) < 5 and abs(event.y - y0) < 5:
            a = tile_at(event.x, event.y)
            if a and passes_filter(a):
                state["sel"].add(a)
        else:
            bx0, by0 = display_to_base(min(x0, event.x), min(y0, event.y))
            bx1, by1 = display_to_base(max(x0, event.x), max(y0, event.y))
            sx0, sy0, sx1, sy1 = bx0 / scale, by0 / scale, bx1 / scale, by1 / scale
            for a, (x, y, w, h) in coords.items():
                if sx0 <= x + w / 2 <= sx1 and sy0 <= y + h / 2 <= sy1 and passes_filter(a):
                    state["sel"].add(a)
        render()

    def on_wheel(event):
        zoom_in = getattr(event, "delta", 0) > 0 or event.num == 4
        factor = 1.25 if zoom_in else 0.8
        bx, by = display_to_base(event.x, event.y)
        state["z"] = max(z_fit(canvas.winfo_width(), canvas.winfo_height()),
                         min(state["z"] * factor, Z_MAX))
        state["ox"] = bx - event.x / state["z"]
        state["oy"] = by - event.y / state["z"]
        render()

    def on_pan_press(event):
        state["pan"] = (event.x, event.y)

    def on_pan_move(event):
        if not state["pan"]:
            return
        lx, ly = state["pan"]
        state["ox"] -= (event.x - lx) / state["z"]
        state["oy"] -= (event.y - ly) / state["z"]
        state["pan"] = (event.x, event.y)
        render()

    def pan_keys(ddx, ddy):
        state["ox"] += ddx / state["z"]
        state["oy"] += ddy / state["z"]
        render()

    def fit():
        state["z"] = None
        state["ox"] = state["oy"] = 0.0
        render()

    def apply_label(label):
        if state["sel"]:
            n = len(state["sel"])
            labeler.set_many(list(state["sel"]), label=label)
            # With an active filter, tiles that leave the set under review (the
            # new label is no longer in the filter) only lose their selection
            # mark -- there is no new color to paint here (the overlay is static).
            if class_filter is not None:
                state["sel"] = {a for a in state["sel"] if passes_filter(a)}
            render()
            update_status(f"-> {n} tile(s) = {label}")

    def apply_border(value):
        if state["sel"]:
            labeler.set_many(list(state["sel"]), border=value)
            update_status(f"-> border={value} on {len(state['sel'])} tile(s)")

    def toggle_overlay():
        state["overlay"] = not state["overlay"]
        render()

    def clear_selection():
        state["sel"] = set()
        render()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<ButtonPress-3>", on_pan_press)
    canvas.bind("<B3-Motion>", on_pan_move)
    for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        canvas.bind(seq, on_wheel)
    canvas.bind("<Configure>", lambda e: render())

    for key, label in LABEL_KEYS:
        window.bind(key, lambda e, r=label: apply_label(r))
    window.bind("b", lambda e: apply_border(1))
    window.bind("n", lambda e: apply_border(0))
    window.bind("o", lambda e: toggle_overlay())
    window.bind("O", lambda e: toggle_overlay())
    window.bind("<Left>", lambda e: pan_keys(80, 0))
    window.bind("<Right>", lambda e: pan_keys(-80, 0))
    window.bind("<Up>", lambda e: pan_keys(0, 80))
    window.bind("<Down>", lambda e: pan_keys(0, -80))
    window.bind("0", lambda e: fit())
    window.bind("<Escape>", lambda e: clear_selection())
    window.bind("q", lambda e: window.destroy())

    window.after(50, render)
    window.mainloop()


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
    p = argparse.ArgumentParser(description="Visual label correction.")
    p.add_argument("prefix", nargs="?", help="Image prefix (e.g. sat1)")
    p.add_argument("--tiles", help="Path to the tiles folder (alternative to the prefix)")
    p.add_argument("--scale", type=float, default=0.25, help="Display scale (default 0.25)")
    p.add_argument("--classes", nargs="+", default=None,
                   help="Restricts the overlay and selection to tiles labeled with these "
                        "classes (e.g.: --classes uncertain)")
    args = p.parse_args()

    if args.classes is not None:
        invalid = [c for c in args.classes if c not in VALID_LABELS]
        if invalid:
            print(f"ERROR: invalid classes in --classes: {invalid}. "
                  f"Valid: {sorted(VALID_LABELS)}")
            sys.exit(1)

    tiles_dir, prefix = resolve(args)
    if not os.path.isdir(tiles_dir):
        print(f"ERROR: tiles folder not found: {tiles_dir}")
        sys.exit(1)

    try:
        coords, step, scene_w, scene_h = read_manifest(tiles_dir, prefix)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    labeler = Labeler(tiles_dir, prefix)
    out_w, out_h = round(scene_w * args.scale), round(scene_h * args.scale)
    print(f"{prefix}: {len(coords)} tiles | scene {scene_w}x{scene_h} "
          f"-> display {out_w}x{out_h}")

    base_real = get_base(prefix, tiles_dir, coords, args.scale, out_w, out_h)
    print("Building class overlay...")
    base_over = build_overlay(base_real, labeler, coords, args.scale, step,
                              class_filter=args.classes)

    run_gui(prefix, labeler, coords, args.scale, step, base_real, base_over,
            class_filter=args.classes)


if __name__ == "__main__":
    main()
