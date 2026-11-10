#!/usr/bin/env python3
"""Quick labeling app for satellite tiles.

Loads the tiles of a folder (left->right, top->bottom) and lets you label each
one with a key/button (sea, land, cloud, object, uncertain), plus an independent
"border" flag. Labels AND advances with a single keypress; going back to fix a
label is allowed. Progress is saved continuously to <prefix>_labels.csv inside
the tiles folder, so you can stop and resume where you left off.

Usage:
    python3 label.py sat2                 # uses data/sat2_tiles/
    python3 label.py --tiles data/sat2_tiles
"""

import argparse
import csv
import os
import sys
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# key -> label. The order also defines the order of the buttons shown.
LABEL_KEYS = [
    ("1", "sea"),
    ("2", "land"),
    ("3", "cloud"),
    ("4", "object"),
    ("5", "uncertain"),
]
VALID_LABELS = {label for _, label in LABEL_KEYS}
CSV_HEADER = ["file", "label", "border", "timestamp"]


class Labeler:
    """Manages the ordered tile list and label persistence (no GUI)."""

    def __init__(self, tiles_dir, prefix, classes=None):
        self.tiles_dir = tiles_dir
        self.prefix = prefix
        self.manifest_csv = os.path.join(tiles_dir, f"{prefix}_tiles.csv")
        self.labels_csv = os.path.join(tiles_dir, f"{prefix}_labels.csv")
        self.class_filter = classes
        self.labels = self._load_labels()  # {file: {"label", "border"}}
        # all_tiles is the COMPLETE ordered list and the basis for persistence
        # (_save), independent of any navigation filter.
        self.all_tiles = self._load_order()
        self.tiles = self.all_tiles  # navigation view (filterable)
        if classes is not None:
            self.tiles = [n for n in self.all_tiles
                          if self.labels.get(n, {}).get("label") in classes]
        self.index = 0 if classes is not None else self.first_unlabeled()

    def _load_order(self):
        """Reads the manifest and sorts by (y, x): top->bottom, left->right.
        Without a manifest, falls back to the alphabetical order of the images in the folder."""
        if os.path.exists(self.manifest_csv):
            rows = []
            with open(self.manifest_csv, newline="") as f:
                for row in csv.DictReader(f):
                    rows.append((int(row["y"]), int(row["x"]), row["file"]))
            rows.sort(key=lambda t: (t[0], t[1]))
            names = [a for _, _, a in rows]
        else:
            names = sorted(x for x in os.listdir(self.tiles_dir)
                           if x.lower().endswith((".png", ".jpg", ".jpeg")))
        return [n for n in names if os.path.exists(os.path.join(self.tiles_dir, n))]

    def _load_labels(self):
        d = {}
        if os.path.exists(self.labels_csv):
            with open(self.labels_csv, newline="") as f:
                for row in csv.DictReader(f):
                    d[row["file"]] = {
                        "label": row.get("label", "") or "",
                        "border": 1 if str(row.get("border", "0")) in ("1", "True", "true") else 0,
                    }
        return d

    def total(self):
        return len(self.tiles)

    def total_labeled(self):
        return sum(1 for n in self.tiles
                   if self.labels.get(n, {}).get("label"))

    def first_unlabeled(self):
        for i, n in enumerate(self.tiles):
            if not self.labels.get(n, {}).get("label"):
                return i
        return max(0, len(self.tiles) - 1)  # everything labeled: stay on the last one

    def current_name(self, index=None):
        i = self.index if index is None else index
        return self.tiles[i] if 0 <= i < len(self.tiles) else None

    def path(self, name):
        return os.path.join(self.tiles_dir, name)

    def state(self, name):
        """Returns the saved (label, border) of a tile (or ('', 0))."""
        e = self.labels.get(name)
        return (e["label"], e["border"]) if e else ("", 0)

    def set_label(self, name, label, border):
        assert label in VALID_LABELS, f"invalid label: {label}"
        self.labels[name] = {"label": label, "border": int(bool(border))}
        self._save()

    def set_border(self, name, border):
        """Updates the border; if the tile already has a label, persists right away."""
        e = self.labels.get(name)
        if e:
            e["border"] = int(bool(border))
            self._save()

    def set_many(self, names, label=None, border=None):
        """Applies label and/or border to several tiles at once, saving the CSV
        a single time at the end (avoids N rewrites when fixing a batch)."""
        if label is not None:
            assert label in VALID_LABELS, f"invalid label: {label}"
        for name in names:
            e = self.labels.get(name, {"label": "", "border": 0})
            if label is not None:
                e["label"] = label
            if border is not None:
                e["border"] = int(bool(border))
            self.labels[name] = e
        self._save()

    def _save(self):
        """Rewrites the whole CSV atomically (temp + replace).

        Iterates over all_tiles, NEVER over self.tiles: with a class filter
        active, self.tiles is a subset and saving from it would erase the labels
        of every other class."""
        tmp = self.labels_csv + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            for name in self.all_tiles:
                e = self.labels.get(name)
                if e and e.get("label"):
                    w.writerow([name, e["label"], e["border"], ts])
        os.replace(tmp, self.labels_csv)

    def advance(self):
        if self.index < len(self.tiles) - 1:
            self.index += 1

    def back(self):
        if self.index > 0:
            self.index -= 1


def resolve_tiles_dir(args):
    """Determines the tiles folder and the prefix from the arguments."""
    if args.tiles:
        tiles_dir = os.path.abspath(args.tiles)
        prefix = args.prefix or os.path.basename(tiles_dir.rstrip("/")).replace("_tiles", "")
    else:
        if not args.prefix:
            args.prefix = input("Enter the image prefix (e.g. sat2): ").strip()
        prefix = args.prefix
        tiles_dir = os.path.join(DATA_DIR, f"{prefix}_tiles")
    return tiles_dir, prefix


def run_gui(labeler):
    import tkinter as tk
    from PIL import Image, ImageTk

    DISPLAY_SIZE = 750  # max side of the displayed image

    window = tk.Tk()
    window.title(f"Labeling — {labeler.prefix}")
    window.configure(bg="#222")

    state = {"border": 0, "cache_name": None, "cache_img": None, "zoom": None}

    lbl_top = tk.Label(window, font=("TkDefaultFont", 12, "bold"),
                       bg="#222", fg="#eee", pady=6)
    lbl_top.pack(fill="x")

    lbl_sub = tk.Label(window, font=("TkDefaultFont", 10),
                       bg="#222", fg="#aaa")
    lbl_sub.pack(fill="x")

    # frame that turns red when border=1
    frame = tk.Frame(window, bg="#222", bd=6)
    frame.pack(padx=10, pady=8)
    image_label = tk.Label(frame, bg="#000")
    image_label.pack()

    lbl_border = tk.Label(window, font=("TkDefaultFont", 11, "bold"),
                          bg="#222", pady=2)
    lbl_border.pack(fill="x")

    bar = tk.Frame(window, bg="#222")
    bar.pack(pady=8)

    def load_image(name):
        """Loads and resizes the tile for display, with a 1-entry cache."""
        if state["cache_name"] == name and state["cache_img"] is not None:
            return state["cache_img"]
        img = Image.open(labeler.path(name)).convert("RGB")
        img.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE))
        tkimg = ImageTk.PhotoImage(img)
        state["cache_name"] = name
        state["cache_img"] = tkimg
        return tkimg

    def prefetch(index):
        """Preloads the next image into the cache for an instant advance."""
        nxt = labeler.current_name(index + 1)
        if nxt and state["cache_name"] != nxt:
            try:
                img = Image.open(labeler.path(nxt)).convert("RGB")
                img.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE))
                state["cache_name"] = nxt
                state["cache_img"] = ImageTk.PhotoImage(img)
            except Exception:
                pass

    def refresh():
        name = labeler.current_name()
        if name is None:
            return
        saved_label, saved_border = labeler.state(name)
        state["border"] = saved_border

        done = labeler.total_labeled()
        total = labeler.total()
        pct = (100.0 * done / total) if total else 0.0
        filter_suffix = f"  [filter: {','.join(labeler.class_filter)}]" if labeler.class_filter else ""
        lbl_top.config(text=f"{labeler.prefix} — {done}/{total} labeled ({pct:.1f}%){filter_suffix}")

        mark = f"  •  current label: {saved_label.upper()}" if saved_label else "  •  (no label)"
        lbl_sub.config(text=f"tile {labeler.index + 1}/{total}   {name}{mark}")

        try:
            tkimg = load_image(name)
            image_label.config(image=tkimg, text="")
            image_label.image = tkimg
        except Exception as e:
            image_label.config(image="", text=f"[error opening {name}]\n{e}", fg="#f88")

        refresh_border()
        window.after(1, lambda: prefetch(labeler.index))

    def refresh_border():
        if state["border"]:
            frame.config(bg="#e33")
            lbl_border.config(text="BORDER: YES  (B to toggle)", fg="#f66")
        else:
            frame.config(bg="#222")
            lbl_border.config(text="border: no  (B to toggle)", fg="#888")

    def label_tile(label):
        name = labeler.current_name()
        if name is None:
            return
        labeler.set_label(name, label, state["border"])
        labeler.advance()
        refresh()

    def toggle_border():
        name = labeler.current_name()
        if name is None:
            return
        state["border"] = 0 if state["border"] else 1
        labeler.set_border(name, state["border"])  # persists if it already has a label
        refresh_border()

    def go_back():
        labeler.back()
        refresh()

    def go_next():
        labeler.advance()
        refresh()

    def zoom():
        name = labeler.current_name()
        if name is None:
            return
        if state["zoom"] is not None and tk.Toplevel.winfo_exists(state["zoom"]):
            state["zoom"].destroy()
            state["zoom"] = None
            return
        top = tk.Toplevel(window)
        top.title(f"Zoom 1:1 — {name}")
        img = Image.open(labeler.path(name)).convert("RGB")
        tkimg = ImageTk.PhotoImage(img)
        cv = tk.Canvas(top, width=min(img.width, 1000), height=min(img.height, 800),
                       bg="#000")
        cv.grid(row=0, column=0, sticky="nsew")
        sx = tk.Scrollbar(top, orient="horizontal", command=cv.xview)
        sy = tk.Scrollbar(top, orient="vertical", command=cv.yview)
        cv.config(xscrollcommand=sx.set, yscrollcommand=sy.set,
                  scrollregion=(0, 0, img.width, img.height))
        sx.grid(row=1, column=0, sticky="ew")
        sy.grid(row=0, column=1, sticky="ns")
        cv.create_image(0, 0, anchor="nw", image=tkimg)
        cv.image = tkimg
        top.bind("<Escape>", lambda e: top.destroy())
        state["zoom"] = top

    def quit_app(*_):
        window.destroy()

    for key, label in LABEL_KEYS:
        tk.Button(bar, text=f"{key} {label.capitalize()}", width=9,
                  command=lambda r=label: label_tile(r)).pack(side="left", padx=3)
    tk.Label(bar, text="  ", bg="#222").pack(side="left")
    tk.Button(bar, text="B Border", width=8, command=toggle_border).pack(side="left", padx=3)
    tk.Button(bar, text="← Back", width=8, command=go_back).pack(side="left", padx=3)
    tk.Button(bar, text="→ Next", width=9, command=go_next).pack(side="left", padx=3)
    tk.Button(bar, text="Z Zoom", width=8, command=zoom).pack(side="left", padx=3)

    for key, label in LABEL_KEYS:
        window.bind(key, lambda e, r=label: label_tile(r))
    window.bind("b", lambda e: toggle_border())
    window.bind("B", lambda e: toggle_border())
    window.bind("<Left>", lambda e: go_back())
    window.bind("<BackSpace>", lambda e: go_back())
    window.bind("<Right>", lambda e: go_next())
    window.bind("z", lambda e: zoom())
    window.bind("Z", lambda e: zoom())
    window.bind("<Escape>", quit_app)
    window.bind("q", quit_app)

    refresh()
    window.mainloop()


def main():
    p = argparse.ArgumentParser(description="Quick labeling of satellite tiles.")
    p.add_argument("prefix", nargs="?", help="Image prefix (e.g. sat2)")
    p.add_argument("--tiles", help="Path to the tiles folder (alternative to the prefix)")
    p.add_argument("--classes", nargs="+", default=None,
                   help="Filter and show only tiles already labeled with these classes "
                        "(e.g.: --classes uncertain)")
    args = p.parse_args()

    if args.classes is not None:
        invalid = [c for c in args.classes if c not in VALID_LABELS]
        if invalid:
            print(f"ERROR: invalid classes in --classes: {invalid}. "
                  f"Valid: {sorted(VALID_LABELS)}")
            sys.exit(1)

    tiles_dir, prefix = resolve_tiles_dir(args)
    if not os.path.isdir(tiles_dir):
        print(f"ERROR: tiles folder not found: {tiles_dir}")
        sys.exit(1)

    labeler = Labeler(tiles_dir, prefix, classes=args.classes)
    if labeler.total() == 0:
        if args.classes is not None:
            print(f"ERROR: no tile labeled with {args.classes}")
        else:
            print(f"ERROR: no tiles found in {tiles_dir}")
        sys.exit(1)

    print(f"Labeling '{prefix}': {labeler.total_labeled()}/{labeler.total()} already labeled. "
          f"Starting at tile {labeler.index + 1}. Labels saved to {labeler.labels_csv}")
    run_gui(labeler)


if __name__ == "__main__":
    main()
