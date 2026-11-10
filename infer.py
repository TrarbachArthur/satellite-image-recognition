#!/usr/bin/env python3
"""Pre-labeling of satellite tiles by inference with already trained models.

Reads the manifest of a tiles folder (produced by generate_tiles.py), runs
inference with one or more trained experiments (ensemble by averaging the
softmax probabilities when there is more than one) and writes a label CSV in the
SAME format label.py produces (file,label,border,timestamp), ready for manual
review with fix_labels.py.

NEVER overwrites an existing <prefix>_labels.csv without --overwrite: that file
may hold manual labeling and the guard prevents losing work.

Never hardcodes counts, scenes or experiments -- everything comes from the tile
manifest and from the experiment directories given on the command line.

Usage:
    python3 infer.py sat3 --experiments experiments/vit_small_pretrained
    python3 infer.py --tiles data/sat3_tiles --experiments experiments/resnet18_base experiments/vit_small_pretrained
    python3 infer.py sat3 --experiments experiments/resnet18_base --uncertain-threshold 0.6
    python3 infer.py sat3 --experiments experiments/resnet18_base --overwrite
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image

from dataset import CLASSES, IDX, DATA_DIR, create_transforms

RED = "\033[91m"
COLOR_RESET = "\033[0m"

LABELS_HEADER = ["file", "label", "border", "timestamp"]


def resolve_tiles_dir(args):
    """Determines the tiles folder and the prefix from the arguments (same as label.py)."""
    if args.tiles:
        tiles_dir = os.path.abspath(args.tiles)
        prefix = args.prefix or os.path.basename(tiles_dir.rstrip("/")).replace("_tiles", "")
    else:
        if not args.prefix:
            args.prefix = input("Enter the image prefix (e.g. sat2): ").strip()
        prefix = args.prefix
        tiles_dir = os.path.join(DATA_DIR, f"{prefix}_tiles")
    return tiles_dir, prefix


class UnlabeledTiles(torch.utils.data.Dataset):
    """Minimal dataset that opens each tile on demand and returns only the image
    tensor (no label -- used for inference on tiles not labeled yet)."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


def load_manifest(tiles_dir, prefix):
    """Reads <tiles_dir>/<prefix>_tiles.csv (exact name) and keeps only the
    images that still exist on disk. Returns the list of file names (in manifest order)."""
    manifest_path = os.path.join(tiles_dir, f"{prefix}_tiles.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"tile manifest not found: {manifest_path}. "
            f"Generate the tiles with 'python3 generate_tiles.py {prefix}' before inferring."
        )

    df_manifest = pd.read_csv(manifest_path, dtype={"file": str})
    if "file" not in df_manifest.columns:
        raise ValueError(
            f"{manifest_path} does not have the 'file' column. "
            f"Columns found: {list(df_manifest.columns)}."
        )

    names = df_manifest["file"].tolist()
    existing = [n for n in names if os.path.exists(os.path.join(tiles_dir, n))]
    missing = len(names) - len(existing)
    if missing > 0:
        print(f"WARNING: {missing} tile(s) listed in the manifest were not "
              f"found on disk and will be ignored.")

    if not existing:
        raise ValueError(f"No tile from manifest {manifest_path} exists on disk.")

    return existing


def check_labels_protection(labels_csv, overwrite):
    if os.path.exists(labels_csv) and not overwrite:
        raise SystemExit(
            f"ERROR: {labels_csv} already exists and may contain manual labeling; "
            f"use --overwrite if you are sure you want to replace it."
        )


def validate_experiments(exp_dirs, checkpoint_name):
    """Checks that each experiment directory exists and has <checkpoint>.pt."""
    for exp_dir in exp_dirs:
        if not os.path.isdir(exp_dir):
            raise SystemExit(f"ERROR: experiment directory not found: {exp_dir}")
        ckpt_path = os.path.join(exp_dir, f"{checkpoint_name}.pt")
        if not os.path.exists(ckpt_path):
            raise SystemExit(
                f"ERROR: checkpoint '{checkpoint_name}.pt' not found in {exp_dir}."
            )


def _build_model_from_checkpoint(ckpt, device):
    cfg = ckpt["config"]
    architecture = cfg["model"]["architecture"]
    try:
        model = timm.create_model(architecture, pretrained=False, num_classes=len(CLASSES))
    except Exception as e:
        raise SystemExit(
            f"ERROR: failed to create model '{architecture}': {e}\n"
            f"Check that the architecture name is valid for timm."
        )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def _infer_probs(model, loader, device, n_tiles, n_classes):
    """Runs softmax inference over the whole loader. Returns a float32 numpy
    array (n_tiles, n_classes) on the CPU, in loader order (shuffle=False)."""
    all_probs = np.zeros((n_tiles, n_classes), dtype=np.float32)
    offset = 0
    for i, imgs in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        outputs = model(imgs)
        probs = torch.nn.functional.softmax(outputs.float(), dim=1).cpu().numpy()
        n = probs.shape[0]
        all_probs[offset:offset + n] = probs
        offset += n
        if (i + 1) % 50 == 0:
            print(f"    batch {i + 1}: {offset}/{n_tiles} tiles processed")
    return all_probs


def run_experiment(exp_dir, checkpoint_name, paths, device, batch, num_workers):
    """Loads an experiment, runs inference on all tiles and returns the softmax
    probabilities (n_tiles, n_classes) as float32 on the CPU. Frees the model
    from VRAM at the end."""
    ckpt_path = os.path.join(exp_dir, f"{checkpoint_name}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, cfg = _build_model_from_checkpoint(ckpt, device)

    img_size = cfg["data"]["img_size"]
    transform = create_transforms("none", img_size, train=False)
    ds = UnlabeledTiles(paths, transform)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    exp_name = os.path.basename(exp_dir)
    print(f"  experiment '{exp_name}': architecture={cfg['model']['architecture']} "
          f"img_size={img_size} checkpoint={checkpoint_name}.pt")
    t0 = time.time()
    probs = _infer_probs(model, loader, device, len(paths), len(CLASSES))
    elapsed = time.time() - t0
    print(f"  experiment '{exp_name}' finished in {elapsed:.1f}s "
          f"({len(paths) / elapsed:.1f} tiles/s)" if elapsed > 0 else
          f"  experiment '{exp_name}' finished in {elapsed:.1f}s")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return probs


def write_labels_csv(labels_csv, names, labels):
    """Writes <prefix>_labels.csv in the same format as label.py: atomic write
    (temp + os.replace), header file,label,border,timestamp."""
    tmp = labels_csv + ".tmp"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LABELS_HEADER)
        for name, label in zip(names, labels):
            w.writerow([name, label, 0, ts])
    os.replace(tmp, labels_csv)


def write_predictions_csv(csv_path, names, predicted, confidences, final_prob):
    tmp = csv_path + ".tmp"
    df_out = pd.DataFrame({
        "file": names,
        "predicted": predicted,
        "confidence": confidences,
    })
    for c in CLASSES:
        df_out[f"prob_{c}"] = final_prob[:, IDX[c]]
    df_out.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)


def main():
    p = argparse.ArgumentParser(
        description="Pre-labeling of satellite tiles by inference with already trained models."
    )
    p.add_argument("prefix", nargs="?", default=None,
                   help="Scene prefix (e.g. sat3); tiles in data/<prefix>_tiles")
    p.add_argument("--tiles", default=None,
                   help="Alternative tiles folder (prefix derived from the folder name)")
    p.add_argument("--experiments", nargs="+", required=True,
                   help="1+ trained experiment directories (ensemble if more than one)")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"],
                   help="Which checkpoint to load from each experiment (default: best)")
    p.add_argument("--uncertain-threshold", type=float, default=0.0,
                   help="0.0 = off. If >0, tiles whose max ensemble probability is below "
                        "this value get the label 'uncertain' (default: 0.0)")
    p.add_argument("--batch", type=int, default=128, help="Inference batch size (default: 128)")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2)")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Allow inferring on CPU when CUDA is unavailable")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite <prefix>_labels.csv even if it already exists")
    args = p.parse_args()

    tiles_dir, prefix = resolve_tiles_dir(args)
    if not os.path.isdir(tiles_dir):
        print(f"ERROR: tiles folder not found: {tiles_dir}")
        sys.exit(1)

    try:
        names = load_manifest(tiles_dir, prefix)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    paths = [os.path.join(tiles_dir, n) for n in names]
    print(f"manifest: {len(names)} tile(s) found in {tiles_dir}")

    # Checked before inference so no time is wasted if the file is protected.
    labels_csv = os.path.join(tiles_dir, f"{prefix}_labels.csv")
    try:
        check_labels_protection(labels_csv, args.overwrite)
    except SystemExit as e:
        print(e)
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        print(f"{RED}WARNING: CUDA is not available in this environment. Inferring on CPU is "
              f"slower.{COLOR_RESET}")
        if not args.allow_cpu:
            print("ERROR: run again with --allow-cpu to force execution on CPU.")
            sys.exit(1)
        device = "cpu"
    print(f"device: {device}")

    try:
        validate_experiments(args.experiments, args.checkpoint)
    except SystemExit as e:
        print(e)
        sys.exit(1)

    # One sequential pass per experiment (ensemble by summing the softmax
    # probabilities, one model at a time to save VRAM).
    probs_sum = np.zeros((len(paths), len(CLASSES)), dtype=np.float32)
    print(f"\nrunning inference with {len(args.experiments)} experiment(s):")
    for exp_dir in args.experiments:
        probs = run_experiment(
            exp_dir, args.checkpoint, paths, device, args.batch, args.num_workers
        )
        probs_sum += probs

    final_prob = probs_sum / len(args.experiments)

    argmax_idx = final_prob.argmax(axis=1)
    confidences = final_prob[np.arange(len(paths)), argmax_idx]
    predicted = [CLASSES[i] for i in argmax_idx]

    if args.uncertain_threshold > 0.0:
        labels = [
            p if c >= args.uncertain_threshold else "uncertain"
            for p, c in zip(predicted, confidences)
        ]
    else:
        labels = list(predicted)

    write_labels_csv(labels_csv, names, labels)
    predictions_csv = os.path.join(tiles_dir, f"{prefix}_auto_predictions.csv")
    write_predictions_csv(predictions_csv, names, predicted, confidences, final_prob)

    print(f"\nsaved: {labels_csv}")
    print(f"saved: {predictions_csv}")

    distribution = pd.Series(labels).value_counts().to_dict()
    n_object_candidates = int((final_prob[:, IDX["object"]] >= 0.5).sum()) if "object" in IDX else 0

    print(f"\n=== summary ({prefix}) ===")
    print(f"n tiles: {len(paths)}")
    print(f"distribution of the written labels: {distribution}")
    print(f"mean confidence (max ensemble probability): {confidences.mean():.4f}")
    print(f"'object' candidates (prob_object >= 0.5): {n_object_candidates}")
    print(f"\nsuggested next steps:")
    print(f"  python3 visualize.py {prefix}    # visualize the generated labeling")
    print(f"  python3 fix_labels.py {prefix}   # review/fix uncertain or misclassified tiles")


if __name__ == "__main__":
    main()
