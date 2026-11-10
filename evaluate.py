#!/usr/bin/env python3
"""Evaluation script for the satellite tile classifier.

Mode 1 -- evaluates an already trained experiment (reads config.yaml and
splits.csv from the experiment's own directory, rebuilds the model with timm,
loads a checkpoint and runs inference on the requested split). Saves metrics,
confusion matrix and predictions inside experiments/<name>/.

Mode 2 -- compares every experiment in experiments/*/ that has a
metrics_<split>.json (written by this script) or metrics_val.json (written by
train.py), and prints/saves a table sorted by f1_macro.

Never hardcodes counts, scenes or present classes -- everything comes from the
experiment artifacts (config.yaml, splits.csv, checkpoints).

Usage:
    python3 evaluate.py experiments/smoke
    python3 evaluate.py experiments/smoke --split val --checkpoint last
    python3 evaluate.py experiments/smoke --allow-cpu
    python3 evaluate.py --compare
    python3 evaluate.py --compare --split val --csv comparison.csv
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from dataset import CLASSES, IDX, DATA_DIR, TilesDataset, create_transforms

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")

RED = "\033[91m"
COLOR_RESET = "\033[0m"


def _load_config_and_splits(exp_dir):
    config_path = os.path.join(exp_dir, "config.yaml")
    splits_path = os.path.join(exp_dir, "splits.csv")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found in {exp_dir}")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"splits.csv not found in {exp_dir}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(splits_path, dtype={"file": str, "label": str, "scene": str, "split": str})
    missing = {"file", "label", "scene", "split"} - set(df.columns)
    if missing:
        raise ValueError(f"splits.csv in {exp_dir} is missing the expected columns: {sorted(missing)}.")

    df["path"] = df.apply(
        lambda r: os.path.join(DATA_DIR, f"{r['scene']}_tiles", r["file"]), axis=1
    )
    return cfg, df


def _build_model(cfg, device):
    try:
        model = timm.create_model(
            cfg["model"]["architecture"],
            pretrained=False,
            num_classes=len(CLASSES),
        )
    except Exception as e:
        raise SystemExit(
            f"ERROR: failed to create model '{cfg['model']['architecture']}': {e}\n"
            f"Check that the architecture name is valid for timm."
        )
    return model.to(device)


def _load_checkpoint(exp_dir, checkpoint_name, device):
    path = os.path.join(exp_dir, f"{checkpoint_name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"checkpoint '{checkpoint_name}.pt' not found in {exp_dir}."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt, path


@torch.no_grad()
def _infer(model, loader, device):
    """Runs (softmax) inference on the loader. Returns numpy arrays y_true, y_pred, y_prob."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        outputs = model(imgs)
        probs = F.softmax(outputs.float(), dim=1)
        y_true.append(labels.numpy())
        y_pred.append(probs.argmax(dim=1).cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true) if y_true else np.zeros(0, dtype=int)
    y_pred = np.concatenate(y_pred) if y_pred else np.zeros(0, dtype=int)
    y_prob = np.concatenate(y_prob) if y_prob else np.zeros((0, len(CLASSES)))
    return y_true, y_pred, y_prob


def _compute_metrics(y_true, y_pred, y_prob, present_classes, epoch, checkpoint_used, n_split):
    all_labels = list(range(len(CLASSES)))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0))

    present_idx = [IDX[c] for c in present_classes]
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=present_idx, average=None, zero_division=0
    )
    per_class = {}
    for c, p, r, f1c, s in zip(present_classes, precisions, recalls, f1s, supports):
        per_class[c] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1c),
            "support": int(s),
        }

    ap_object = None
    if "object" in present_classes:
        y_true_bin = (y_true == IDX["object"]).astype(int)
        ap_object = float(average_precision_score(y_true_bin, y_prob[:, IDX["object"]]))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "per_class": per_class,
        "average_precision_object": ap_object,
        "checkpoint_epoch": epoch,
        "checkpoint_used": checkpoint_used,
        "n": n_split,
    }


def _save_confusion_matrix(y_true, y_pred, csv_path):
    all_labels = list(range(len(CLASSES)))
    matrix = confusion_matrix(y_true, y_pred, labels=all_labels)
    df_matrix = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
    df_matrix.index.name = "true\\predicted"
    df_matrix.to_csv(csv_path)


def _save_predictions(df_split, y_pred, y_prob, csv_path):
    out = pd.DataFrame({
        "file": df_split["file"].tolist(),
        "scene": df_split["scene"].tolist(),
        "label": df_split["label"].tolist(),
        "predicted": [CLASSES[i] for i in y_pred],
    })
    for c in CLASSES:
        out[f"prob_{c}"] = y_prob[:, IDX[c]]
    out.to_csv(csv_path, index=False)


def _print_metrics(metrics, split, exp_name):
    print(f"\n=== metrics ({exp_name}, split={split}) ===")
    print(f"checkpoint: {metrics['checkpoint_used']}  (epoch {metrics['checkpoint_epoch']})")
    print(f"n: {metrics['n']}")
    print(f"accuracy:          {metrics['accuracy']:.4f}")
    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"f1_macro:          {metrics['f1_macro']:.4f}")
    if metrics["average_precision_object"] is not None:
        print(f"average_precision (object, one-vs-rest): {metrics['average_precision_object']:.4f}")
    else:
        print("average_precision (object): N/A (class 'object' absent from the split)")
    print("\nper class:")
    for c, m in metrics["per_class"].items():
        print(f"  {c:8s} precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"f1={m['f1']:.4f}  support={m['support']}")


def evaluate_mode(args):
    exp_dir = os.path.normpath(args.experiment)
    exp_name = os.path.basename(exp_dir)

    if not os.path.isdir(exp_dir):
        print(f"ERROR: experiment directory not found: {exp_dir}")
        sys.exit(1)

    try:
        cfg, df = _load_config_and_splits(exp_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        print(f"{RED}WARNING: CUDA is not available in this environment. Evaluating on CPU is "
              f"slower.{COLOR_RESET}")
        if not args.allow_cpu:
            print("ERROR: run again with --allow-cpu to force execution on CPU.")
            sys.exit(1)
        device = "cpu"
    print(f"device: {device}")

    df_split = df[df["split"] == args.split].reset_index(drop=True)
    if len(df_split) == 0:
        print(f"ERROR: split '{args.split}' has no examples in {exp_dir}/splits.csv.")
        sys.exit(1)

    missing_files = [p for p in df_split["path"] if not os.path.exists(p)]
    if missing_files:
        examples = missing_files[:3]
        print(f"ERROR: {len(missing_files)} tile(s) of split '{args.split}' were not "
              f"found on disk (e.g.: {examples}). Check DATA_DIR in dataset.py.")
        sys.exit(1)

    present_classes = [c for c in CLASSES if (df_split["label"] == c).any()]

    model = _build_model(cfg, device)
    try:
        ckpt, ckpt_path = _load_checkpoint(exp_dir, args.checkpoint, device)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    model.load_state_dict(ckpt["state_dict"])
    ckpt_epoch = ckpt["epoch"]

    img_size = cfg["data"]["img_size"]
    eval_transform = create_transforms("none", img_size, train=False)
    ds_split = TilesDataset(df_split, eval_transform)
    dl_split = torch.utils.data.DataLoader(
        ds_split,
        batch_size=cfg["training"]["batch"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    y_true, y_pred, y_prob = _infer(model, dl_split, device)

    metrics = _compute_metrics(
        y_true, y_pred, y_prob, present_classes,
        epoch=ckpt_epoch, checkpoint_used=f"{args.checkpoint}.pt", n_split=len(df_split),
    )

    metrics_path = os.path.join(exp_dir, f"metrics_{args.split}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    matrix_path = os.path.join(exp_dir, f"confusion_matrix_{args.split}.csv")
    _save_confusion_matrix(y_true, y_pred, matrix_path)

    predictions_path = os.path.join(exp_dir, f"predictions_{args.split}.csv")
    _save_predictions(df_split, y_pred, y_prob, predictions_path)

    print(f"\nsaved: {metrics_path}")
    print(f"saved: {matrix_path}")
    print(f"saved: {predictions_path}")

    _print_metrics(metrics, args.split, exp_name)


def _experiment_epochs(exp_dir):
    """Number of epochs run (rows of history.csv), if it exists."""
    history_path = os.path.join(exp_dir, "history.csv")
    if not os.path.exists(history_path):
        return "-"
    try:
        df_hist = pd.read_csv(history_path)
        return int(len(df_hist))
    except Exception:
        return "-"


def _split_metrics_row(exp_name, exp_dir, split, metrics):
    """Format produced by this script (evaluate_mode): metrics_<split>.json."""
    per_class = metrics.get("per_class", {})
    m_object = per_class.get("object", {})
    return {
        "experiment": exp_name,
        "split": split,
        "acc": metrics.get("accuracy", np.nan),
        "bal_acc": metrics.get("balanced_accuracy", np.nan),
        "f1_macro": metrics.get("f1_macro", np.nan),
        "rec_object": m_object.get("recall", np.nan),
        "prec_object": m_object.get("precision", np.nan),
        "ap_object": metrics.get("average_precision_object", np.nan),
        "n_object": m_object.get("support", 0),
        "epochs": _experiment_epochs(exp_dir),
    }


def _train_val_metrics_row(exp_name, exp_dir, metrics):
    """Format produced by train.py: metrics_val.json (no precision/ap, nested
    recall_per_class). The 'object' support comes from splits_summary.json."""
    recall_per_class = metrics.get("recall_per_class", {})

    # Real 'object' support in val (a fixed 0 would be misleading: it would look
    # like "no objects in the split" when it is actually "unknown").
    n_object = np.nan
    summary_path = os.path.join(exp_dir, "splits_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as f:
                n_object = json.load(f).get("val", {}).get("object", 0)
        except Exception:
            pass

    return {
        "experiment": exp_name,
        "split": "val",
        "acc": metrics.get("acc", np.nan),
        "bal_acc": metrics.get("balanced_accuracy", np.nan),
        "f1_macro": metrics.get("f1_macro", np.nan),
        "rec_object": recall_per_class.get("object", np.nan),
        "prec_object": np.nan,
        "ap_object": np.nan,
        "n_object": n_object,
        "epochs": metrics.get("best_epoch", _experiment_epochs(exp_dir)),
    }


def compare_mode(args):
    exp_dirs = sorted(d for d in glob.glob(os.path.join(EXPERIMENTS_DIR, "*")) if os.path.isdir(d))
    if not exp_dirs:
        print(f"ERROR: no experiments found in {EXPERIMENTS_DIR}.")
        sys.exit(1)

    rows = []
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)

        split_path = os.path.join(exp_dir, f"metrics_{args.split}.json")
        if os.path.exists(split_path):
            with open(split_path, encoding="utf-8") as f:
                metrics = json.load(f)
            # metrics_val.json may have been written by train.py (keys 'acc'/
            # 'recall_per_class') instead of this script ('accuracy'/'per_class'),
            # so detect the format from the keys before interpreting it.
            if "accuracy" in metrics:
                rows.append(_split_metrics_row(exp_name, exp_dir, args.split, metrics))
            else:
                rows.append(_train_val_metrics_row(exp_name, exp_dir, metrics))
            continue

        train_val_path = os.path.join(exp_dir, "metrics_val.json")
        if os.path.exists(train_val_path):
            with open(train_val_path, encoding="utf-8") as f:
                metrics = json.load(f)
            rows.append(_train_val_metrics_row(exp_name, exp_dir, metrics))
            continue

        print(f"WARNING: experiment '{exp_name}' has neither metrics_{args.split}.json nor "
              f"metrics_val.json; skipping.")

    if not rows:
        print("ERROR: no experiment with metrics available to compare.")
        sys.exit(1)

    table = pd.DataFrame(rows)
    table = table.sort_values("f1_macro", ascending=False, na_position="last").reset_index(drop=True)

    columns = ["experiment", "split", "acc", "bal_acc", "f1_macro",
               "rec_object", "prec_object", "ap_object", "n_object", "epochs"]
    table = table[columns]

    print(table.to_string(index=False))

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nsaved: {args.csv}")


def main():
    p = argparse.ArgumentParser(
        description="Evaluation of the satellite tile classifier (sea/land/cloud/object)."
    )
    p.add_argument("experiment", nargs="?", default=None,
                   help="Path to the experiment directory (e.g.: experiments/smoke)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="Split to evaluate/compare (default: test)")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"],
                   help="Checkpoint to load in evaluation mode (default: best)")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Allow evaluating on CPU when CUDA is unavailable")
    p.add_argument("--compare", action="store_true",
                   help="Comparison mode: scans experiments/*/ instead of evaluating a single one")
    p.add_argument("--csv", default=None,
                   help="Path to save the comparison table (used with --compare)")
    args = p.parse_args()

    if args.compare:
        compare_mode(args)
        return

    if not args.experiment:
        print("ERROR: provide the experiment directory, or use --compare.")
        print("Usage: python3 evaluate.py experiments/<name> [--split test|val] [--checkpoint best|last]")
        print("       python3 evaluate.py --compare [--split test] [--csv comparison.csv]")
        sys.exit(1)

    evaluate_mode(args)


if __name__ == "__main__":
    main()
