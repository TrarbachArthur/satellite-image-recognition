#!/usr/bin/env python3
"""Training script for the satellite tile classifier.

Reads a YAML config (deep-merged over sensible defaults), prepares the data via
dataset.py, trains a timm model and records everything (history.csv, TensorBoard,
checkpoints) in experiments/<name>/. Never hardcodes counts or scene names --
everything comes from the config and from what prepare_data() finds on disk.

Usage:
    python3 train.py --config configs/smoke.yaml
    python3 train.py --config configs/exp1.yaml --allow-cpu
    python3 train.py --config configs/exp1.yaml --overwrite
    python3 train.py --config configs/exp1.yaml --resume
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from torch.utils.tensorboard import SummaryWriter

from dataset import CLASSES, IDX, create_loaders, class_weights, prepare_data

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")

RED = "\033[91m"
COLOR_RESET = "\033[0m"

DEFAULTS = {
    "name": None,
    "seed": 42,
    "data": {
        "scenes": ["sat1", "sat2"],
        "exclude_border": True,
        "fraction": 1.0,
        "max_per_class": None,
        "img_size": 224,
    },
    "split": {
        "method": "random",
        "fractions": [0.70, 0.15, 0.15],
        "block_px": 2048,
    },
    "model": {
        "architecture": "resnet18",
        "pretrained": True,
    },
    "balancing": {
        "method": "none",
        "max_weight": 20.0,
    },
    "loss": {
        "type": "ce",
        "gamma": 2.0,
        "label_smoothing": 0.0,
    },
    "augmentation": "light",
    "training": {
        "epochs": 30,
        "batch": 128,
        "lr": 3.0e-4,
        "weight_decay": 0.05,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "amp": True,
        "num_workers": 2,
        "early_stopping_patience": 7,
        "checkpoint_metric": "f1_macro",
    },
}

_VALID_ENUMS = {
    ("split", "method"): {"random", "spatial"},
    ("balancing", "method"): {"none", "weights", "sampler"},
    ("loss", "type"): {"ce", "focal"},
    ("training", "optimizer"): {"adamw", "sgd"},
    ("training", "scheduler"): {"cosine", "none"},
    ("training", "checkpoint_metric"): {"f1_macro", "balanced_accuracy", "val_loss"},
}
_VALID_AUGMENTATIONS = {"none", "light", "heavy"}


def _deep_merge(base, override):
    """Recursively merges 'override' onto 'base' (nested dicts; other types,
    including lists, are replaced wholesale)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _validate_config(cfg):
    if not cfg.get("name"):
        raise ValueError(
            "invalid config: the 'name' field is required (it defines the "
            "experiments/<name>/ folder)."
        )

    for (section, field), valid in _VALID_ENUMS.items():
        value = cfg[section][field]
        if value not in valid:
            raise ValueError(
                f"invalid config: {section}.{field}={value!r} is not a valid option. "
                f"Use one of: {sorted(valid)}."
            )

    if cfg["augmentation"] not in _VALID_AUGMENTATIONS:
        raise ValueError(
            f"invalid config: augmentation={cfg['augmentation']!r} is not a valid "
            f"option. Use one of: {sorted(_VALID_AUGMENTATIONS)}."
        )

    fractions = cfg["split"]["fractions"]
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-3:
        raise ValueError(
            f"invalid config: split.fractions must have 3 values summing to ~1.0; "
            f"got {fractions!r}."
        )

    if not cfg["model"]["architecture"]:
        raise ValueError("invalid config: model.architecture cannot be empty.")

    # The architecture itself is free-form (any name timm accepts); it is only
    # validated when the model is created, where timm's error is already clear.


def load_config(path):
    """Reads the YAML at 'path', deep-merges it over DEFAULTS and validates it.

    Returns the fully resolved config dict (with all defaults applied) -- the
    same dict later saved to experiments/<name>/config.yaml.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError(
            f"invalid config: {path} does not contain a top-level YAML mapping."
        )

    cfg = _deep_merge(DEFAULTS, user_cfg)
    _validate_config(cfg)
    return cfg


class FocalLoss(nn.Module):
    """Simple focal loss: per-sample CE -> pt=exp(-ce) -> ((1-pt)**gamma*ce).mean().

    'alpha' (per-class weights) is passed as the inner cross-entropy weight,
    exactly like CrossEntropyLoss(weight=...).
    """

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_criterion(cfg, weights):
    loss_type = cfg["loss"]["type"]
    ls = cfg["loss"]["label_smoothing"]
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=ls)
    if loss_type == "focal":
        return FocalLoss(gamma=cfg["loss"]["gamma"], alpha=weights, label_smoothing=ls)
    raise ValueError(f"unknown loss type: {loss_type!r}")  # already validated


def build_optimizer(cfg, parameters):
    t = cfg["training"]
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(parameters, lr=t["lr"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(
            parameters, lr=t["lr"], momentum=0.9, weight_decay=t["weight_decay"]
        )
    raise ValueError(f"unknown optimizer: {t['optimizer']!r}")  # already validated


def build_scheduler(cfg, optimizer):
    t = cfg["training"]
    if t["scheduler"] == "none":
        return None
    if t["scheduler"] != "cosine":
        raise ValueError(f"unknown scheduler: {t['scheduler']!r}")  # already validated

    epochs = t["epochs"]
    warmup = min(2, epochs)

    def lr_lambda(epoch_idx):
        if warmup > 0 and epoch_idx < warmup:
            return (epoch_idx + 1) / warmup
        progress = (epoch_idx - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, amp_enabled):
    model.train()
    loss_sum = 0.0
    total_n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=amp_enabled):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * imgs.size(0)
        total_n += imgs.size(0)

    return loss_sum / total_n if total_n else float("nan")


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled):
    """Evaluates on the given loader. Returns (val_loss, y_true, y_pred, y_prob),
    all concatenated into numpy arrays (y_prob: per-class softmax probabilities)."""
    model.eval()
    loss_sum = 0.0
    total_n = 0
    y_true, y_pred, y_prob = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device, enabled=amp_enabled):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        loss_sum += loss.item() * imgs.size(0)
        total_n += imgs.size(0)

        probs = F.softmax(outputs.float(), dim=1)
        y_true.append(labels.cpu().numpy())
        y_pred.append(probs.argmax(dim=1).cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true) if y_true else np.zeros(0, dtype=int)
    y_pred = np.concatenate(y_pred) if y_pred else np.zeros(0, dtype=int)
    y_prob = np.concatenate(y_prob) if y_prob else np.zeros((0, len(CLASSES)))
    val_loss = loss_sum / total_n if total_n else float("nan")
    return val_loss, y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, present_classes):
    """Classification metrics via sklearn. f1_macro/acc/bal_acc consider all
    CLASSES; per-class recall is reported only for the present ones."""
    all_labels = list(range(len(CLASSES)))
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)

    present_idx = [IDX[c] for c in present_classes]
    recalls = recall_score(
        y_true, y_pred, labels=present_idx, average=None, zero_division=0
    )
    recall_per_class = {c: float(r) for c, r in zip(present_classes, recalls)}

    return {
        "acc": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "recall_per_class": recall_per_class,
    }


def prepare_experiment_dir(name, overwrite, resume):
    exp_dir = os.path.join(EXPERIMENTS_DIR, name)
    history = os.path.join(exp_dir, "history.csv")
    already_exists = os.path.isdir(exp_dir) and os.path.exists(history)

    if resume and not already_exists:
        raise SystemExit(
            f"ERROR: --resume was passed, but there is no history.csv in {exp_dir}. "
            f"There is nothing to resume; remove --resume to start a new experiment."
        )

    if already_exists and not resume:
        if overwrite:
            shutil.rmtree(exp_dir)
        else:
            raise SystemExit(
                f"ERROR: experiment '{name}' already exists in {exp_dir} (history.csv "
                f"present). Pick another name, delete the folder manually, or run "
                f"again with --overwrite to start from scratch."
            )

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "tb"), exist_ok=True)
    return exp_dir


def main():
    p = argparse.ArgumentParser(
        description="Training of the satellite tile classifier (sea/land/cloud/object)."
    )
    p.add_argument("--config", required=True, help="Path to the YAML config")
    p.add_argument(
        "--allow-cpu", action="store_true",
        help="Allow training on CPU when CUDA is unavailable (much slower)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Delete the existing experiment (if any) and start from scratch",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Continue the existing experiment from last.pt",
    )
    args = p.parse_args()

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    name = cfg["name"]
    exp_dir = prepare_experiment_dir(name, args.overwrite, args.resume)
    print(f"experiment '{name}' -> {exp_dir}")

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = "cuda"
    else:
        print(f"{RED}WARNING: CUDA is not available in this environment. Training on CPU is "
              f"MUCH slower.{COLOR_RESET}")
        if not args.allow_cpu:
            print("ERROR: run again with --allow-cpu to force execution on CPU.")
            sys.exit(1)
        device = "cpu"
    print(f"device: {device}")

    try:
        df, summary = prepare_data(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR preparing the data: {e}")
        sys.exit(1)

    print("split summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    df[["file", "label", "scene", "split"]].to_csv(
        os.path.join(exp_dir, "splits.csv"), index=False
    )
    with open(os.path.join(exp_dir, "splits_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(exp_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    val_present_classes = [c for c in CLASSES if summary.get("val", {}).get(c, 0) > 0]

    if cfg["balancing"]["method"] == "sampler" and cfg["augmentation"] == "none":
        print("WARNING: balancing.method='sampler' with augmentation='none' makes the "
              "WeightedRandomSampler repeat exactly the same minority-class tiles "
              "many times per epoch (risk of memorizing/overfitting). "
              "Consider augmentation 'light' or 'heavy'.")

    dl_train, dl_val, dl_test = create_loaders(
        df,
        img_size=cfg["data"]["img_size"],
        augmentation=cfg["augmentation"],
        batch=cfg["training"]["batch"],
        num_workers=cfg["training"]["num_workers"],
        balancing_method=cfg["balancing"]["method"],
        max_weight=cfg["balancing"]["max_weight"],
        seed=seed,
    )
    df_train = df[df["split"] == "train"]

    try:
        model = timm.create_model(
            cfg["model"]["architecture"],
            pretrained=cfg["model"]["pretrained"],
            num_classes=len(CLASSES),
        )
    except Exception as e:
        if cfg["model"]["pretrained"]:
            print(
                f"ERROR: failed to create model '{cfg['model']['architecture']}' with "
                f"pretrained weights (probably a network failure while downloading them): {e}\n"
                f"Check your internet connection or set model.pretrained: false."
            )
        else:
            print(
                f"ERROR: failed to create model '{cfg['model']['architecture']}': {e}\n"
                f"Check that the architecture name is valid for timm."
            )
        sys.exit(1)
    model = model.to(device)

    weights = None
    if cfg["balancing"]["method"] == "weights":
        weights = class_weights(df_train, cfg["balancing"]["max_weight"]).to(device)
    criterion = build_criterion(cfg, weights)

    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer)

    amp_enabled = bool(cfg["training"]["amp"]) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_epochs = cfg["training"]["epochs"]
    metric_key = cfg["training"]["checkpoint_metric"]
    higher_is_better = metric_key != "val_loss"

    best_path = os.path.join(exp_dir, "best.pt")
    last_path = os.path.join(exp_dir, "last.pt")
    history_path = os.path.join(exp_dir, "history.csv")

    best_value = -float("inf") if higher_is_better else float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1

    if args.resume and os.path.exists(last_path):
        print(f"resuming from {last_path}")
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_value = best_ckpt["metrics"][metric_key]
            best_epoch = best_ckpt["epoch"]
        print(f"resuming from epoch {start_epoch}")

    write_header = not os.path.exists(history_path)
    history_fields = (
        ["epoch", "train_loss", "val_loss", "acc", "balanced_accuracy", "f1_macro"]
        + [f"recall_{c}" for c in val_present_classes]
        + ["lr", "time_s"]
    )

    tb_writer = SummaryWriter(log_dir=os.path.join(exp_dir, "tb"))
    patience = cfg["training"]["early_stopping_patience"]

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, dl_train, optimizer, criterion, scaler, device, amp_enabled)
        val_loss, y_true, y_pred, _y_prob = evaluate(model, dl_val, criterion, device, amp_enabled)
        metrics = compute_metrics(y_true, y_pred, val_present_classes)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        parts = [
            f"epoch {epoch}/{total_epochs}",
            f"train_loss {train_loss:.3f}",
            f"val_loss {val_loss:.3f}",
            f"acc {metrics['acc']:.3f}",
            f"bal_acc {metrics['balanced_accuracy']:.3f}",
            f"f1 {metrics['f1_macro']:.3f}",
        ]
        for c in val_present_classes:
            parts.append(f"rec_{c} {metrics['recall_per_class'][c]:.3f}")
        parts.append(f"({elapsed:.1f}s)")
        print("  ".join(parts))

        tb_writer.add_scalar("loss/train", train_loss, epoch)
        tb_writer.add_scalar("loss/val", val_loss, epoch)
        tb_writer.add_scalar("val/acc", metrics["acc"], epoch)
        tb_writer.add_scalar("val/balanced_acc", metrics["balanced_accuracy"], epoch)
        tb_writer.add_scalar("val/f1_macro", metrics["f1_macro"], epoch)
        for c in val_present_classes:
            tb_writer.add_scalar(f"val/recall_{c}", metrics["recall_per_class"][c], epoch)
        tb_writer.add_scalar("lr", current_lr, epoch)
        tb_writer.flush()

        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "acc": metrics["acc"], "balanced_accuracy": metrics["balanced_accuracy"],
            "f1_macro": metrics["f1_macro"], "lr": current_lr, "time_s": elapsed,
        }
        for c in val_present_classes:
            row[f"recall_{c}"] = metrics["recall_per_class"][c]
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=history_fields)
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)

        full_metrics = dict(row)
        current_value = full_metrics[metric_key]
        improved = (current_value > best_value) if higher_is_better else (current_value < best_value)

        if improved:
            best_value = current_value
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": full_metrics,
                    "config": cfg,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "metrics": full_metrics,
                "config": cfg,
            },
            last_path,
        )

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"early stopping: no improvement in {metric_key} for {patience} epochs "
                  f"(best epoch: {best_epoch}, value: {best_value:.4f})")
            break

    tb_writer.close()

    if not os.path.exists(best_path):
        print("WARNING: no 'best.pt' checkpoint was saved (no epoch ran); "
              "skipping the final evaluation.")
        return

    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    final_val_loss, y_true_final, y_pred_final, _ = evaluate(model, dl_val, criterion, device, amp_enabled)
    final_metrics = compute_metrics(y_true_final, y_pred_final, val_present_classes)
    final_metrics["val_loss"] = final_val_loss
    final_metrics["best_epoch"] = best_ckpt["epoch"]

    with open(os.path.join(exp_dir, "metrics_val.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    print("final evaluation (best checkpoint) on val:")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
