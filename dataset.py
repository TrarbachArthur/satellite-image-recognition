#!/usr/bin/env python3
"""Data layer of the satellite tile classification training pipeline.

Reads the label CSVs produced by label.py, filters/samples/splits the data into
train/val/test and builds the PyTorch Dataset/DataLoader objects used by train.py.

The amount of labeled data keeps changing (more scenes, more tiles), so nothing
here depends on fixed counts, on which classes are present or on hardcoded scene
names -- everything is derived from the CSVs at runtime. The only fixed constants
are the model's target classes (CLASSES) and the order of the splits.

Usage (quick demo, trains nothing):
    python3 dataset.py sat1 sat2
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 'uncertain' never reaches training: filter_labels() always drops it. A class
# missing from the loaded data (e.g. a scene without 'object') is handled
# normally by the rest of the module (count 0).
CLASSES = ["sea", "land", "cloud", "object"]
IDX = {c: i for i, c in enumerate(CLASSES)}

SPLITS = ["train", "val", "test"]

# Extracts (y, x) from the file name, e.g. sat1_y000768_x009472.png
_POSITION_REGEX = re.compile(r"_y(\d+)_x(\d+)\.")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def load_labels(scenes, data_dir=DATA_DIR):
    """Reads data/<scene>_tiles/<scene>_labels.csv (exact name) for each scene and
    concatenates everything into a single DataFrame.

    Output columns: file, label, border (int), scene, x (int), y (int), path (abs).

    Only the exact file name "<scene>_labels.csv" is read (no glob), so stray
    files such as "<scene>_labels_bak.csv" are ignored.
    """
    parts = []
    for scene in scenes:
        scene_dir = os.path.join(data_dir, f"{scene}_tiles")
        csv_path = os.path.join(scene_dir, f"{scene}_labels.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Label CSV not found for scene '{scene}': {csv_path}. "
                f"Label the scene with 'python3 label.py {scene}' before training."
            )

        df_scene = pd.read_csv(csv_path, dtype={"file": str, "label": str})
        expected_columns = {"file", "label", "border", "timestamp"}
        missing = expected_columns - set(df_scene.columns)
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing the expected columns: {sorted(missing)}. "
                f"Columns found: {list(df_scene.columns)}."
            )

        df_scene["border"] = df_scene["border"].fillna(0).astype(int)
        df_scene["scene"] = scene

        xs, ys = [], []
        for file in df_scene["file"]:
            m = _POSITION_REGEX.search(str(file))
            if not m:
                raise ValueError(
                    f"Could not extract the (y, x) position from file name "
                    f"'{file}' in scene '{scene}'. Expected the pattern "
                    f"'..._y<Y>_x<X>.<ext>' (e.g. sat1_y000768_x009472.png)."
                )
            ys.append(int(m.group(1)))
            xs.append(int(m.group(2)))
        df_scene["y"] = ys
        df_scene["x"] = xs
        df_scene["path"] = df_scene["file"].apply(
            lambda f: os.path.abspath(os.path.join(scene_dir, f))
        )

        parts.append(df_scene[["file", "label", "border", "scene", "x", "y", "path"]])

    if not parts:
        raise ValueError("load_labels needs at least one scene.")

    df = pd.concat(parts, ignore_index=True)

    duplicates = df.loc[df["file"].duplicated(keep=False), "file"]
    if len(duplicates) > 0:
        examples = sorted(duplicates.unique())[:5]
        raise ValueError(
            f"Duplicate file names across scenes: {examples}"
            f"{' (and more)' if len(duplicates.unique()) > 5 else ''}. "
            f"Check whether the scenes overlap or whether some tiles share a name."
        )

    return df


def filter_labels(df, exclude_border):
    """Drops 'uncertain' and any label outside CLASSES; optionally drops tiles
    marked border=1 as well. Prints the counts before/after."""
    before = df["label"].value_counts().to_dict()
    print(f"filter_labels: before -> total={len(df)} counts={before}")

    df2 = df[df["label"].isin(CLASSES)].copy()
    if exclude_border:
        df2 = df2[df2["border"] != 1].copy()

    after = df2["label"].value_counts().to_dict()
    print(f"filter_labels: after -> total={len(df2)} counts={after} "
          f"(exclude_border={exclude_border})")

    return df2.reset_index(drop=True)


def limit(df, fraction, max_per_class, seed):
    """Class-stratified sampling, reproducible via seed.

    fraction: keeps ~fraction of each present class. Floor of 3 examples per
        class (or n, if the class has fewer) -- the minimum for split_data() to
        place 1 example in each of the 3 splits.
    max_per_class: cap applied after the fraction (None = no cap). Classes whose
        result is already within the cap pass through untouched.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for cls, group in df.groupby("label"):
        n = len(group)
        n_keep = min(n, max(3, int(round(fraction * n))))
        idx = group.index.to_numpy()
        chosen = rng.choice(idx, size=n_keep, replace=False)
        if max_per_class is not None and len(chosen) > max_per_class:
            chosen = rng.choice(chosen, size=max_per_class, replace=False)
        print(f"limit: {cls}: {n} -> {len(chosen)}")
        parts.append(df.loc[chosen])

    if not parts:
        return df.iloc[0:0].copy()

    return pd.concat(parts).sort_index().reset_index(drop=True)


def _split_random(df, fractions, rng):
    split = pd.Series(index=df.index, dtype=object)
    for _, group in df.groupby("label"):
        idx = group.index.to_numpy()
        n = len(idx)
        idx_perm = idx[rng.permutation(n)]
        c1 = min(n, int(round(fractions[0] * n)))
        c2 = min(n, max(c1, int(round((fractions[0] + fractions[1]) * n))))
        if n >= 3:
            # at least 1 example of the class in each split (with small n,
            # rounding would zero out val and/or test)
            c1 = min(max(c1, 1), n - 2)
            c2 = min(max(c2, c1 + 1), n - 1)
        split.loc[idx_perm[:c1]] = "train"
        split.loc[idx_perm[c1:c2]] = "val"
        split.loc[idx_perm[c2:]] = "test"
    return split


def _split_spatial(df, fractions, rng, block_px):
    block_ids = list(zip(df["scene"], df["y"] // block_px, df["x"] // block_px))
    df_tmp = df.copy()
    df_tmp["_block"] = block_ids

    class_counts = df_tmp["label"].value_counts()
    present = [c for c in CLASSES if c in class_counts.index]
    if not present:
        raise ValueError("split_data (spatial): no valid class present in the DataFrame.")
    rare_class = class_counts.loc[present].idxmin()

    block_info = {}
    for bid, sub in df_tmp.groupby("_block"):
        block_info[bid] = {
            "indices": sub.index.to_numpy(),
            "total": len(sub),
            "rare": int((sub["label"] == rare_class).sum()),
        }

    blocks_with_rare = [b for b, info in block_info.items() if info["rare"] > 0]
    other_blocks = [b for b, info in block_info.items() if info["rare"] == 0]

    total_rare = sum(block_info[b]["rare"] for b in blocks_with_rare)
    total_tiles = len(df_tmp)

    target_rare = {s: fractions[i] * total_rare for i, s in enumerate(SPLITS)}
    target_total = {s: fractions[i] * total_tiles for i, s in enumerate(SPLITS)}
    current_rare = {s: 0 for s in SPLITS}
    current_total = {s: 0 for s in SPLITS}

    block_split = {}

    # Blocks containing the globally rarest class are assigned first so that
    # each split ends up with a share of it close to 'fractions'.
    order = rng.permutation(len(blocks_with_rare))
    for i in order:
        b = blocks_with_rare[i]
        info = block_info[b]
        chosen = max(SPLITS, key=lambda s: target_rare[s] - current_rare[s])
        block_split[b] = chosen
        current_rare[chosen] += info["rare"]
        current_total[chosen] += info["total"]

    # Remaining blocks, shuffled and assigned by tile deficit.
    order2 = rng.permutation(len(other_blocks))
    for i in order2:
        b = other_blocks[i]
        info = block_info[b]
        chosen = max(SPLITS, key=lambda s: target_total[s] - current_total[s])
        block_split[b] = chosen
        current_total[chosen] += info["total"]

    split = df_tmp["_block"].map(block_split)
    split.index = df_tmp.index
    return split


def split_data(df, method, fractions, seed, block_px=2048):
    """Adds the 'split' column (train/val/test) to the DataFrame.

    method 'random': stratified by class (does not prevent spatial leakage).
    method 'spatial': groups tiles into blocks (scene, y//block_px, x//block_px)
        and assigns whole blocks to a single split, so the same block never
        appears in two splits (anti spatial leakage).

    At the end, checks that every class present in df has at least 1 example in
    each split -- otherwise raises ValueError.
    """
    if len(df) == 0:
        raise ValueError("split_data received an empty DataFrame; nothing to split.")
    if len(fractions) != 3:
        raise ValueError(f"fractions must have 3 values (train, val, test); got: {fractions}")

    rng = np.random.default_rng(seed)

    if method == "random":
        split = _split_random(df, fractions, rng)
    elif method == "spatial":
        split = _split_spatial(df, fractions, rng, block_px)
    else:
        raise ValueError(f"unknown split method: {method!r}. Use 'random' or 'spatial'.")

    df = df.copy()
    df["split"] = split

    present = [c for c in CLASSES if (df["label"] == c).any()]
    missing = [
        (c, s) for c in present for s in SPLITS
        if not (((df["label"] == c) & (df["split"] == s)).any())
    ]
    if missing:
        details = ", ".join(f"{c}/{s}" for c, s in missing)
        raise ValueError(
            f"The split resulted in 0 examples for: {details}. "
            f"Increase 'fraction' in limit(), label more tiles of these classes, "
            f"or change the split method/parameters."
        )

    return df


def split_summary(df):
    """Per-class count in each split, saved to splits_summary.json for auditing."""
    present = [c for c in CLASSES if (df["label"] == c).any()]
    summary = {}
    for s in SPLITS:
        sub = df[df["split"] == s]
        summary[s] = {c: int((sub["label"] == c).sum()) for c in present}
    summary["total"] = {c: int((df["label"] == c).sum()) for c in present}
    return summary


class TilesDataset(torch.utils.data.Dataset):
    """Dataset that opens each tile from disk on demand (never preloads
    everything into RAM). Keeps only lists of paths/labels, not the DataFrame."""

    def __init__(self, df, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, IDX[self.labels[i]]


class _RandomRotation90:
    """Rotates the image by a random multiple of 90 degrees (0/90/180/270).

    Nadir satellite tiles are rotation invariant, so this is a safe augmentation
    (unlike an arbitrary rotation). Uses torch's global RNG; for full run-to-run
    reproducibility the caller must seed it with torch.manual_seed(...) before
    training.
    """

    def __call__(self, img):
        angle = int(torch.randint(0, 4, (1,)).item()) * 90
        if angle == 0:
            return img
        return T.functional.rotate(img, angle)


def create_transforms(level, img_size, train):
    """Builds the torchvision transform pipeline.

    - eval (train=False), or train=True with level='none':
        Resize + ToTensor + Normalize (ImageNet).
    - train=True, level='light':
        horizontal/vertical flips + random 90-degree rotation + resize + normalize.
    - train=True, level='heavy':
        light + RandomResizedCrop(scale=0.6-1.0) + ColorJitter + occasional
        GaussianBlur (p~0.2) + RandomErasing (p=0.1, after ToTensor).
    """
    normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if (not train) or level == "none":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            normalize,
        ])

    if level not in ("light", "heavy"):
        raise ValueError(f"unknown augmentation level: {level!r}. Use 'none', 'light' or 'heavy'.")

    steps = [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        _RandomRotation90(),
    ]

    if level == "light":
        steps.append(T.Resize((img_size, img_size)))
    else:
        steps.append(T.RandomResizedCrop(img_size, scale=(0.6, 1.0)))
        steps.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1))
        steps.append(T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2))

    steps.append(T.ToTensor())
    steps.append(normalize)

    if level == "heavy":
        steps.append(T.RandomErasing(p=0.1))

    return T.Compose(steps)


def class_weights(df_train, max_weight):
    """weight_c = n_total / (n_present_classes * n_c), clipped at max_weight.

    Classes absent from df_train get weight 0. Returns a float tensor in
    CLASSES order.
    """
    counts = df_train["label"].value_counts()
    n_total = len(df_train)
    present = [c for c in CLASSES if counts.get(c, 0) > 0]

    weights = torch.zeros(len(CLASSES), dtype=torch.float32)
    if not present or n_total == 0:
        return weights

    n_present = len(present)
    for c in present:
        n_c = int(counts[c])
        weight = n_total / (n_present * n_c)
        if max_weight is not None:
            weight = min(weight, max_weight)
        weights[IDX[c]] = weight

    return weights


def create_loaders(df, img_size, augmentation, batch, num_workers, balancing_method, max_weight, seed):
    """Builds the three DataLoaders (train, val, test) from a DataFrame that
    already has the 'split' column.

    balancing_method == 'sampler': uses a WeightedRandomSampler on train, with
        per-sample weight = class weight (clipped at max_weight). Otherwise the
        train loader just shuffles.
    """
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    train_transform = create_transforms(augmentation, img_size, train=True)
    eval_transform = create_transforms("none", img_size, train=False)

    ds_train = TilesDataset(df_train, train_transform)
    ds_val = TilesDataset(df_val, eval_transform)
    ds_test = TilesDataset(df_test, eval_transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = None
    if balancing_method == "sampler":
        class_w = class_weights(df_train, max_weight)
        class_indices = torch.tensor([IDX[r] for r in df_train["label"]], dtype=torch.long)
        sample_weights = class_w[class_indices]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(df_train),
            replacement=True,
            generator=generator,
        )

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        # drop_last only when there is more than one batch; otherwise a smoke
        # test with little data (len(train) < batch) would yield 0 batches per epoch.
        drop_last=(len(ds_train) >= batch),
        generator=generator if sampler is None else None,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    return dl_train, dl_val, dl_test


def prepare_data(cfg):
    """Chains load_labels -> filter_labels -> limit -> split_data -> split_summary.

    Expects cfg in the format:
        cfg["seed"]: int
        cfg["data"]: {
            "scenes": [...],
            "data_dir": optional (default DATA_DIR),
            "exclude_border": bool (default False),
            "fraction": float (default 1.0),
            "max_per_class": int or None (default None),
        }
        cfg["split"]: {
            "method": "random" or "spatial" (default "random"),
            "fractions": [train, val, test] (default [0.7, 0.15, 0.15]),
            "block_px": int (default 2048, only used by the 'spatial' method),
        }

    Returns (df_with_split, summary).
    """
    cfg_data = cfg["data"]
    cfg_split = cfg["split"]
    seed = cfg["seed"]

    data_dir = cfg_data.get("data_dir", DATA_DIR)
    df = load_labels(cfg_data["scenes"], data_dir=data_dir)
    df = filter_labels(df, exclude_border=cfg_data.get("exclude_border", False))
    df = limit(
        df,
        fraction=cfg_data.get("fraction", 1.0),
        max_per_class=cfg_data.get("max_per_class", None),
        seed=seed,
    )
    df = split_data(
        df,
        method=cfg_split.get("method", "random"),
        fractions=cfg_split.get("fractions", [0.7, 0.15, 0.15]),
        seed=seed,
        block_px=cfg_split.get("block_px", 2048),
    )
    summary = split_summary(df)
    return df, summary


if __name__ == "__main__":
    scenes = sys.argv[1:]
    if not scenes:
        print("Usage: python3 dataset.py <scene> [<scene> ...]")
        sys.exit(1)

    try:
        df = load_labels(scenes)
        df = filter_labels(df, exclude_border=False)
        df = split_data(df, method="random", fractions=[0.7, 0.15, 0.15], seed=42)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\nLoaded {len(df)} labeled tiles from {len(scenes)} scene(s): {scenes}")
    print(json.dumps(split_summary(df), indent=2, ensure_ascii=False))
