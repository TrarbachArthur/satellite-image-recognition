# Satellite Image Recognition

Converts satellite images in "pan format" (separate bands) into high-resolution RGB tiles
and lets you label them, producing a dataset to train a recognition model.

> **Full guide:** this README is the quick overview. For the detailed walkthrough of every
> process, the expected output of each step and the reference for all settings
> (CLI and YAML), see **[USAGE.md](USAGE.md)**.

## What the project does

1. **Fusion + tiling** (`generate_tiles.py`): fuses the color bands with the panchromatic band
   (pansharpening) at full resolution and cuts the result into a grid of PNG tiles.
2. **Labeling** (`label.py`): an interface to quickly classify each tile and save the labels
   to CSV.
3. **Visualization** (`visualize.py`): builds images showing what was labeled — one image per
   class and a scene overview with the tiles outlined by color.
4. **Training** (`train.py` + `evaluate.py`): trains classifiers (CNNs and Visual
   Transformers via timm) on the labeled tiles, with experiments driven by YAML files, and
   compares the results.
5. **Pre-labeling** (`infer.py`): uses the trained models to automatically label new scenes —
   you only review the predictions (with `fix_labels.py`) instead of labeling from scratch.

There is also `generate_rgb.py`, which generates a single RGB image of the whole scene
(overview) instead of tiles.

## Installation

```bash
pip install -r requirements.txt
```

## Step by step

### 1. Put the images in `data/`

The `data/` folder already exists in the repository but is **empty** — put the bands of each
scene here. Each scene has a **prefix** and five TIFF files, one per band:

```
data/
  <prefix>_r.tif     # red
  <prefix>_g.tif     # green
  <prefix>_b.tif     # blue
  <prefix>_p.tif     # panchromatic (high resolution, grayscale)
  <prefix>_nir.tif   # near infrared
```

E.g. for scene `sat2`, the files are `sat2_r.tif`, `sat2_g.tif`, ... `sat2_nir.tif`.

### 2. Generate the tiles

```bash
python3 generate_tiles.py <prefix>
```

Creates the `data/<prefix>_tiles/` folder with 1024×1024 PNG tiles and an index
`<prefix>_tiles.csv` (`file, x, y, width, height`).

Useful options:

```bash
python3 generate_tiles.py <prefix> --tile 640      # tile size
python3 generate_tiles.py <prefix> --overlap 128   # overlap between tiles
python3 generate_tiles.py <prefix> --format jpg    # save as JPG
python3 generate_tiles.py <prefix> --skip-empty    # drop uniform tiles (sea/black)
```

### 3. Label the tiles

```bash
python3 label.py <prefix>
```

Opens the tiles in order (left→right, top→bottom). Label with the keyboard or the buttons;
labeling can be interrupted and resumed at any time.

| Key | Action |
|---|---|
| `1`–`5` | sea / land / cloud / object / uncertain → labels **and advances** |
| `B` | toggles the **border** flag (independent of the class) |
| `←` / `Backspace` | go back to fix |
| `→` | advance without changing |
| `Z` | 1:1 zoom |
| `Esc` / `q` | quit (everything saved) |

Labels are saved to `data/<prefix>_tiles/<prefix>_labels.csv`
(`file, label, border, timestamp`).

### 4. Visualize the labels

```bash
python3 visualize.py <prefix>               # scale 0.1 (default)
python3 visualize.py <prefix> --scale 0.2   # more detail (larger files)
```

Generates in `data/<prefix>_viz/`:

- `<prefix>_overview.png` — the downscaled scene with each tile outlined by a square in the
  color of its class; tiles marked as border get a magenta outline; includes a legend.
- `<prefix>_composite_<class>.png` — one image per class (sea/land/cloud/object/uncertain)
  with the real image of that class's tiles at the right position and the rest black.
- `<prefix>_composite_border.png` — the same, only with the tiles marked as border.

### 5. Fix wrong labels

If the visualization reveals labeling mistakes, use the correction app instead of editing the
CSV by hand:

```bash
python3 fix_labels.py <prefix>                      # scale 0.25 (default)
python3 fix_labels.py <prefix> --scale 0.2          # lighter, if it gets heavy
python3 fix_labels.py <prefix> --classes uncertain  # show/select only these classes
```

Shows the scene with **zoom and pan** and a per-class colored overlay over the real image.
Zoom in to check details, click a tile to select it, or **drag** a rectangle (or
**Shift+click**) to select several; then apply the right label. Changes are written to the
same `<prefix>_labels.csv`. (On the first run it builds and caches the background image — it
may take a few minutes; later runs are fast.)

| Control | Action |
|---|---|
| mouse wheel | zoom in/out at the cursor |
| right button (drag) / arrows | move the image (pan) |
| `0` | fit the whole scene to the window |
| click / drag / Shift+click | select one / several tiles |
| `1`–`5` | applies sea / land / cloud / object / uncertain to the selected tiles |
| `B` / `N` | sets / clears the **border** flag of the selected tiles |
| `O` | toggles the class overlay |
| `Esc` / `q` | clear the selection / quit (everything already saved) |

### 6. Train the model

Training uses PyTorch with a GPU (install the CUDA wheel before the other dependencies):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Each experiment is a YAML file in `configs/` controlling the architecture (any timm model,
incl. ViT), pretraining, augmentation, balancing, data fraction, split, etc.
Only the `name` field is required — the rest has defaults (see `DEFAULTS` in `train.py`).

```bash
python3 train.py --config configs/smoke.yaml      # validates the pipeline in ~1 min (2% of the data)
python3 train.py --config configs/resnet18_base.yaml
python3 train.py --config configs/vit_small_pretrained.yaml
tensorboard --logdir experiments                  # curves of all experiments
```

Outputs in `experiments/<name>/`: frozen config, `splits.csv` (the exact split used),
`history.csv`, checkpoints (`best.pt`/`last.pt`) and TensorBoard logs. Use
`--resume` to continue an interrupted run and `--overwrite` to start over.

Ready-made hypotheses in `configs/` (each differs from the baseline by 1–2 lines):

| Config | Hypothesis |
|---|---|
| `resnet18_base` | baseline with no imbalance handling |
| `resnet18_weights` / `resnet18_sampler` / `resnet18_focal` | 3 ways of handling imbalance |
| `resnet18_with_border` | does keeping border tiles change anything? |
| `resnet18_spatial_split` | how much does the random split inflate the metrics (spatial leakage)? |
| `vit_small_pretrained` / `convnext_tiny` / `efficientnet_b0` | architecture comparison |

### 7. Evaluate and compare

```bash
python3 evaluate.py experiments/resnet18_base        # evaluates on test (metrics, matrix, predictions)
python3 evaluate.py --compare                        # aggregated table of all experiments
```

**Note on the 'object' class:** with very few labeled examples, the metrics for this class
have high variance (the `n_object` column of the comparison shows the support). Until there
is more labeling, treat 'object' results as qualitative; the per-tile probabilities saved in
`predictions_*.csv` help mine new candidates to label.

### 8. Pre-label new scenes with the models

```bash
python3 infer.py sat3 --experiments experiments/vit_small_pretrained experiments/resnet18_sampler --uncertain-threshold 0.7
python3 fix_labels.py sat3   # review the predictions (uncertain ones show up in yellow)
```

Automatically generates the new scene's `labels.csv` (optional ensemble of several models;
low-confidence tiles can become 'uncertain' for review). Never overwrites existing labeling
without `--overwrite`. Details in [USAGE.md](USAGE.md).

## Output

Each scene is processed independently:

```
data/<prefix>_tiles/
  <prefix>_yYYYYYY_xXXXXXX.png   # tiles (the name holds the position in the original scene)
  <prefix>_tiles.csv             # tile index
  <prefix>_labels.csv            # labels (created by the labeling step)
```

## Extra utility

```bash
python3 generate_rgb.py <prefix> --scale 0.1   # a single downscaled RGB image of the scene
```
