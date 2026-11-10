# Complete usage guide

Reference for every process in the pipeline: how to run it, what to expect from each step and
all available settings (command line and YAML).

General flow:

```
.tif bands → [1] generate_tiles.py → PNG tiles → [2] label.py → labels.csv
                                                      ↓
              [5] train.py ← configs/*.yaml   [3] visualize.py (check)
                     ↓                        [4] fix_labels.py (fix errors)
              [6] evaluate.py → metrics / comparison
                     ↓
              [7] infer.py → pre-labels new scenes with the trained models
                             (review with fix_labels.py → more data → new training)
```

Convention: each satellite scene has a **prefix** (e.g. `sat1`) and lives in `data/`. All
scripts accept the prefix as a positional argument; without it, they ask interactively.

---

## 1. `generate_tiles.py` — bands → RGB tiles

Fuses the color bands (R, G, B) with the panchromatic band (Brovey algorithm) at full
resolution, by streaming (never loads the whole scene into RAM), and cuts the result into a
grid of tiles.

**Expected input:** `data/<prefix>_r.tif`, `_g.tif`, `_b.tif`, `_p.tif`, `_nir.tif`
(`nir` is not used in the RGB, but is assumed present in the naming pattern).

```bash
python3 generate_tiles.py <prefix> [options]
```

| Option | Default | Effect |
|---|---|---|
| `--tile N` | 1024 | Tile side in pixels. For classification at 2 m/px GSD, 256 is a good value (see the discussion in the project history). |
| `--scale F` | 1.0 | Output resolution scale (1.0 = panchromatic resolution). |
| `--overlap N` | 0 | Overlap between neighboring tiles, in pixels. Useful for detection; for classification, leave 0 (avoids nearly duplicated tiles and train/val leakage). |
| `--format png\|jpg` | png | Tile format. JPG gives smaller files with slight loss. |
| `--output-dir DIR` | `data/<prefix>_tiles` | Output folder. |
| `--skip-empty` | off | Drops nearly uniform tiles (homogeneous open sea, black frame). Greatly reduces the volume. |

**Expected result:** `data/<prefix>_tiles/` with the PNGs `<prefix>_yYYYYYY_xXXXXXX.png`
(the name encodes the tile's position in the scene) and the index `<prefix>_tiles.csv`
(`file,x,y,width,height`). For a ~56k×58k px scene with 1024 tiles: ~3200 tiles in ~5 min,
constant RAM ~3.5 GB. Contrast is normalized globally (2nd–98th percentiles estimated on a
sample), so all tiles share the same color scale.

### Related utility: `generate_rgb.py`

The same Brovey fusion, but generates **a single image** of the scene instead of tiles —
useful for a quick overview before deciding the tiling parameters.

```bash
python3 generate_rgb.py <prefix> --scale 0.1   # single downscaled PNG (10% of the resolution)
```

| Option | Default | Effect |
|---|---|---|
| `--scale F` | 1.0 | Output scale. Note: on large scenes, 1.0 is infeasible (gigapixel image); use ≤0.25. |
| `--output PATH` | `data/<prefix>_rgb.png` | Output file. |

**Expected result:** an 8-bit RGB PNG of the whole scene at the requested scale.

---

## 2. `label.py` — quick labeling

Tkinter interface that shows the tiles in order (left→right, top→bottom) and labels with a
single keypress. Interruptible: you can close and reopen and it continues from the first
unlabeled tile.

```bash
python3 label.py <prefix>
python3 label.py --tiles data/other_folder_tiles    # arbitrary folder
python3 label.py <prefix> --classes uncertain       # navigates ONLY the tiles with these labels
```

`--classes` (1+ values among sea/land/cloud/object/uncertain) restricts navigation to the
tiles whose current label is in the filter — ideal for reviewing a specific class in sequence
(e.g. the 'uncertain' ones from pre-labeling). It starts from the first tile of the filter;
the progress and the window top show the active filter. The CSV stays complete: relabeling a
tile out of the filter does not remove the labels of the other classes.

| Control | Action |
|---|---|
| `1` `2` `3` `4` `5` | sea / land / cloud / object / uncertain → **saves and advances** |
| `B` | toggles the current tile's **border** flag (independent of the class; does not advance) |
| `←` / `Backspace` | back to the previous tile (to fix; the new key overwrites) |
| `→` | advance without changing |
| `Z` | 1:1 zoom of the tile (scrollable window); `Esc` closes it |
| `Esc` / `q` | quit (everything is already saved) |

**Expected result:** `data/<prefix>_tiles/<prefix>_labels.csv`
(`file,label,border,timestamp`), one row per labeled tile, rewritten atomically on every
action (safe to stop at any time). The `uncertain` label marks undecided tiles — they stay
out of training. The `border` flag identifies tiles with scene-edge noise, which can be
filtered out later.

---

## 3. `visualize.py` — check the labeling

Generates audit images from the labels.

```bash
python3 visualize.py <prefix> [--scale 0.1] [--tiles DIR] [--output-dir DIR]
```

| Option | Default | Effect |
|---|---|---|
| `--scale F` | 0.1 | Scale of the generated images (0.1 → a 56k scene becomes ~5.6k px). |
| `--tiles DIR` | `data/<prefix>_tiles` | Input tiles/labels folder. |
| `--output-dir DIR` | `data/<prefix>_viz` | Output folder. |

**Expected result**, in `data/<prefix>_viz/` (~2 min for 34k tiles, RAM <1 GB):
- `<prefix>_overview.png` — real scene with each tile outlined by the class color
  (sea=blue, land=green, cloud=white, object=red with a thick stroke, uncertain=yellow)
  + magenta outline on border tiles + legend.
- `<prefix>_composite_<class>.png` — one per class: that class's tiles at their real
  position, the rest black. Useful to see "where the model will find each thing".
- `<prefix>_composite_border.png` — only the tiles marked as border.

---

## 4. `fix_labels.py` — fix labels visually

Shows the whole scene with zoom/pan and a translucent overlay per class; mistakes jump out
(e.g. a green square in the middle of the sea). Select tiles and reassign the class without
touching the CSV.

```bash
python3 fix_labels.py <prefix> [--scale 0.25] [--tiles DIR]
```

| Option | Default | Effect |
|---|---|---|
| `--scale F` | 0.25 | Resolution of the background image. On the 1st run it is built (opens all tiles, minutes) and **cached** in `data/<prefix>_viz/<prefix>_base_e<scale>.png`; later runs open instantly. |
| `--tiles DIR` | `data/<prefix>_tiles` | Tiles/labels folder. |
| `--classes C [C ...]` | (all) | Class filter: only tiles with these labels get the overlay **and** only they are selectable — drag over a whole region and only the filter's tiles receive the new label. E.g.: `--classes uncertain` to review only the uncertain ones from pre-labeling. |

| Control | Action |
|---|---|
| mouse wheel | zoom at the cursor (up to 8× over the base) |
| right button dragging / arrows | pan |
| `0` | fit the whole scene to the window |
| click | selects 1 tile |
| drag a rectangle | selects all tiles in the region |
| Shift+click / Shift+drag | adds to the selection |
| `1`–`5` | applies sea/land/cloud/object/uncertain to **all selected** (saves immediately) |
| `B` / `N` | sets / clears the border flag of the selected tiles |
| `O` | toggles the colored overlay |
| `Esc` / `q` | clear selection / quit |

**Expected result:** the same `<prefix>_labels.csv` updated (one atomic rewrite per
correction batch). The top bar shows the clicked tile's label and the total selected.

---

## 5. `train.py` — train an experiment

Trains a 4-class classifier (sea/land/cloud/object) with PyTorch + timm, driven by a YAML.
A GPU is strongly recommended.

```bash
python3 train.py --config configs/<exp>.yaml [flags]
```

| Flag | Effect |
|---|---|
| `--config PATH` | (required) Experiment YAML. |
| `--allow-cpu` | Without CUDA the script aborts by default; this flag forces CPU training (slow). |
| `--overwrite` | Deletes the existing `experiments/<name>/` and starts over. Without it, running an experiment that already exists is an error (protection against accidental loss). |
| `--resume` | Continues an interrupted run from `last.pt` (model, optimizer, epoch). |

### All YAML keys (with defaults)

Only `name` is required; the rest inherits the defaults below. Each experiment must have a
unique `name` — it becomes the `experiments/<name>/` folder.

```yaml
name: my_experiment       # (required) experiment name/folder
seed: 42                  # global seed (data, split, torch) — same seed = same split

data:
  scenes: [sat1, sat2]    # scene prefixes; reads data/<p>_tiles/<p>_labels.csv for each
  exclude_border: true    # true: drops tiles with border=1 | false: keeps them (testable hypothesis)
  fraction: 1.0           # stratified fraction of the data (0.02 = ~2% of each class;
                          #   floor of 3 examples/class). Use a small value for smoke tests.
  max_per_class: null     # cap of examples per class (e.g. 4000). Never reduces classes
                          #   that already have fewer than the cap. null = no cap.
  img_size: 224           # side of the model's input image. 224 = the regime of the
                          #   pretrained weights (recommended); 256 = native tile
                          #   resolution (hypothesis: does losing 12% of resolution matter?)

split:
  method: random          # random: stratified by class (optimistic baseline)
                          # spatial: whole blocks of tiles go to a single split —
                          #   eliminates leakage between nearly identical neighboring tiles
  fractions: [0.70, 0.15, 0.15]   # train / val / test (sum ~1.0)
  block_px: 2048          # only for the spatial method: block side in scene px
                          #   (2048 = blocks of 8x8 tiles of 256)

model:
  architecture: resnet18  # any timm name. Tested on this GPU (8GB):
                          #   resnet18, resnet50, vit_small_patch16_224,
                          #   deit_small_patch16_224, convnext_tiny,
                          #   efficientnet_b0, efficientnet_b2
  pretrained: true        # true: ImageNet weights (downloaded the 1st time) | false: from scratch

balancing:                # AXIS 1 of imbalance handling: acts on the data/weights
  method: none            # none: baseline (rare classes tend to recall 0)
                          # weights: CrossEntropy weighted by class rarity
                          # sampler: minority oversampling (WeightedRandomSampler);
                          #   use light/heavy augmentation with it (otherwise it memorizes the minority)
  max_weight: 20.0        # cap of the per-class weight (uncapped, 'object' would weigh ~270x)

loss:                     # AXIS 2, orthogonal to balancing
  type: ce                # ce: cross-entropy | focal: focuses on hard examples
  gamma: 2.0              # only for focal: focus strength (higher = more focus on hard ones)
  label_smoothing: 0.0    # CE target smoothing (e.g. 0.1)

augmentation: light       # none: only resize+normalize
                          # light: H/V flips + 90 rotations (safe for nadir imagery)
                          # heavy: light + random crop (0.6-1.0) + color jitter +
                          #   occasional blur + random erasing

training:
  epochs: 30              # max epochs (early stopping may stop earlier)
  batch: 128              # reduce on OutOfMemory (e.g. 96 for ViT/ConvNeXt, 64 for B2)
  lr: 3.0e-4              # learning rate (pretrained ViT prefers ~1e-4)
  weight_decay: 0.05
  optimizer: adamw        # adamw | sgd (momentum 0.9)
  scheduler: cosine       # cosine with a 2-epoch warmup | none
  amp: true               # mixed precision (faster and cheaper in VRAM)
  num_workers: 2          # data loading processes. Do NOT go above 2-3 (11GB RAM)
  early_stopping_patience: 7    # epochs without improvement before stopping (0 = off)
  checkpoint_metric: f1_macro   # what defines the "best" checkpoint:
                                #   f1_macro | balanced_accuracy | val_loss
```

**Expected result**, in `experiments/<name>/`:

| File | Contents |
|---|---|
| `config.yaml` | Frozen copy of the resolved config (with defaults) — reproducibility. |
| `splits.csv` | The exact split used (`file,label,scene,split`) — source of truth for evaluation. |
| `splits_summary.json` | Per-class count in each split — audit it before trusting the metrics. |
| `history.csv` | One row per epoch: losses, acc, balanced_acc, f1_macro, per-class recall, lr, time. |
| `best.pt` / `last.pt` | Checkpoints (best by `checkpoint_metric` / last for resuming). |
| `metrics_val.json` | Metrics of the best checkpoint on **val** (test is left for evaluate.py). |
| `tb/` | TensorBoard logs. |

Console: 1 line per epoch, e.g.
`epoch 5/30  train_loss 0.016  val_loss 0.014  acc 0.997  bal_acc 0.708  f1 0.726  rec_sea 0.999 ... (32.6s)`.
Time reference on the RTX 2070 (AMP, 224px): resnet18 ≈ 33 s/epoch with 25% of the data
(~2 min/epoch with 100%); ViT-S/ConvNeXt ≈ 2–4×; a full experiment finishes in 30–90 min.

Follow the curves of all experiments side by side:
```bash
tensorboard --logdir experiments
```

### Ready-made configs (`configs/`)

| Config | What it tests |
|---|---|
| `smoke.yaml` | End-to-end pipeline in ~1 min (2% of the data, 2 epochs). Run it first. |
| `resnet18_base.yaml` | Baseline: pretrained CNN, no imbalance handling. |
| `resnet18_weights.yaml` | Hypothesis: class weights in the loss. |
| `resnet18_sampler.yaml` | Hypothesis: minority oversampling (+ heavy augmentation). |
| `resnet18_focal.yaml` | Hypothesis: focal loss. |
| `resnet18_with_border.yaml` | Hypothesis: does keeping border tiles change anything? |
| `resnet18_spatial_split.yaml` | Hypothesis: how much does the random split inflate the metrics (leakage)? |
| `vit_small_pretrained.yaml` | Pretrained Visual Transformer. |
| `convnext_tiny.yaml` | Modern CNN. |
| `efficientnet_b0.yaml` | Efficient CNN (cheap inference). |

To create a new hypothesis: copy the closest YAML, change the `name` and change 1–2 fields.

---

## 6. `evaluate.py` — evaluate and compare

### Evaluate an experiment on the test set

```bash
python3 evaluate.py experiments/<name> [--split test|val|train] [--checkpoint best|last] [--allow-cpu]
```

Uses the experiment's **frozen** `splits.csv` (re-derives nothing) and the chosen checkpoint.
The test split is only touched here — while iterating on hypotheses, compare on val and keep
test for the final decision.

**Expected result**, in the experiment directory + printed to the console:
- `metrics_<split>.json` — accuracy, balanced_accuracy, f1_macro, per-class
  precision/recall/f1/support, average_precision of the 'object' class.
- `confusion_matrix_<split>.csv` — rows = true, columns = predicted.
- `predictions_<split>.csv` — `file,scene,label,predicted,prob_<class>...` per tile. Use the
  probabilities to mine 'object' candidates and prioritize the next labeling.

### Compare all experiments

```bash
python3 evaluate.py --compare [--split test|val] [--csv comparison.csv]
```

Scans `experiments/*/`, aggregates the metrics into a table sorted by f1_macro and prints it
(optionally saving to CSV). Experiments not yet evaluated on the requested split fall back to
the training run's `metrics_val.json` (the `split` column indicates this); with none, they are
skipped with a warning.

```
experiment             split  acc    bal_acc  f1_macro  rec_object  prec_object  ap_object  n_object  epochs
resnet18_sampler       test   0.981  0.912    0.874     0.750       0.600        0.55       8         22
...
```

The `n_object` column (support in the split) is deliberately always visible: with few
examples, `rec_object 0.750` means "6 of 8" — read the 'object' metrics as indicative, not
conclusive, while the class has only a few dozen labels.

---

## 7. `infer.py` — pre-labeling by inference

Uses already trained models to generate an initial `<prefix>_labels.csv` for a **new** scene
(already tiled, unlabeled). Instead of labeling from scratch, you only **review** the
predictions — much faster. It also serves as a generalization test of the models on unseen
scenes.

```bash
python3 infer.py sat3 --experiments experiments/vit_small_pretrained
python3 infer.py sat3 --experiments experiments/vit_small_pretrained experiments/resnet18_sampler --uncertain-threshold 0.7
```

| Option | Default | Effect |
|---|---|---|
| `--experiments DIR [DIR ...]` | (required) | 1+ trained experiments. With 2+, runs an **ensemble** (mean of the probabilities) — better pre-labels, at the cost of ~N inference passes (minutes each). |
| `--checkpoint best\|last` | best | Checkpoint of each experiment. |
| `--uncertain-threshold F` | 0.0 | 0.0 = always writes the predicted class, even with low confidence. >0 = tiles with max probability below F get `uncertain` (they show up yellow in fix_labels.py and stay out of training until reviewed). |
| `--batch N` / `--num-workers N` | 128 / 2 | Inference parameters. |
| `--overwrite` | off | **Protection**: if the scene already has a `labels.csv` (possible manual labeling!), the script aborts; this flag forces the replacement. |
| `--tiles DIR` / `--allow-cpu` | — | As in the other scripts. |

**Expected result**, in the tiles folder (~1–3 min per model per scene on the 2070):
- `<prefix>_labels.csv` — same format as manual labeling (`file,label,border,timestamp`,
  `border=0`), directly compatible with `fix_labels.py`/`label.py`/`visualize.py`/`train.py`.
- `<prefix>_auto_predictions.csv` — `file,predicted,confidence,prob_<class>...` — a record of
  what was automatic; use it to prioritize review (lowest confidence first) and to mine
  'object' candidates (high `prob_object`).
- Console summary: predicted distribution, number of uncertain tiles, mean confidence, object
  candidates.

**Recommended review flow:**

```bash
python3 infer.py sat3 --experiments experiments/vit_small_pretrained experiments/resnet18_sampler --uncertain-threshold 0.7
python3 visualize.py sat3    # quick visual audit of the predictions
python3 fix_labels.py sat3 --classes uncertain   # 1st pass: review ONLY the uncertain ones
python3 fix_labels.py sat3      # 2nd pass: general review; mark the borders by
                                #   region selection (B)
# after the review, include the scene in training: data.scenes: [sat1, sat2, sat3] in the YAML
```

The model does **not predict the border flag** — tiles come out with `border=0` and the
scene's borders must be marked during review (in fix_labels.py: drag to select the border
strip and press `B`).

---

## Interpretation and known limitations

- **Overall accuracy is misleading**: with sea+land dominating, ~99% accuracy coexists with
  recall 0 on the rare classes. Look at `balanced_accuracy`, `f1_macro` and per-class recall.
- **'object' class**: while there are only a few dozen examples, the realistic outcome is:
  (i) an excellent sea/land/cloud classifier (scene filter), (ii) qualitative evidence about
  'object', (iii) using `predictions_*.csv` to find candidates and label more. Do not draw
  statistical conclusions from 7–8 val/test examples.
- **Spatial vs random split**: if the metrics drop a lot on the spatial one, the random number
  was inflated by leakage between neighboring tiles — trust the spatial one.
- **Resources**: 8 GB VRAM (reduce `batch` if it runs out), 11 GB RAM (do not raise
  `num_workers`).
- The data keeps growing: no script assumes current counts; new scenes enter through
  `data.scenes` in the YAML after being labeled.
