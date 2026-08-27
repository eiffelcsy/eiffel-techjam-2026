# Datasets

Two corpora, three jobs:

| corpus | role |
|---|---|
| NTIRE 2026 train | stage 0 (head fit) and stage 1 (adapter fit) |
| NTIRE 2026 val + val_hard | stage-0 epoch selection, stage-1 held-out-image validation |
| WildFake (COCO val2017 + DALL-E 3) | **the eval set** — held out from everything above |

NTIRE needs ~230 GB free (114 GB of zips + the same again unpacked). WildFake
needs ~28 GB of archives; images are referenced in place, never copied.

Every dataset config lives in `eval_pipeline/configs/datasets/` and carries its
own detailed rationale in comments. `data/` sits at the **repo root** because
both packages read it — a dataset config's `../data/...` resolves the same
whether the working directory is `eval_pipeline/` or `grace_adapter/`.

## 1. NTIRE — download

```bash
cd "$(git rev-parse --show-toplevel)"

.venv/bin/hf download deepfakesMSU/NTIRE-RobustAIGenDetection-train \
  --repo-type dataset --local-dir /tmp/ntire_train_dl

.venv/bin/hf download deepfakesMSU/NTIRE-RobustAIGenDetection-val \
  --repo-type dataset --local-dir /tmp/ntire_val_dl \
  --include "val_images.zip" "val_images_hard.zip" "val_labels.csv" "val_hard_labels.csv"
```

## 2. NTIRE — unpack

```bash
mkdir -p data/ntire_train data/ntire_val/images data/ntire_val_hard/images

for i in 0 1 2 3 4 5; do
  unzip -q /tmp/ntire_train_dl/shard_$i.zip -d data/ntire_train/
done

cp /tmp/ntire_val_dl/val_labels.csv      data/ntire_val/
cp /tmp/ntire_val_dl/val_hard_labels.csv data/ntire_val_hard/
unzip -q /tmp/ntire_val_dl/val_images.zip      -d data/ntire_val/images
unzip -q /tmp/ntire_val_dl/val_images_hard.zip -d data/ntire_val_hard/images
```

Target layout — **verify before building**, the zips may add a nesting level:

```
data/ntire_train/shard_{0..5}/images/*.jpg
data/ntire_train/shard_{0..5}/labels.csv
data/ntire_val/images/*.jpg          data/ntire_val/val_labels.csv
data/ntire_val_hard/images/*.jpg     data/ntire_val_hard/val_hard_labels.csv
```

```bash
find data/ntire_train/shard_0/images data/ntire_val/images \
     data/ntire_val_hard/images -name "*.jpg" | head -3
```

If there is an extra directory level, flatten it or fix `root:` in the matching
config — one line each.

## 3. WildFake — download and unpack

WildFake (Hong et al., AAAI 2025) is hosted on ModelScope, not the Hub. Only two
archives are needed; the metadata tables are small.

```bash
pip install modelscope
modelscope download --dataset hy2628982280/WildFake \
    split_train_test/csv_file/total_split/test_metadata.csv \
    split_train_test/csv_file/total_split/train_metadata.csv \
    Images/Real/coco.zip \
    Images/Diffusion_based/DALLE.zip \
    --local_dir data/wildfake
```

On Windows, set `PYTHONIOENCODING=utf-8` first — the downloader prints a `->`
through the console codepage and dies on the encode otherwise.

Unpack both under `data/wildfake/images/` so the tables' relative paths resolve
against it:

```
data/wildfake/images/Real/coco/coco2017/val2017/*.jpg
data/wildfake/images/Diffusion_based/DALLE/Advanced/DALLE3/*/*.jpg
```

`DALLE.zip` is ~25 GB and also holds DALL-E 2; `coco.zip` is ~2.4 GB and also
holds train2017/test2017. Only the `Advanced/DALLE3/` and `val2017/` subtrees are
read, selected by `path_prefix` — WildFake's own columns cannot express the
subset, since every COCO image carries `Architecture=coco` whichever directory it
came from.

## 4. Build manifests

```bash
cd eval_pipeline
for d in ntire_train ntire_val ntire_val_hard wildfake_coco_dalle3; do
  PYTHONPATH=. ../.venv/bin/python scripts/build_manifest.py \
    --config configs/datasets/$d.yaml
done
```

Four builds only. `ntire_val_distorted.yaml` carries no `source:` — it is the
other split of the manifest `ntire_val.yaml` writes. `ntire_train_eval.yaml` is
superseded and selects zero rows.

Expected:

| manifest | rows |
|---|---|
| `ntire_train` | 277,643, all `split: train` (all six shards) |
| `ntire_val` | 5000 `undistorted` + 5000 `distorted`, each balanced 50/50 |
| `ntire_val_hard` | 2500 `hard`, balanced 1250/1250 |
| `wildfake` | 13,841 — 4998 real (COCO val2017) / 8843 fake (DALL-E 3) |

A mismatch means the unpack nesting is wrong. `on_missing: error` is the default
for every CSV source here for that reason: an archive unpacked one level deeper
than expected would otherwise build a benchmark quietly missing most of its
images.

**Never rebuild a manifest a feature cache was rendered against.** Rebuilding
changes `manifest_sha`, and `grace_adapter`'s cache is fingerprinted on it.

## 5. Clean up

```bash
rm -rf /tmp/ntire_train_dl /tmp/ntire_val_dl
```

## Notes that affect results

- **Shard 5 is training data now.** It used to be held out under
  `split: validation` (read through `ntire_train_eval.yaml`). Stage-0 selection
  moved to the challenge's own val sets, which made holding it back pure cost.
- **NTIRE train mixes in-the-wild transformations into both classes** and ships
  no `is_distorted` flag to filter them. GRACE's premise is a head fit on *clean*
  images, so "fit on clean data" is only approximately true of any head trained
  here. It cannot be filtered — it is a property of the dataset.
- **`ntire_val_distorted` and `ntire_val_hard` are level-0-only evaluations.**
  The degradation grid stacked on top of an unknown prior transform makes both
  the L0 reference and L1's one-cause claim untrue.
- **WildFake is the reported benchmark**, which is what makes selecting on NTIRE
  val sound: nothing selected on those sets is ever reported against them.
