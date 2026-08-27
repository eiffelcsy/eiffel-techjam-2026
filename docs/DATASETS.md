# Datasets

NTIRE 2026 Robust AI-Generated Image Detection. Needs ~230 GB free
(114 GB of zips + the same again unpacked).

## 1. Download

```bash
cd "$(git rev-parse --show-toplevel)"

.venv/bin/hf download deepfakesMSU/NTIRE-RobustAIGenDetection-train \
  --repo-type dataset --local-dir /tmp/ntire_train_dl

.venv/bin/hf download deepfakesMSU/NTIRE-RobustAIGenDetection-val \
  --repo-type dataset --local-dir /tmp/ntire_val_dl \
  --include "val_images.zip" "val_images_hard.zip" "val_labels.csv" "val_hard_labels.csv"
```

## 2. Unpack

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

If there's an extra directory level, flatten it or fix `root:` in the matching
config under `eval_pipeline/configs/datasets/` — one line each.

## 3. Build manifests

```bash
cd eval_pipeline
for d in ntire_train ntire_val ntire_val_hard; do
  PYTHONPATH=. ../.venv/bin/python scripts/build_manifest.py \
    --config configs/datasets/$d.yaml
done
```

Three builds only. `ntire_train_eval.yaml` and `ntire_val_distorted.yaml` carry
no `source:` — they are the other split of the manifests these three write.

Expected: `ntire_train` 277,650 rows (shards 0–4 `train`, shard 5 `validation`);
`ntire_val` 5000/5000 `undistorted` + 5000/5000 `distorted`; `ntire_val_hard`
1250/1250. A mismatch means the unpack nesting is wrong.

## 4. Clean up

```bash
rm -rf /tmp/ntire_train_dl /tmp/ntire_val_dl
```
