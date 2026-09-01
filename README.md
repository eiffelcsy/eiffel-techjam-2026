> **Report (PDF):** [RESONANCE — Frequency-Aware GRACE-fully Corrected Pixel Features for Robust AIGC Image Detection Under Real-World Transformations](report/Resonance_Tiktok_Techjam_2026.pdf)

# RESONANCE: Frequency-Aware GRACE-fully Corrected Pixel Features for Robust AIGC Image Detection Under Real-World Transformations

Robust AI-generated-image detection. The project tests whether a detector that
collapses under image degradation (JPEG, blur, resize) can be repaired by a
small, label-free module (`GRACE -- Gated Residual Adapter for Clean-feature Estimation`)
that corrects the detector's internal features back toward what a clean image
would produce, supplemented by a frequency branch that re-reads the image at
native resolution in a DCT basis the resize destroys.

It wraps a frozen DINOv3 ViT-S/16 trunk plus a fitted MLP head, and provides a
full evaluation harness around it: degradation sweeps, retention curves, cache
rendering, and training scripts for each stage. Images are read as seeded
multi-scale crops (128–256 px), so no global resolution shortcut survives. See
`docs/PIPELINE.md` (how the model works), `docs/DATA.md` (what it's trained on),
and `docs/EXPERIMENTS.md` (what's been run and why).

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/eiffelcsy/eiffel-techjam-2026.git
cd eiffel-techjam-2026

# create and activate a virtual environment, then:
pip install -e ".[dev]"        # base install + pytest
```

Optional extras:

```bash
pip install -e ".[wandb]"      # training-run tracking (otherwise off by default)
pip install -e ".[dashboard]"  # Streamlit dashboard (app.py)
```

### Model access

The trunk (`facebook/dinov3-vits16-pretrain-lvd1689m`) is gated on the Hugging
Face Hub. Accept the license at
<https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m>, run
`hf auth login` once, or point `backbone_id` at a local mirror.

### Data

Training uses a combined corpus assembled from two parts:

- **WildFake** — a weighted, custom-sampled 49,999-image train split (plus a
  10,000-image held-out validation split), skewed toward recent diffusion
  models over the older SD material that dominates the raw corpus.
- **SOTA additions** — 6,000 FLUX and 200 Seedream generated images streamed
  from the Hugging Face Hub, plus a disjoint sample of extra reals, appended to
  the WildFake manifest (58,656 train / 14,000 val in total).

Fetch the WildFake slice, then build the combined manifest:

```bash
python scripts/misc/fetch_wildfake_train.py --dry-run     # plan, download nothing
python scripts/misc/fetch_wildfake_train.py               # download (resumable)
python scripts/main/build_combined_manifest.py load_data/configs/datasets/wildfake_train_combined.yaml
```

Each WildFake archive is mined for just the sampled members and deleted, so peak
disk is tens of GB rather than hundreds; the SOTA images stream on demand.

#### Test / benchmark data (`wildfake-coco-dalle3`)

The evaluation benchmark is **13,841 images = 4,998 COCO val2017 reals + 8,843
DALL-E 3 "Advanced" fakes**, held out from training by construction (`DALLE` and
`coco` are excluded from the training sample). The metadata tables and both image
archives come from the WildFake ModelScope repo (`hy2628982280/WildFake`); the
metadata is small, the images are not (`coco.zip` ~2.4 GB, `DALLE.zip` ~25 GB):

```bash
pip install modelscope

# On Windows, set PYTHONIOENCODING=utf-8 first -- the downloader prints a "->"
# through the console codepage and dies on the encode otherwise.
modelscope download --dataset hy2628982280/WildFake \
    split_train_test/csv_file/total_split/test_metadata.csv \
    split_train_test/csv_file/total_split/train_metadata.csv \
    Images/Real/coco.zip \
    Images/Diffusion_based/DALLE.zip \
    --local_dir data/wildfake_test
```

Unpack both archives under `data/wildfake_test/images/` so the tables' paths
hang off it — they name `./Real/coco/...` and `./Diffusion_based/DALLE/...`, so
those two directories must be immediate children of `data/wildfake_test/images/`:

```bash
mkdir -p data/wildfake_test/images
unzip -o data/wildfake_test/coco.zip  -d data/wildfake_test/images
unzip -o data/wildfake_test/DALLE.zip -d data/wildfake_test/images
rm data/wildfake_test/coco.zip data/wildfake_test/DALLE.zip
```

Expected layout:

```
data/wildfake_test/split_train_test/csv_file/total_split/test_metadata.csv
data/wildfake_test/split_train_test/csv_file/total_split/train_metadata.csv
data/wildfake_test/images/Real/coco/coco2017/val2017/*.jpg
data/wildfake_test/images/Diffusion_based/DALLE/Advanced/DALLE3/*/*.jpg
```

Build the manifest — 13,841 rows (4,998 real / 8,843 generated):

```bash
python scripts/main/build_manifest.py --config load_data/configs/datasets/wildfake_coco_dalle3.yaml
```

Images are referenced in place (no second copy, no re-encode), so only the two
archives are ever on disk at once. See `docs/DATA.md` for the full data pipeline.

## Reproducing the results

Everything runs from the repo root with the venv active. The full chain, in
order — data, then train each stage, then evaluate on the held-out benchmark:

0. **Data** — train and test corpora (see [Data](#data)):

   ```bash
   python scripts/misc/fetch_wildfake_train.py
   python scripts/main/build_combined_manifest.py load_data/configs/datasets/wildfake_train_combined.yaml
   python scripts/main/build_manifest.py --config load_data/configs/datasets/wildfake_coco_dalle3.yaml
   ```

1. **Stage 0 — probe** (fit the frozen trunk's classification head on clean
   features):

   ```bash
   python scripts/main/train_probe.py train/configs/probe/dinov3_wildfake_multiscale.yaml
   ```

   Writes `checkpoints/probe/dinov3_wildfake_multiscale/head.pt`, loaded by
   every detector config.

2. **Render the feature cache** (spatial + native-frequency), then append the
   SOTA rows:

   ```bash
   python scripts/main/build_cache.py train/configs/cache/dinov3_multiscale_nativefreq.yaml
   python scripts/misc/append_cache.py train/configs/cache/dinov3_multiscale_nativefreq_combined.yaml
   ```

3. **Stage 1 — GRACE adapter**:

   ```bash
   python scripts/main/train_adapter.py train/configs/train/dinov3_multiscale.yaml
   ```

   Writes `checkpoints/grace/dinov3_multiscale/ema.pt`, loaded by every adapted
   detector config.

4. **Stage 2 — frequency enricher**:

   ```bash
   python scripts/main/train_enrich.py train/configs/train/dinov3_enrich_nativefreq.yaml
   ```

   Writes `checkpoints/grace/dinov3_enrich_nativefreq/enricher.pt`.

5. **Evaluate** on the held-out benchmark, crop200 arm. Each run writes
   `results/{run_id}__{detector}__{dataset}.json` and prints the headline
   retention table:

   ```bash
   python scripts/main/run_eval.py --config eval/configs/runs/dinov3_poc_baseline_arms.yaml   # unadapted retention (the denominator)
   python scripts/main/run_eval.py --config eval/configs/runs/dinov3_poc_freq.yaml            # +grace and +grace-freq (the headline rows)
   ```

6. **Report** — baseline-normalized retention and the summary tables/figures:

   ```bash
   python scripts/misc/compare.py \
     --baseline results/dinov3_poc_baseline_arms__dinov3-wildfake-crop200__wildfake-coco-dalle3.json \
     --adapted  results/dinov3_poc_freq__dinov3-crop200+grace__wildfake-coco-dalle3.json

   python scripts/misc/compare.py \
     --baseline results/dinov3_poc_baseline_arms__dinov3-wildfake-crop200__wildfake-coco-dalle3.json \
     --adapted  results/dinov3_poc_freq__dinov3-crop200+grace-freq__wildfake-coco-dalle3.json

   python scripts/main/report.py --results results/ --out results/summary
   ```

Optional gate and ablations: the identity null-adapter check
(`eval/configs/runs/dinov3_poc_identity.yaml` + `compare.py --assert-identity`)
runs right after the baseline, before anything is trained; all stage-1 sweeps
and seed-variance runs are in `docs/EXPERIMENTS.md`. Run the tests with
`pytest`.

## Dashboard

A minimal Streamlit app wraps the inference path (`eval.inference.predict_dir`)
for interactively scoring images with any detector config:

```bash
pip install -e ".[dashboard]"   # if not installed above
streamlit run app.py
```

From the sidebar you can pick a detector config (`eval/configs/detectors/*.yaml`),
the device, batch size, and see the model's end-to-end parameter count. Then
either upload one or more images or paste a path to a folder of images, and hit
**Run inference** to get P(AI-generated) per image. The model loads once and is
cached across reruns.

## Future work (in order of triviality and importance)

1. **Hyperparameter Sweeps**: To find optimal hyperparameters for a wider range of input training data, better generalization to held-out images.
2. **Clean up the data loading pipeline**: Remove the need for appending the SOTA rows, it should fetch and build manifest together with the WildFake data.
3. **Assess detector invariance**: GRACE is detector-agnostic by construction and should be evaluated to other frozen backbones and heads to see how much the observed robustness gain transfers.
4. **Broader benchmark**: Span more generators, real-world capture-and-redistribution pipelines to more accurately assess deployment impact.
5. **Methodology ideas**: Scalar severity estimate could be replaced by richer self-supervised degradation description, frequency branch improvements like learnable band structures, top-k coefficient selection etc.