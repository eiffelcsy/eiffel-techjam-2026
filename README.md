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
disk is tens of GB rather than hundreds; the SOTA images stream on demand. See
`docs/DATA.md` for the full data pipeline.

## Typical flow

1. Fit the head (stage 0): `python scripts/main/train_probe.py train/configs/probe/dinov3_wildfake_multiscale.yaml`
2. Render the base feature cache (spatial + native-frequency): `python scripts/main/build_cache.py train/configs/cache/dinov3_multiscale_nativefreq.yaml`
3. Append the SOTA rows: `python scripts/misc/append_cache.py train/configs/cache/dinov3_multiscale_nativefreq_combined.yaml`
4. Train the GRACE adapter (stage 1): `python scripts/main/train_adapter.py train/configs/train/dinov3_multiscale.yaml`
5. Train the frequency enricher (stage 2): `python scripts/main/train_enrich.py train/configs/train/dinov3_enrich_nativefreq.yaml`
6. Evaluate: `python scripts/main/run_eval.py --config eval/configs/runs/dinov3_poc_baseline.yaml`

Run the tests with `pytest`.

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