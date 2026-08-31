# aigc-robustness

Robust AI-generated-image detection. The project tests whether a detector that
collapses under image degradation (JPEG, blur, resize) can be repaired by a
small, label-free module (`GRACE -- Gated Residual Adapter for Clean-feature Estimation`)
that corrects the detector's internal features back toward what a clean image 
would produce — optionally supplemented by a frequency branch that re-reads the 
image in a basis the resize destroyed.

It wraps a frozen DINOv3 ViT-S/16 trunk plus a fitted MLP head, and provides a
full evaluation harness around it: degradation sweeps, retention curves, cache
rendering, and training scripts for each stage. See `docs/PIPELINE.md` (how the
model works), `docs/DATA.md` (what it's trained on), and `docs/EXPERIMENTS.md`
(what's been run and why).

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
```

### Model access

The trunk (`facebook/dinov3-vits16-pretrain-lvd1689m`) is gated on the Hugging
Face Hub. Accept the license at
<https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m>, run
`hf auth login` once, or point `backbone_id` at a local mirror.

### Data

The training data is a custom-sampled subset of the WildFake dataset
(~1200 GB in full; this project uses a 50k-image slice). Fetch exactly the
sampled images:

```bash
python scripts/misc/fetch_wildfake_train.py --dry-run     # plan, download nothing
python scripts/misc/fetch_wildfake_train.py               # download (resumable)
```

Each archive is mined for just the sampled members and deleted, so peak disk is
tens of GB rather than hundreds. For more information on data pipeline, see
`docs/DATA.md`.

## Typical flow

1. Fit the head (stage 0): `python scripts/main/train_probe.py train/configs/probe/dinov3_wildfake_multiscale.yaml`
2. Render the feature cache: `python scripts/main/build_cache.py train/configs/cache/dinov3_multiscale.yaml`
3. Train the adapter (stage 1): `python scripts/main/train_adapter.py train/configs/train/dinov3_multiscale.yaml`
4. Evaluate: `python scripts/main/run_eval.py --config eval/configs/runs/dinov3_poc_baseline.yaml`

Run the tests with `pytest`.
