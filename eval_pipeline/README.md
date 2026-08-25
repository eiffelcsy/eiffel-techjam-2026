# robust-aigc-eval

Evaluation harness for AI-generated image detectors under everyday image
handling, plus a standalone inference script.

Model-agnostic and dataset-agnostic: no model id, dataset id, or device appears
anywhere in the code. Detectors and dataset sources are named in config by
dotted import path, so pointing the harness at something new is a config file,
never a patch.

## What it answers

> A detector scores 0.97 AUC on pristine images. What does it score after the
> image has been posted, thumbnailed, filtered, or screenshotted?

## Why this evaluation strategy

**Clean AUC is not the number you deploy against.** Detectors lean on local,
high-frequency traces left by the generator's upsampling stack -- exactly the
signal that quantisation, low-pass filtering and resampling destroy first. The
clearest evidence is from the training side: Wang et al. found that augmenting
with the very same post-processing operations is what buys a detector any
cross-generator generalization at all, "even when the target images are not
post-processed themselves" [1]. A benchmark that scores only pristine images is
measuring the one condition under which the fragile signal is intact.

The transforms here are therefore ordinary image handling, not adversarial
perturbation. No attacker is required: re-encoding for a feed already erases
the evidence.

**L1 is a one-factor-at-a-time sweep** because that is the field's standard
robustness protocol -- JPEG swept over quality factor, blur over sigma, resize
over scale. Keeping it makes these numbers comparable to published robustness
tables, and every result is attributable to exactly one cause.

**L2/L3 exist because single-perturbation testing is the known gap, and this
design agrees with where the literature landed rather than claiming to precede
it.** Cui et al. put the limitation plainly: existing robustness protocols
"typically apply only one perturbation at a time, such as JPEG at a fixed
quality factor or Gaussian blur at a single strength" [2]. Their answer,
RealDeg-Bench, is a compound-degradation benchmark over seven operators (JPEG,
Gaussian blur, resize, Gaussian noise, brightness, contrast, saturation) with a
degradation depth of N ∈ {1..5} sequential operators. That is nearly this
harness's operator pool -- it bundles brightness/contrast/saturation into one
`color_jitter` and adds `center_crop` -- and its depth range is exactly this
harness's L2 (2 transforms) and L3 (3-5). Two independent designs converging on
the same structure is the argument for it. Real propagation stacks operations:
re-compress, resize, re-upload.

**Retention is chance-corrected** -- `(auc_deg - 0.5) / (auc_clean - 0.5)` --
because 0.5 is the floor, not zero. Reporting a raw ratio would score a
detector that has collapsed to coin-flipping at ~0.5, reading as "half the
performance retained" when in fact none of it is.

**The error split is reported separately from AUC** because real and generated
images demonstrably do not respond to the same perturbation in the same way:
natural images "preserve stable semantic representations under small,
structured perturbations, whereas generated images exhibit markedly larger
feature drift" [3]. A single pooled AUC averages that asymmetry away. Which
*direction* a given transform pushes a given detector -- compression plausibly
smoothing generative traces so fakes read as real (FNR climbs), noise plausibly
adding high-frequency energy so reals read as fake (FPR climbs) -- is this
harness's hypothesis, and the FPR/FNR table at a clean-fixed threshold is what
tests it. Do not read it as an established result.

## The transform grid

Six transforms, each a real-world artefact rather than an adversarial attack,
each with an explicit parameter list.

| Transform | Parameter | Values | Real-world analog |
|---|---|---|---|
| JPEG compression | quality | 90, 70, 50, 30 | Social-media re-encode, messaging |
| Gaussian blur | sigma | 0.5, 1.0, 2.0 | Out-of-focus |
| Resize | scale | 0.5x, 0.25x then upscale | Thumbnail generation |
| Gaussian noise | sigma | 0.02, 0.05, 0.10 | Low-light sensor noise |
| Color jitter | strength | +/-20% brightness, contrast, saturation | Filter apps, auto-enhance |
| Center crop | keep | 80% | Profile-picture cropping, framing |

Parameters apply identically at every resolution -- no clamping, no minimum-size
guard. Some are resolution-relative (`scale`, `keep`) and some absolute (`sigma`
in pixels), which is a property of the artefacts themselves. Degradation happens
at the image's native size, *before* a detector's own preprocessing, so the
degraded eval set is identical no matter which detector scores it.

## The four composition levels

| Level | What | How |
|---|---|---|
| L0 | clean | reference: sets the threshold and the retention denominator |
| L1 | single | the 14 grid conditions, one at a time (exhaustive sweep) |
| L2 | pair | 2 distinct transforms, params drawn from the grid, random order |
| L3 | multi | 3-5 distinct transforms, sampled the same way |

L1 is a one-factor-at-a-time sweep: every result is attributable to exactly one
cause. L2/L3 are Monte-Carlo samples of the composition space -- too large to
enumerate -- drawn per image and keyed on `(index, level, replicate)` so the
degraded set is identical across runs and detectors. Transforms are drawn
without replacement and applied in shuffled order, because they do not commute.

`n_replicates` re-draws L2/L3 over the same images: more coverage of the
combination space and a tighter CI, without growing the eval set.

## What comes out

1. **Headline** -- detectors x levels: clean, L1, L2, L3 AUC and retention.
2. **Interaction** -- measured L2/L3 retention against what the L1 marginals
   predict under independence. A large negative gap is the finding: the
   single-transform benchmark overstated that detector's robustness.
3. **By transform (L1)** -- detectors x the six transforms, plus degradation
   curves (AUC vs parameter, one panel per transform).
4. **Error split** -- FPR and FNR at a threshold fixed on clean data.

Plus a worst-recipe table: which combinations at L2/L3 did the damage.

`summary.operating_envelope` is the deepest level still at or above the
retention floor (`retention_floor` in `eval.yaml`, default 0.5 -- a reporting
convention, not a derived quantity).

## Layout

```
configs/           eval.yaml (one run) + degradations.yaml (the transform grid)
pipeline/
  config.py        yaml -> dataclasses
  data/            manifest (the one table everything keys off) + sources + datasets
  degrade/         ops.py (the six transforms) + conditions.py (levels, sampling)
  detectors/       frozen-detector contract + one generic Hub adapter
  eval/            metrics, runner, report
  inference/       image dir -> P(generated) per image
  utils/           seeding, io, import-path resolution
scripts/           build_manifest / run_eval / report / predict
tests/             one end-to-end run on synthetic data
```

## Install

```bash
pip install -e ".[dev]"
```

`pyproject.toml` is the single source of truth for dependencies -- there is no
`requirements.txt`. Python >= 3.10; runs on CUDA, MPS, or CPU with no config
change (`device: auto`).

The one thing the project file cannot express portably is torch's CUDA wheel
index, which is cluster-specific. On a GPU node, install torch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

Set `HF_HOME` (and `HF_DATASETS_CACHE`) if `$HOME` is small or read-only --
`transformers` and `datasets` honour them natively.

## Pointing it at a model and a dataset

Both are specs of the same shape: a dotted `target` plus its constructor `args`.
Nothing below requires touching code in `pipeline/`.

**A detector** -- `configs/detectors/<name>.yaml`. Any Hub
`AutoModelForImageClassification` checkpoint works through the generic adapter:

```yaml
name: my-detector                 # display name, used in result filenames
target: pipeline.detectors.hf.HFImageClassifier
args:
  model_id: some-org/some-detector
  # fake_labels: [artificial]     # only if the head's id2label is unusual
device: auto
```

Which output index means *generated* is resolved from the model's own
`config.id2label`, not assumed -- Hub detectors disagree, and a fair number put
the fake class at index 0. An unrecognised or ambiguous label map raises rather
than guessing, because guessing wrong yields a plausible-looking `1 - AUC`.

For a detector the generic adapter cannot cover, subclass `FrozenDetector` in
your own module and point `target:` at it. Two contract details matter:
`forward` returns a `(B,)` fake-minus-real logit (higher = generated), and if
your class carries real weights, override `preprocess_fn()` to return a
callable that closes over only what preprocessing needs. The Dataset is forked
into DataLoader workers, so a preprocessing function holding the model pickles
every parameter -- which fails outright on MPS and CUDA and wastes memory on
CPU. `pipeline/detectors/hf.py` shows the pattern in about six lines.

**A dataset** -- `configs/data/<name>.yaml`. Any Hub image dataset with a label
column:

```yaml
target: pipeline.data.sources.HFImageDatasetSource
args:
  dataset_id: some-org/some-dataset
  split: test
  fake_classes: [FAKE]            # class names meaning *generated*
  # real_classes: [REAL]          # name these on a multi-class dataset
  generator: whatever-made-them
  limit: 2000                     # per class; omit for all
  streaming: true                 # required for large repos, see below
out: data/some-dataset/manifest.parquet
```

Label polarity is declared, never inferred: public AIGC datasets disagree on
whether `0` means real or generated. When the label column is a plain integer
rather than a `ClassLabel`, class names are its stringified values, so an
unnamed binary label is `fake_classes: ["1"]`.

**On a dataset with more than two classes, name `real_classes` explicitly.**
Left unset, everything not listed as fake is folded into *real* -- which is
right for a binary dataset and silently wrong for anything else. Naming both
sets drops the classes in neither, so a three-class dataset (real / fully
synthetic / locally tampered) can be evaluated as the two-class question the
detector was actually trained for, instead of quietly relabelling the third
class as authentic.

**`streaming: true` is not cosmetic on a large repo.** A plain
`load_dataset(...)` materializes every shard before yielding one row -- for a
140 GB dataset that is the whole 140 GB, even with `limit` set. Streaming plus
`limit` fetches only the shards it needs. Set `HF_TOKEN` too: unauthenticated
streaming is rate-limited and drops connections mid-shard.

`ImageDirSource` is the equivalent for local directories. A new kind of source
is a new class plus a new `target:` line.

## Usage

```bash
python scripts/build_manifest.py --config configs/data/<name>.yaml
python scripts/run_eval.py --config configs/eval.yaml       # -> results/*.json
python scripts/report.py --results results/                 # -> tables + figures

python scripts/predict.py --image-dir path/to/images \
    --detector configs/detectors/<name>.yaml --out preds.json
```

`preds.json` is a list of `{"image_path": str, "pred": float}` where `pred` is
P(AI-generated).

`data.max_images` is the knob that moves wall-clock by an order of magnitude: a
full run is `1 + 14 + 2 x n_replicates` conditions over the eval subset. Use a
small value for the first loop.

## Invariants

- **Format neutrality.** Every image is decoded through one identical path
  before a detector sees it, and sources write a single on-disk format, so no
  container statistic is left correlated with the label.
- **Fixed eval subset.** Sampled deterministically from a seed, so it is
  identical across runs without needing a cache file.
- **Paired conditions.** Every condition scores the same images in the same
  order, so clean and degraded scores can be compared per image.
- **Deterministic sampling.** `stable_seed(index, level, replicate)`, hashed
  with blake2b and never with the builtin `hash()` (which is salted per
  process) -- every detector sees the identical degraded set, so a difference
  between detectors is never a difference in the draw.
- **Recipes are logged.** What was applied to each image is recorded, so the
  composed levels can be analysed after the fact rather than just scored.
- **Frozen detectors.** Loaded, evaluated, never trained.

## References

1. Wang, Wang, Zhang, Owens, Efros. *CNN-Generated Images Are Surprisingly Easy
   to Spot... for Now.* CVPR 2020, pp. 8692-8701.
   <https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html>
2. Cui, Liu, Zou, Qin, Xu, Wang, Wei, Zhou, Liu, Wang, Wu. *GlobalForge: Towards
   Robust AI-Generated Image Detection.* arXiv:2607.14684 (introduces
   RealDeg-Bench). <https://arxiv.org/abs/2607.14684>
3. Wang, Chen, Zhang, Bian, Guo, Ma, Li. *RA-Det: Towards Universal Detection of
   AI-Generated Images via Robustness Asymmetry.* arXiv:2603.01544.
   <https://arxiv.org/abs/2603.01544>
