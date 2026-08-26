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
degradation depth of N ∈ {1..5} sequential operators. That is this harness's
operator pool -- brightness, contrast and saturation are separate operators
here too, swept in both directions, and `center_crop` is the one addition --
and its depth range is exactly this harness's L2 (2 transforms) and L3 (3-5). Two independent designs converging on
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

Eleven transforms, each a real-world artefact rather than an adversarial
attack, each with an explicit parameter list.

| Transform | Group | Parameter | Values | Real-world analog |
|---|---|---|---|---|
| JPEG compression | compression | quality | 90, 70, 50, 30 | Social-media re-encode, messaging |
| Gaussian blur | blur | sigma | 0.5, 1.0, 2.0 | Out-of-focus |
| Resize | resampling | scale | 0.5x, 0.25x then upscale | Thumbnail generation |
| Gaussian noise | noise | sigma | 0.02, 0.05, 0.10 | Low-light sensor noise |
| Brightness up / down | photometric | factor | 1.2 / 0.8 | Filter apps, auto-enhance |
| Contrast up / down | photometric | factor | 1.2 / 0.8 | Filter apps, auto-enhance |
| Saturation up / down | photometric | factor | 1.2 / 0.8 | Filter apps, auto-enhance |
| Center crop | framing | keep | 80% | Profile-picture cropping, framing |

The six `photometric` transforms are colour jitter unbundled: +/-20% on each of
brightness, contrast and saturation, one attribute at a time. They share a
group, so the report can aggregate them back into a single colour-jitter
family, but each is its own L1 condition -- a detector that only breaks on a
brightness lift is visible as such, rather than averaged into one stochastic
"jitter" number.

Parameters apply identically at every resolution -- no clamping, no minimum-size
guard. Some are resolution-relative (`scale`, `keep`) and some absolute (`sigma`
in pixels), which is a property of the artefacts themselves. Degradation happens
at the image's native size, *before* a detector's own preprocessing, so the
degraded eval set is identical no matter which detector scores it.

## The four composition levels

| Level | What | How |
|---|---|---|
| L0 | clean | reference: sets the threshold and the retention denominator |
| L1 | single | the 19 grid conditions, one at a time (exhaustive sweep) |
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
3. **By transform (L1)** -- detectors x the eleven transforms, plus degradation
   curves (AUC vs parameter, one panel per transform).
4. **Error split** -- FPR and FNR at a threshold fixed on clean data.

Plus a worst-recipe table: which combinations at L2/L3 did the damage.

`summary.operating_envelope` is the deepest level still at or above the
retention floor (`retention_floor` in the run config, default 0.5 -- a
reporting convention, not a derived quantity).

## Layout

```
../data/           MATERIALIZED DATASETS -- at the repo root, shared with
                   ../grace_adapter, because both packages read them and a
                   dataset config's `../data/...` resolves the same from either
configs/
  datasets/        what a dataset IS -- source, manifest path, split
  detectors/       what a detector IS -- import path and args
  runs/            detectors x datasets, plus how the run executes
  degradations.yaml  the transform grid
  defaults.yaml    annotated reference: every key with its default
pipeline/
  config.py        yaml -> dataclasses
  data/            manifest (the one table everything keys off) + sources + datasets
  degrade/         ops.py (the eleven transforms) + conditions.py (levels, sampling)
  detectors/       frozen-detector contract, one generic Hub adapter, the zoo
  eval/            metrics, runner, report
  inference/       image dir -> P(generated) per image
  utils/           seeding, io, import-path resolution
scripts/           build_manifest / run_eval / report / predict
tests/             one end-to-end run on synthetic data
third_party/       the zoo's upstream repos, cloned by hand (gitignored)
checkpoints/       the zoo's weights, downloaded by hand (gitignored)
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

Three kinds of config, one file each, one directory each:

```
configs/datasets/<name>.yaml    what a dataset IS
configs/detectors/<name>.yaml   what a detector IS
configs/runs/<name>.yaml        detectors x datasets
```

A dataset and a detector each name their component the same way -- a dotted
`target` plus its constructor `args` -- and a run references the other two by
path rather than restating them. Nothing below requires touching code in
`pipeline/`. `configs/defaults.yaml` is the annotated reference: every key of
every kind with its default, and not a file the harness loads.

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

`pipeline/detectors/dinov3.py` is the one detector here that is not somebody
else's published checkpoint: a frozen DINOv3 ViT-S/16 trunk plus an MLP probe
head, fit on clean images by `../grace_adapter/scripts/train_probe.py`. It exists
so GRACE has a detector whose trunk/head seam is a construction rather than a
reconstruction of a vendored repo's internals -- see that project's README
section 8. The backbone is a licence-gated Hub repo; `backbone_id` takes any
mirror.

For a detector the generic adapter cannot cover, subclass `FrozenDetector` in
your own module and point `target:` at it. Two contract details matter:
`forward` returns a `(B,)` fake-minus-real logit (higher = generated), and if
your class carries real weights, override `preprocess_fn()` to return a
callable that closes over only what preprocessing needs. The Dataset is forked
into DataLoader workers, so a preprocessing function holding the model pickles
every parameter -- which fails outright on MPS and CUDA and wastes memory on
CPU. `pipeline/detectors/hf.py` shows the pattern in about six lines.

**A dataset** -- `configs/datasets/<name>.yaml`. Any Hub image dataset with a
label column:

```yaml
name: some-dataset                # short id, used in result filenames
manifest: ../data/some-dataset/manifest.parquet
split: test                       # which split to score; null = every row

source:                           # read only by build_manifest.py
  target: pipeline.data.sources.HFImageDatasetSource
  args:
    dataset_id: some-org/some-dataset
    split: test
    fake_classes: [FAKE]          # class names meaning *generated*
    # real_classes: [REAL]        # name these on a multi-class dataset
    generator: whatever-made-them
    limit: 2000                   # per class; omit for all
    streaming: true               # required for large repos, see below
```

`manifest` is where the table goes, and `source` says how to build it -- once
the manifest exists, evaluation reads `manifest` and `split` and ignores
`source` entirely.

**`../data/`, not `data/`.** Config paths resolve against the working directory,
and this harness is not the only thing that reads a dataset config: `run_eval.py`
runs with the CWD here, while `../grace_adapter`'s training scripts run with the
CWD there. `data/` therefore sits at the **repo root**, above both, and
`../data/...` names the same directory from either. A manifest path inside one
package would resolve for one caller and silently `FileNotFoundError` for the
other. The two `split` keys are not the same knob: the inner one
picks which split to *pull* from the Hub, the outer one picks which split to
*score* out of the built manifest.

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

**`CsvMetadataSource` is for the benchmarks that are not on the Hub** -- shipped
as archives plus a metadata table, which is where the label, the provenance and
the official split actually live. `configs/datasets/wildfake_coco_dalle3.yaml` is
the worked example, and the reason the class exists is its `path_prefix` knob:
WildFake files every COCO image under one `coco` architecture, so no column in
its table separates val2017 from train2017, and the subset can only be named by
path. It **references images in place and copies nothing** -- not to save disk,
but because those files are already the bytes the dataset authors shipped, and
re-saving a JPEG as PNG resamples away exactly the compression artifacts an AIGC
detector keys on. `on_missing: error` is the default for the same reason
`streaming` matters above: an archive unpacked one level deeper than expected
otherwise yields a benchmark quietly missing most of its images, and the
polarity check only notices when a whole class disappears.

**`ConcatSource` chains several sources into one manifest**, which is how a
dataset gets a *train* split. Evaluation never needs one -- the harness scores a
held-out set and nothing else -- but `../grace_adapter` fits a classification
head and trains an adapter, and both must be provably disjoint from what is
scored. `configs/datasets/ntire_train.yaml` is the worked example: NTIRE's shards
0-4 and its held-out shard 5 in one table, keyed by disjoint manifest indices.

Its `prefix: true` default is load-bearing, not tidy. `HFImageDatasetSource`
names files by position within the split it is iterating, so two splits of the
same dataset both start at `images/00000000_0.png`; without a per-source
subdirectory the second silently overwrites the first and the manifest's train
and validation rows end up pointing at the same pixels.

**A run** -- `configs/runs/<name>.yaml`. Pairs the detectors with the datasets to
score them on:

```yaml
run_id: my_run
detector: configs/detectors/my-detector.yaml   # path, or the mapping inline
datasets:
  - configs/datasets/my-dataset.yaml           # paths, or mappings inline

degrade: {}         # `{}` takes every default: all four levels, all eleven
                    # transforms, n_replicates: 3, seed: 0
```

That is a complete run config -- everything else takes its default from
`configs/defaults.yaml`. Loader knobs live here rather than on the dataset:
`batch_size` and `num_workers` are properties of this machine, and
`max_images` is a scope decision for this run, so none of them describe what
the dataset is.

Write `detectors:` with a list instead of `detector:` to score several in one
command. Each is loaded, run against every dataset, and released before the next
is built, so a zoo costs one detector's memory however many are named:

```yaml
detectors:
  - configs/detectors/gapl.yaml
  - configs/detectors/rine-ldm.yaml
```

The loader knobs are shared across that list, so a detector needing its own
`batch_size` belongs in its own run config. Nothing is lost by splitting it:
results are keyed by `(detector, dataset)`, not by run, so separate runs writing
into the same `results/` still compare in one table.

## The detector zoo

Four detectors are configured beyond the generic Hub adapter. All three
published ones are frozen, fake-positive, and thresholded at logit zero, which
is exactly what `FrozenDetector.forward` is defined to return -- so no sign
flips or probability conversions sit between them and the metrics.

| config | model | backbone | input | licence |
|---|---|---|---|---|
| `bfree` | B-Free (CVPR 2025) | DINOv2 ViT-B/14 + 4 registers | native res, 5 token-space crops @504 | non-commercial |
| `gapl` | GAPL | CLIP ViT-L/14 + LoRA, 64 prototypes | 224 centre crop, ImageNet stats | MIT (declared) |
| `rine-ldm` | RINE (ECCV 2024), diffusion-trained | frozen CLIP ViT-L/14, all 24 blocks | 224 centre crop, CLIP stats | Apache-2.0 |
| `rine-4class` | RINE, ProGAN-trained | same | same | Apache-2.0 |

None of the three upstream repos is pip-installable, so they are cloned by hand
under `third_party/` and put on `sys.path` for one import
(`pipeline/detectors/_vendor.py`). `third_party/` and `checkpoints/` are
gitignored: a clone of this repo cannot rebuild the zoo on its own, so record
the upstream SHAs in `_vendor.py`'s `REPOS` once you have cloned, and a drifting
checkout warns rather than passing silently.

```bash
pip install -e ".[dev,zoo]"    # timm, peft, and OpenAI CLIP from git

git clone https://github.com/grip-unina/B-Free.git third_party/B-Free
git clone https://github.com/UltraCapture/GAPL.git third_party/GAPL
git clone https://github.com/mever-team/rine.git   third_party/rine
```

Weights:

- **B-Free** -- unzip
  [`BFREE_dino2reg4.zip`](https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip)
  (~330 MB, md5 `f3f53fa647848b16cf81c913f148a198`) into `checkpoints/bfree/`,
  giving `checkpoints/bfree/BFREE_dino2reg4/{config.yaml,model_epoch_best.pth}`.
  It is served from a single university host with no mirror; keep a private copy
  of the zip once it downloads.
- **GAPL** -- `hf download AbyssLumine/GAPL checkpoint.pt --local-dir checkpoints/gapl`
  (1.22 GB). Take the prototypes from the clone, not the Hub: the Hub's
  `stage1.pt` is 6 KB and looks truncated next to the repo's 526 KB copy.
- **RINE** -- already in the clone at `third_party/rine/ckpt/`. Only the frozen
  CLIP ViT-L/14 downloads, into `~/.cache/clip` on first use.

Then:

```bash
python scripts/run_eval.py --config configs/runs/zoo_on_ntire.yaml
python scripts/run_eval.py --config configs/runs/zoo_bfree_on_ntire.yaml
python scripts/report.py --results results/
```

**B-Free is a separate run because of `batch_size`.** It takes the image at
native resolution -- the 504x504 cropping happens inside the network, in token
space -- so the tensors in a batch are not the same size and cannot stack.
`batch_size: 1` is a correctness requirement for it, not a memory knob.

Two upstream loading habits are worked around in the adapters rather than
patched into the clones, since an edited clone is lost on the next pull: GAPL
resolves CLIP from a hardcoded absolute path under `/root/.cache`, and RINE
assigns weights through a string-manipulating `exec()` that cannot fail loudly.
Both repos also load with `strict=False`, which leaves an untrained head in
place and reads as a weak detector rather than an error -- so both adapters
check the key sets and raise. If a zoo member scores near 0.5 AUC on clean
images, that check is the first thing to re-read.

**Check parity against upstream before trusting a sweep.** A wrong
normalization constant or a flipped polarity produces numbers that look
entirely reasonable, and no amount of downstream aggregation will reveal it.
`predict.py` is the tool -- it scores a directory with the same adapter the
harness uses:

```bash
python scripts/predict.py --image-dir third_party/B-Free/demo_images \
    --detector configs/detectors/bfree.yaml --batch-size 1 --out bfree_demo.json
```

B-Free ships expected outputs in `demo_images/results.csv`, so that one is a
direct comparison: `pred` here is `sigmoid` of the logit in their file. For RINE
and GAPL, run their own demo script (`demo/demo.py`; `inference_image.py`, *not*
the `inference.py` their README names) on the same images and compare. In every
case the fakes must score above the reals -- a detector that has been polarity
-flipped reports a plausible `1 - AUC` rather than an error.

## Usage

```bash
python scripts/build_manifest.py --config configs/datasets/<name>.yaml
python scripts/run_eval.py --config configs/runs/<name>.yaml   # -> results/*.json
python scripts/report.py --results results/                    # -> tables + figures

python scripts/predict.py --image-dir path/to/images \
    --detector configs/detectors/<name>.yaml --out preds.json
```

`preds.json` is a list of `{"image_path": str, "pred": float}` where `pred` is
P(AI-generated).

`run_eval.py` writes one file per dataset, `results/{run_id}__{detector}__{dataset}.json`.

`max_images` in the run config is the knob that moves wall-clock by an order of
magnitude: a full run is `1 + 19 + 2 x n_replicates` conditions over the eval
subset. Use a small value for the first loop. `--levels` and `--transforms`
override the config for a smoke run without editing it -- `--levels 0 1` scores
clean plus the single-transform grid and nothing else (L0 is always added back
if you leave it out, since it sets the threshold and the retention
denominator).

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
