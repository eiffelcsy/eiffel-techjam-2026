# robust-aigc-eval

Evaluation harness for AI-generated image detectors under everyday image
handling, plus a standalone inference script.

Model- and dataset-agnostic: no model id, dataset id or device appears anywhere
in the code. Detectors and dataset sources are named in config by dotted import
path, so pointing the harness at something new is a config file, never a patch.

> A detector scores 0.97 AUC on pristine images. What does it score after the
> image has been posted, thumbnailed, filtered, or screenshotted?

## Why this design

- **Clean AUC is not the number you deploy against.** Detectors lean on local,
  high-frequency traces left by the generator's upsampling stack -- exactly the
  signal quantisation, low-pass filtering and resampling destroy first [1].
- **The transforms are ordinary image handling, not adversarial attack.** No
  attacker is required: re-encoding for a feed already erases the evidence.
- **L1 is a one-factor-at-a-time sweep** -- the field's standard robustness
  protocol, so these numbers stay comparable to published tables and every
  result is attributable to exactly one cause.
- **L2/L3 compose**, because single-perturbation testing is the known gap [2].
  Real propagation stacks operations: re-compress, resize, re-upload.
- **Retention is chance-corrected** -- `(auc_deg - 0.5) / (auc_clean - 0.5)` --
  because 0.5 is the floor, not zero. A raw ratio would score a collapsed
  detector at ~0.5, reading as "half retained" when none of it is.
- **The error split is reported separately from AUC**, because real and
  generated images demonstrably do not respond to the same perturbation the same
  way [3]. Which *direction* a transform pushes a given detector is this
  harness's hypothesis, tested by the FPR/FNR table -- not an established result.

## Install

```bash
pip install -e ".[dev]"          # add ",zoo" for the vendored detectors
```

`pyproject.toml` is the single source of truth for dependencies -- there is no
`requirements.txt`. Python >= 3.10; runs on CUDA, MPS or CPU with no config
change (`device: auto`).

On a GPU node, install torch first -- its CUDA wheel index is cluster-specific
and cannot be expressed portably in the project file:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

Set `HF_HOME` (and `HF_DATASETS_CACHE`) if `$HOME` is small or read-only.

## Usage

```bash
python scripts/build_manifest.py --config configs/datasets/<name>.yaml
python scripts/run_eval.py       --config configs/runs/<name>.yaml   # -> results/*.json
python scripts/report.py         --results results/                  # -> summary.md + figures

python scripts/predict.py --image-dir path/to/images \
    --detector configs/detectors/<name>.yaml --out preds.json
```

`run_eval.py` writes one file per (detector, dataset):
`results/{run_id}__{detector}__{dataset}.json`. `preds.json` is a list of
`{"image_path": str, "pred": float}` where `pred` is P(AI-generated).

`max_images` in the run config moves wall-clock by an order of magnitude: a full
run is `1 + 19 + 2 x n_replicates` conditions over the eval subset. `--levels`
and `--transforms` override the config for a smoke run without editing it
(`--levels 0 1` scores clean plus the single-transform grid; L0 is always added
back, since it sets the threshold and the retention denominator).

## The transform grid

Eleven transforms, each a real-world artefact with an explicit parameter list.

| Transform | Group | Parameter | Values | Real-world analog |
|---|---|---|---|---|
| JPEG compression | compression | quality | 90, 70, 50, 30 | Social re-encode, messaging |
| Gaussian blur | blur | sigma | 0.5, 1.0, 2.0 | Out-of-focus |
| Resize | resampling | scale | 0.5x, 0.25x then upscale | Thumbnail generation |
| Gaussian noise | noise | sigma | 0.02, 0.05, 0.10 | Low-light sensor noise |
| Brightness up / down | photometric | factor | 1.2 / 0.8 | Filter apps, auto-enhance |
| Contrast up / down | photometric | factor | 1.2 / 0.8 | Filter apps, auto-enhance |
| Saturation up / down | photometric | factor | 1.2 / 0.8 | Filter apps, auto-enhance |
| Center crop | framing | keep | 80% | Profile-picture framing |

The six `photometric` entries are colour jitter unbundled: +/-20% on one
attribute at a time, so a detector that only breaks on a brightness lift is
visible as such rather than averaged into one stochastic "jitter" number. They
share a group, so the report can aggregate them back.

Parameters apply identically at every resolution -- some are resolution-relative
(`scale`, `keep`), some absolute (`sigma` in pixels), which is a property of the
artefacts themselves. Degradation happens at the image's native size, *before* a
detector's own preprocessing, so the degraded eval set is identical whichever
detector scores it.

The grid lives in `configs/degradations.yaml`.

## The four composition levels

| Level | What | How |
|---|---|---|
| L0 | clean | reference: sets the threshold and the retention denominator |
| L1 | single | the 19 grid conditions, one at a time (exhaustive sweep) |
| L2 | pair | 2 distinct transforms, params from the grid, random order |
| L3 | multi | 3-5 distinct transforms, sampled the same way |

L2/L3 are Monte-Carlo samples of a composition space too large to enumerate,
drawn per image and keyed on `(index, level, replicate)` -- so the degraded set
is identical across runs and detectors. Transforms are drawn without replacement
and applied in shuffled order, because they do not commute. `n_replicates`
re-draws L2/L3 over the same images: more coverage and a tighter CI without
growing the eval set.

## What comes out

1. **Headline** -- detectors x levels: clean, L1, L2, L3 AUC and retention.
2. **Interaction** -- measured L2/L3 retention against what the L1 marginals
   predict under independence. A large negative gap is the finding: the
   single-transform benchmark overstated that detector's robustness.
3. **By transform (L1)** -- detectors x transforms, plus degradation curves
   (AUC vs parameter, one panel per transform).
4. **Error split** -- FPR and FNR at a threshold fixed on clean data.

Plus a worst-recipe table: which L2/L3 combinations did the damage.
`summary.operating_envelope` is the deepest level still at or above
`retention_floor` (default 0.5 -- a reporting convention, not a derived
quantity).

## Configuration

Three kinds, one file each, one directory each. A dataset and a detector each
name their component the same way -- a dotted `target` plus its constructor
`args` -- and a run references the other two by path.

```
configs/datasets/<name>.yaml    what a dataset IS
configs/detectors/<name>.yaml   what a detector IS
configs/runs/<name>.yaml        detectors x datasets, plus how the run executes
```

`configs/defaults.yaml` is the annotated reference -- every key of every kind
with its default. The harness never loads it. Every shipped config carries its
own rationale in comments; this section is the shape, not the detail.

**A detector.** Any Hub `AutoModelForImageClassification` checkpoint works
through the generic adapter:

```yaml
name: my-detector                 # display name, used in result filenames
target: pipeline.detectors.hf.HFImageClassifier
args:
  model_id: some-org/some-detector
  # fake_labels: [artificial]     # only if the head's id2label is unusual
device: auto
```

Which output index means *generated* is resolved from the model's own
`config.id2label`, never assumed -- a fair number of Hub detectors put the fake
class at index 0, and guessing wrong yields a plausible-looking `1 - AUC`. An
ambiguous label map raises.

For anything the generic adapter cannot cover, subclass `FrozenDetector` and
point `target:` at it. Two contract details matter: `forward` returns a `(B,)`
fake-minus-real logit (higher = generated), and if your class carries real
weights, override `preprocess_fn()` to return a callable closing over only what
preprocessing needs -- the Dataset is forked into DataLoader workers, so a
preprocessing function holding the model pickles every parameter (which fails
outright on MPS and CUDA). `pipeline/detectors/hf.py` shows the pattern in about
six lines.

**A dataset.**

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
    limit: 2000                   # per class
    streaming: true               # required for large repos
```

Once the manifest exists, evaluation reads `manifest` and `split` and ignores
`source` entirely. The two `split` keys are not the same knob: the inner one
picks which split to *pull*, the outer one which split to *score*.

Four sources ship: `HFImageDatasetSource`, `ImageDirSource`,
`CsvMetadataSource` (for benchmarks shipped as archives plus a metadata table --
it references images in place and never re-encodes, and its `path_prefix` knob
subsets by path when no column can), and `ConcatSource` (several sources into
one manifest, each child given its own image subdirectory so two splits cannot
overwrite each other's files).

Things that bite:

- **`../data/`, not `data/`.** `data/` sits at the **repo root**, above both
  packages, because `../grace_adapter`'s scripts read the same dataset configs
  from their own working directory. A manifest path inside one package resolves
  for one caller and silently `FileNotFoundError`s for the other.
- **Label polarity is declared, never inferred** -- public AIGC datasets
  disagree about whether `0` means real. An unnamed integer label column is
  `fake_classes: ["1"]`.
- **On >2 classes, name `real_classes` explicitly.** Left unset, everything not
  listed as fake folds into *real* -- right for a binary dataset, silently wrong
  otherwise.
- **`streaming: true` is not cosmetic on a large repo.** A plain `load_dataset`
  materializes every shard before yielding a row, `limit` or no `limit`.
- **`on_missing: error` is the default** for the CSV source: an archive unpacked
  one level deeper than expected otherwise builds a benchmark quietly missing
  most of its images.

**A run.**

```yaml
run_id: my_run
detector: configs/detectors/my-detector.yaml   # path, or the mapping inline
datasets:
  - configs/datasets/my-dataset.yaml           # paths, or mappings inline

degrade: {}         # `{}` takes every default: four levels, eleven transforms,
                    # n_replicates: 3, seed: 0
```

Loader knobs (`batch_size`, `num_workers`, `max_images`) live here rather than
on the dataset: they are properties of this machine and this run's scope, not of
the dataset. Write `detectors:` with a list to score several in one command --
each is loaded, run against every dataset, and released before the next is
built, so a zoo costs one detector's memory. The loader knobs are shared across
that list, so a detector needing its own `batch_size` gets its own run config;
nothing is lost, since results key on `(detector, dataset)` rather than on run.

## What is configured here

**Datasets** -- see `../docs/DATASETS.md` for the download and unpack recipe.

| config | what | rows |
|---|---|---|
| `ntire_train` | NTIRE 2026 train, all six shards, `split: train` | 277,643 |
| `ntire_val` | NTIRE val, `split: undistorted` | 5000 |
| `ntire_val_distorted` | same manifest, `split: distorted` -- the challenge's own degradations | 5000 |
| `ntire_val_hard` | NTIRE hard subset: adversarial embedding, neural codec, recompression chains | 2500 |
| `wildfake_coco_dalle3` | COCO val2017 real + DALL-E 3 Advanced fake | 4998 / 8843 |
| `ntire_train_eval` | **superseded** -- selects zero rows, kept so the change is loud | 0 |

`ntire_val_distorted` and `ntire_val_hard` are **level-0-only** evaluations: our
grid stacked on top of an unknown prior transform makes both the L0 reference
and L1's one-cause claim untrue.

**Detectors:**

| config | model | backbone | input | licence |
|---|---|---|---|---|
| `sdxl` | Organika/sdxl-detector | Swin | processor default | via the generic Hub adapter |
| `bfree` | B-Free (CVPR 2025) | DINOv2 ViT-B/14 + 4 registers | native res, 5 token-space crops @504 | non-commercial |
| `gapl` | GAPL | CLIP ViT-L/14 + LoRA, 64 prototypes | 224 centre crop, ImageNet stats | MIT (declared) |
| `rine-ldm` | RINE (ECCV 2024), diffusion-trained | frozen CLIP ViT-L/14, all 24 blocks | 224 centre crop, CLIP stats | Apache-2.0 |
| `rine-4class` | RINE, ProGAN-trained | same | same | Apache-2.0 |
| `dinov3-ntire` | built here -- see below | frozen DINOv3 ViT-S/16 + MLP probe | 224 resize | -- |
| `dinov3-ntire-crop` | same, `input_mode: crop` | same | 224 centre crop at native res | -- |

The published detectors are frozen, fake-positive, and thresholded at logit
zero -- exactly what `FrozenDetector.forward` is defined to return, so no sign
flips or probability conversions sit between them and the metrics.

**Runs:** `zoo_on_ntire` (gapl + both RINEs), `zoo_bfree_on_ntire` (B-Free alone,
see below), `sdxl_on_ntire`, and the GRACE proof-of-concept trio —
`dinov3_poc_baseline`, `dinov3_poc_baseline_crop` (the preprocessing ablation)
and `dinov3_poc_grace`. The two PoC baselines and the GRACE arms all score
**WildFake**: NTIRE val is a selection set for that path, and
`grace_adapter/scripts/compare.py` refuses two result files from different eval
sets because retention is only comparable on one.

### The DINOv3 probe detector

`pipeline/detectors/dinov3.py` is the one detector here that is not somebody
else's published checkpoint: a frozen DINOv3 ViT-S/16 trunk plus an MLP probe
head, fit on clean images by `../grace_adapter/scripts/train_probe.py`. It
exists so GRACE has a detector whose trunk/head seam is a *construction* rather
than a reconstruction of a vendored repo's internals. The backbone is a
licence-gated Hub repo; `backbone_id` takes any mirror.

`input_mode` (`INPUT_MODES` in that module) is the knob worth knowing about:

- `resize` (default) -- squash the whole image to 224x224, the processor's own
  transform.
- `crop` -- a centre 224x224 window at the source's native resolution.

On the previous dataset (SID_Set), `resize` produced a probe at 0.9999 val AUC
that still held 0.985 AUC after a 32x32 round trip -- a resolution at which no
generation trace survives. That head had learned *content*, not forensics, and a
detector that never collapses cannot demonstrate a repair. `crop` is the fix,
and it is a fix to preprocessing rather than to the head: the traces live at the
pixel scale, so the trunk has to be shown pixels at that scale. The two are
separate detector configs, not a flag on one, because their heads are not
interchangeable -- the trunk sees a different image scale in each mode. **That
finding has not been reproduced on NTIRE**; the mechanism is a property of the
preprocessing rather than of that dataset, so the arm is still the right one to
have, but the numbers are history.

The 32x32 round trip is worth keeping as a standing check on any head fit here:
one reading generation traces must fall toward chance under it.

## The detector zoo -- setup

None of the three upstream repos is pip-installable, so they are cloned by hand
under `third_party/` and put on `sys.path` for one import
(`pipeline/detectors/_vendor.py`). `third_party/` and `checkpoints/` are
gitignored: a clone of this repo cannot rebuild the zoo on its own, so record
the upstream SHAs in `_vendor.py`'s `REPOS` once you have cloned -- a drifting
checkout then warns rather than passing silently.

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
  Single university host, no mirror -- keep a private copy of the zip.
- **GAPL** -- `hf download AbyssLumine/GAPL checkpoint.pt --local-dir checkpoints/gapl`
  (1.22 GB). Take the prototypes from the clone, not the Hub: the Hub's
  `stage1.pt` is 6 KB and looks truncated next to the repo's 526 KB copy.
- **RINE** -- already in the clone at `third_party/rine/ckpt/`. Only the frozen
  CLIP ViT-L/14 downloads, into `~/.cache/clip` on first use.

```bash
python scripts/run_eval.py --config configs/runs/zoo_on_ntire.yaml
python scripts/run_eval.py --config configs/runs/zoo_bfree_on_ntire.yaml
python scripts/report.py --results results/
```

**B-Free is a separate run because of `batch_size`.** It takes the image at
native resolution -- the 504x504 cropping happens inside the network, in token
space -- so the tensors in a batch cannot stack. `batch_size: 1` is a
correctness requirement, not a memory knob.

Two upstream loading habits are worked around in the adapters rather than
patched into the clones (an edited clone is lost on the next pull): GAPL
resolves CLIP from a hardcoded absolute path under `/root/.cache`, and RINE
assigns weights through a string-manipulating `exec()` that cannot fail loudly.
Both repos also load with `strict=False`, which leaves an untrained head in
place and reads as a weak detector rather than an error -- so both adapters
check the key sets and raise. **If a zoo member scores near 0.5 AUC on clean
images, that check is the first thing to re-read.**

**Check parity against upstream before trusting a sweep.** A wrong normalization
constant or a flipped polarity produces entirely reasonable-looking numbers, and
no downstream aggregation will reveal it. `predict.py` scores a directory with
the same adapter the harness uses:

```bash
python scripts/predict.py --image-dir third_party/B-Free/demo_images \
    --detector configs/detectors/bfree.yaml --batch-size 1 --out bfree_demo.json
```

B-Free ships expected outputs in `demo_images/results.csv` (`pred` here is
`sigmoid` of the logit in their file). For RINE and GAPL, run their own demo
script (`demo/demo.py`; `inference_image.py`, *not* the `inference.py` their
README names) on the same images. In every case the fakes must score above the
reals.

## Invariants

- **Format neutrality.** Every image decodes through one identical path before a
  detector sees it, and sources write a single on-disk format -- no container
  statistic is left correlated with the label.
- **Fixed eval subset.** Sampled deterministically from a seed, so it is
  identical across runs without a cache file.
- **Paired conditions.** Every condition scores the same images in the same
  order, so clean and degraded scores compare per image.
- **Deterministic sampling.** `stable_seed(index, level, replicate)`, blake2b
  and never the builtin `hash()` (salted per process) -- every detector sees the
  identical degraded set, so a difference between detectors is never a
  difference in the draw.
- **Recipes are logged**, so the composed levels can be analysed after the fact
  rather than just scored.
- **Frozen detectors.** Loaded, evaluated, never trained.

## Layout

```
../data/           MATERIALIZED DATASETS -- at the repo root, shared with
                   ../grace_adapter: a dataset config's `../data/...` resolves
                   the same from either package's working directory
configs/           datasets/ detectors/ runs/ + degradations.yaml, defaults.yaml
pipeline/
  config.py        yaml -> dataclasses
  data/            manifest (the one table everything keys off) + sources + datasets
  degrade/         ops.py (the eleven transforms) + conditions.py (levels, sampling)
  detectors/       frozen-detector contract, the generic Hub adapter, the zoo
  eval/            metrics, runner, report
  inference/       image dir -> P(generated) per image
  utils/           seeding, io, import-path resolution
scripts/           build_manifest / run_eval / report / predict
tests/             26 tests, incl. one end-to-end run on synthetic data
third_party/       the zoo's upstream repos, cloned by hand (gitignored)
checkpoints/       the zoo's weights, downloaded by hand (gitignored)
```

## References

1. Wang, Wang, Zhang, Owens, Efros. *CNN-Generated Images Are Surprisingly Easy
   to Spot... for Now.* CVPR 2020.
   <https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html>
2. Cui et al. *GlobalForge: Towards Robust AI-Generated Image Detection.*
   arXiv:2607.14684 -- introduces RealDeg-Bench, whose seven-operator pool and
   depth range N in {1..5} independently match this harness's L2/L3 design.
   <https://arxiv.org/abs/2607.14684>
3. Wang et al. *RA-Det: Towards Universal Detection of AI-Generated Images via
   Robustness Asymmetry.* arXiv:2603.01544.
   <https://arxiv.org/abs/2603.01544>
