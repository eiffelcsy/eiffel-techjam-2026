# GRACE: Gated Residual Adapter for Clean-feature Estimation

**Restoring AI-generated-image detectors under everyday image degradation, without fine-tuning them.**

A frozen AIGC detector scores 0.96 AUC on pristine images. What does it score
after the image has been JPEG'd for a feed, thumbnailed, filtered, or
screenshotted? This repository contains (i) a model- and dataset-agnostic
harness that *measures* that collapse, and (ii) a ~0.4M-parameter adapter,
trained **without labels**, that repairs it at the detector's own trunk/head
seam.

```
baseline  logit = head( trunk(x) )
GRACE     logit = head( adapter(trunk(x)) )                          label-free
GRACE-D   logit = head( adapter(trunk(x)) ) + b * aux(D, severity)   + labels
                                              D = adapter(f_deg) - f_deg
```

The detector is never fine-tuned, never re-trained, and never even loaded
differently. The adapted model re-enters the evaluation harness as an ordinary
`FrozenDetector`, so the baseline and every adapted arm come out of the same
code path, the same degraded images, and the same result schema.

---

## 1. Overview

### The problem

Detectors of generated images lean on **local high-frequency traces** — the
pixel statistics a generator's upsampling stack leaves behind. Quantisation,
low-pass filtering and resampling destroy exactly those, and they are the first
three things that happen to an image on its way through a social platform. The
number that collapses is chance-corrected **retention**:

```
retention = (auc_degraded - 0.5) / (auc_clean - 0.5)
```

*Of the detector's skill above chance on clean images, what fraction survives?*

### The two claims

**GRACE — the evidence is displaced, not destroyed.** The degraded features
still carry the discriminative signal; it has moved somewhere the frozen head no
longer looks. If so, a tiny residual correction at the seam recovers a real
slice of the retention gap. The parameter budget is a *design constraint*, not a
convenience: if a large adapter turns out to be needed, the claim is false and
the result is far less interesting.

**GRACE-D — the damage is itself a signal.** Generated images drift further in
embedding space under perturbation than real ones do (RA-Det). An adapter
trained purely to *erase* drift is therefore destroying forensic evidence while
its reconstruction loss falls — asymmetrically. So keep the quantity the adapter
already computes: Δ, its own estimate of the drift, available at test time
without ever seeing the clean image. This also breaks the restoration ceiling —
a perfect restorer can at best recover the clean-image score (retention 1.0), but
*how much damage the image took* is information the clean image never contained.

|  | Adapter | Aux head | Labels | Ceiling |
|---|---|---|---|---|
| **GRACE** | trained | — | none | retention ≤ 1.0 |
| **GRACE-D** | *bit-identical, frozen* | trained | yes | may exceed 1.0 |

Stage 2 never touches the adapter, so both variants ship the same weights and
"the adapter is trained without labels" stays literally true. That separation is
what makes the erasure question (E4) testable at all.

### Three design decisions worth knowing

1. **The training loop contains no trunk forward.** The trunk is frozen and a
   clean image never changes, so clean features are constant — cache them. The
   same holds for the *degraded* side: every recipe is drawn from
   `stable_seed(index, level, replicate, seed)`, a blake2b hash rather than a
   running RNG counter, so a degraded view is a pure function of (image,
   condition) and **epoch == replicate**. Epoch 7's degradation of image 412 is
   computable now, without having run epochs 0–6 — which is what makes rendering
   every epoch offline possible. One training step is then two memmap reads and
   a 2-layer MLP. See [grace/cache/schedule.py](grace_adapter/grace/cache/schedule.py).

2. **The objective spends capacity where it changes the decision.** Plain MSE
   treats every feature direction as equally worth fixing; the head does not — it
   collapses the vector into one scalar, so only error inside its sensitive
   subspace can move AUC. The error is weighted by the head's Jacobian, written
   as a blend so that `eps = 1` is *exactly* `F.mse_loss` and the plain-MSE
   ablation is one config key. See [grace/train/weighting.py](grace_adapter/grace/train/weighting.py)
   and [grace/train/losses.py](grace_adapter/grace/train/losses.py).

3. **Identity at initialization, exactly.** The last projection of every adapter
   block is zero-initialised, so an untrained adapter returns its input
   bit-for-bit whatever the gate or the severity conditioning happen
   to be — and `beta = 0` makes GRACE-D *identical* to GRACE at init. Without
   this, any change in clean AUC would be unattributable.

### The evaluation protocol

| Level | What | Conditions |
|---|---|---|
| **L0** | clean — the reference every retention divides by | 1 |
| **L1** | one transform at a time (OFAT, the field's standard protocol) | 19 grid points over 11 transforms |
| **L2** | 2 composed transforms, random order | 3 replicate draws |
| **L3** | 3–5 composed transforms | 3 replicate draws |

Eleven transforms — JPEG, Gaussian blur, resize, Gaussian noise, brightness ±,
contrast ±, saturation ±, centre crop — each an ordinary image-handling artefact
rather than an adversarial attack. Degradation is applied at native resolution
**before** any detector's own preprocessing, and every recipe is deterministic in
`(index, level, replicate)`, so every detector and every arm sees byte-identical
degraded images. A difference between two rows is a difference between models,
never a difference in the draw. Grid: [configs/degradations.yaml](eval_pipeline/configs/degradations.yaml).

---

## 2. Repository layout

```
eval_pipeline/     the measurement harness — model- and dataset-agnostic
  pipeline/          config, manifests, sources, the degradation grid,
                     the frozen-detector contract, metrics, runner, report
  configs/           datasets/ detectors/ runs/ + degradations.yaml
  scripts/           build_manifest . run_eval . report . predict

grace_adapter/     the method
  grace/
    splits/          the trunk/head seam, added AROUND a detector; verified at
                     construction (head(trunk(x)) == detector(x))
    cache/           schedule (index,epoch)->degradation . writer . memmap reader
    models/          GatedResidualAdapter . DiscrepancyHead . severity . factory
    train/           Jacobian weighting . losses . diagnostics . EMA . loops
    detectors/       AdaptedDetector — re-enters the harness as a FrozenDetector
    probe/           stage 0: fit the PoC detector's own head
  configs/           probe/ cache/ train/ detectors/
  scripts/           train_probe . build_cache . analyze_drift . train_adapter .
                     train_discrepancy . compare . sweep_beta . seed_stats .
                     run_all.sh — every experiment in one command

data/              materialized datasets (gitignored) — at the REPO ROOT, because
                   both packages read it and a config's `../data/...` must resolve
                   the same from either working directory
docs/              DATASETS.md . PIPELINE.md . EXPERIMENTS.md . RESULTS.md
```

Detectors and datasets are defined in **exactly one place**
([eval_pipeline/configs/](eval_pipeline/configs/)); `grace_adapter` references
them by path so the adapted model cannot drift from what was benchmarked.

**Documentation map.** [grace_adapter/README.md](grace_adapter/README.md) argues
*why* the method is shaped the way it is; [docs/PIPELINE.md](docs/PIPELINE.md)
walks the code component by component; [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
is the operator's document (every arm, its prerequisites, exact commands, and
how to read each artifact); [docs/DATASETS.md](docs/DATASETS.md) is the download
and unpack recipe; [eval_pipeline/README.md](eval_pipeline/README.md) documents
the harness and the detector zoo.

---

## 3. Setup

### 3.1 Environment

Python >= 3.10. Runs on CUDA, MPS or CPU with no config change (`device: auto`).
`pyproject.toml` is the single source of truth for dependencies — there is no
`requirements.txt`.

```bash
git clone <this-repo> && cd eiffel-techjam-2026
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# On a GPU node, install torch FIRST: its CUDA wheel index is cluster-specific
# and cannot be expressed portably in a project file.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -e "./eval_pipeline[dev]"        # the harness
pip install -e "./grace_adapter[dev,poc]"    # the method + the DINOv3 PoC detector
```

Optional extras:

```bash
pip install -e "./eval_pipeline[dev,zoo]"    # published detectors (timm, peft, CLIP)
pip install -e "./grace_adapter[wandb]"      # tracking; a null object when absent
```

Set `HF_HOME` (and `HF_DATASETS_CACHE`) if `$HOME` is small or read-only.

### 3.2 Backbone access

The PoC trunk `facebook/dinov3-vits16-pretrain-lvd1689m` is a **licence-gated**
Hub repo. Accept the licence on the model page and run `hf auth login` once — or
point `backbone_id` in
[configs/detectors/dinov3-ntire.yaml](eval_pipeline/configs/detectors/dinov3-ntire.yaml)
at a mirror you already have. Nothing in the code assumes the official id.

### 3.3 Data

Full recipe in [docs/DATASETS.md](docs/DATASETS.md). Two corpora, three jobs:

| corpus | role | disk |
|---|---|---|
| NTIRE 2026 train | stage 0 (head fit) and stage 1 (adapter fit) | ~230 GB |
| NTIRE 2026 val + val_hard | stage-0 epoch selection, stage-1 held-out-image validation | included above |
| WildFake (COCO val2017 + DALL-E 3) | **the eval set** — held out from everything above | ~28 GB of archives |

```bash
# NTIRE (Hugging Face)
hf download deepfakesMSU/NTIRE-RobustAIGenDetection-train \
  --repo-type dataset --local-dir /tmp/ntire_train_dl
hf download deepfakesMSU/NTIRE-RobustAIGenDetection-val \
  --repo-type dataset --local-dir /tmp/ntire_val_dl \
  --include "val_images.zip" "val_images_hard.zip" "val_labels.csv" "val_hard_labels.csv"

# WildFake (ModelScope — not on the Hub)
pip install modelscope
modelscope download --dataset hy2628982280/WildFake \
    split_train_test/csv_file/total_split/test_metadata.csv \
    split_train_test/csv_file/total_split/train_metadata.csv \
    Images/Real/coco.zip \
    Images/Diffusion_based/DALLE.zip \
    --local_dir data/wildfake
```

Unpack per [docs/DATASETS.md](docs/DATASETS.md) sections 2–3, then build the
manifests:

```bash
cd eval_pipeline
for d in ntire_train ntire_val ntire_val_hard wildfake_coco_dalle3; do
  python scripts/build_manifest.py --config configs/datasets/$d.yaml
done
```

Expected row counts — **a mismatch means the unpack nesting is wrong**, and
`on_missing: error` is the default precisely so that fails loudly rather than
building a benchmark quietly missing most of its images:

| manifest | rows |
|---|---|
| `ntire_train` | 277,643 |
| `ntire_val` | 5000 undistorted + 5000 distorted |
| `ntire_val_hard` | 2500 |
| `wildfake` | 13,841 (4998 real / 8843 fake) |

> **Never rebuild a manifest a feature cache was rendered against.** The cache
> stores features by row position and fingerprints the manifest (`manifest_sha`)
> to prove the rows still line up.

---

## 4. Reproducing the results

### 4.1 The whole path in one command

```bash
cd grace_adapter
bash scripts/run_all.sh                # every experiment, in order
bash scripts/run_all.sh --list         # print the 22 steps and exit
bash scripts/run_all.sh --smoke        # 2 epochs, 2 cache views -- proves wiring
bash scripts/run_all.sh --from 8       # resume at step 8
bash scripts/run_all.sh --skip-slow    # everything except E6, the live control
WANDB=1 bash scripts/run_all.sh        # every stage tracked under one group
```

Twenty-two steps covering **every** experiment — D1, E0 through E9, the seed
floor, and both prerequisites — and every one of them idempotent: existing
manifests are skipped rather than rebuilt, the cache resumes at view granularity,
a training run is skipped if its `summary.json` exists, and a harness run is
skipped if its result JSON exists. An interrupted run is resumed by re-running
the same command.

The order is the argument, not a convenience: the first seven steps train nothing
GRACE touches, and each of them can end the project before it costs anything —
no collapse (P2), a content shortcut (D1), a wrong seam (E1), or a ceiling of
zero (E0). Step 5 is a **hard stop**: `compare.py --assert-identity` exits
non-zero unless the null adapter reproduces the baseline exactly.
[scripts/run_all.sh](grace_adapter/scripts/run_all.sh) prints what to read, in
order, when it finishes.

### 4.2 Stage by stage

```bash
# -- P1 . stage 0: the ONE place a detector is trained -----------------------
#    Frozen trunk + a ~400k-param MLP head, fit on CLEAN images only.
cd grace_adapter
python scripts/train_probe.py configs/probe/dinov3_ntire.yaml

# -- P2 . the baseline retention curve GRACE is measured against -------------
#    Run BEFORE training an adapter: if retention does not collapse here, there
#    is no gap to close, at ~8% of what the stage-1 cache costs.
cd ../eval_pipeline
python scripts/run_eval.py --config configs/runs/dinov3_poc_baseline.yaml

# -- P3 . render the feature caches (resumable at shard granularity) ---------
#    One decode per image, every view of it built from that decode in the workers.
cd ../grace_adapter
python scripts/build_cache.py configs/cache/dinov3.yaml --dry-run    # ALWAYS first
python scripts/build_cache.py configs/cache/dinov3.yaml              # ~6.3 GB
python scripts/build_cache.py configs/cache/dinov3_val.yaml          # ~115 MB
python scripts/build_cache.py configs/cache/dinov3_val_hard.yaml     # ~58 MB

# -- E0 . does the drift asymmetry hold on this data? (minutes, no GPU) ------
python scripts/analyze_drift.py \
  --cache cache/dinov3-ntire \
  --dataset  ../eval_pipeline/configs/datasets/ntire_train.yaml \
  --detector ../eval_pipeline/configs/detectors/dinov3-ntire.yaml \
  --split    grace.splits.dinov3.DINOv3Split \
  --out      results/dinov3_poc_drift.json

# -- stage 1 . the label-free adapter (arm B) and its control (arm A) --------
python scripts/train_adapter.py configs/train/dinov3_clean.yaml       # minutes
python scripts/train_adapter.py configs/train/dinov3_degraded.yaml    # minutes

# -- stage 2 . the supervised discrepancy head, adapter frozen ---------------
python scripts/train_discrepancy.py configs/train/dinov3_discrepancy.yaml  # seconds

# -- score all three arms through the SAME harness as the baseline ----------
cd ../eval_pipeline
python scripts/run_eval.py --config configs/runs/dinov3_poc_grace.yaml
python scripts/report.py --results results/          # -> summary.md + figures
```

### 4.3 Reading the numbers

```bash
cd grace_adapter
R=../eval_pipeline/results
B=$R/dinov3_poc_baseline__dinov3-ntire__wildfake-coco-dalle3.json

python scripts/compare.py --baseline $B --adapted $R/dinov3_poc_grace__dinov3+identity__wildfake-coco-dalle3.json
python scripts/compare.py --baseline $B --adapted $R/dinov3_poc_grace__dinov3+grace__wildfake-coco-dalle3.json
python scripts/compare.py --baseline $B --adapted $R/dinov3_poc_grace__dinov3+grace-d__wildfake-coco-dalle3.json
```

Three things govern every number in this project:

- **Use [compare.py](grace_adapter/scripts/compare.py) for every
  GRACE-versus-baseline claim.** It normalises retention by the **baseline's**
  clean AUC. The harness's own convention divides by each detector's own clean
  AUC, which for GRACE-D would shrink the very improvement being claimed —
  GRACE-D's clean AUC is roughly the baseline's (Δ is ~0 on a clean image) while
  its degraded AUC can go *up*. `> 1.0` then means something concrete: the
  adapted detector on a degraded image beating the original detector on a clean
  one. It refuses two result files from different eval sets.
- **WildFake is the reported benchmark; NTIRE val is a selection set.** Stage 0
  selects its epoch on NTIRE val + val_hard, and stage 1 validates on them at the
  image level. Nothing selected on those sets is ever reported against them.
- **`retention > 1.0` is impossible for GRACE by construction.** The best a
  restorer can do is recover the clean-image score. If you see it, you are
  looking at a GRACE-D run.

### 4.4 The experiment index

Every arm is documented in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) with its
prerequisites, exact commands, and how to read each artifact. Each ablation
differs from its `dinov3_clean.yaml` baseline in **exactly one key**, repeated
verbatim rather than inherited, and
[tests/test_configs.py](grace_adapter/tests/test_configs.py) enforces it.

| # | Experiment | Asks | Cost |
|---|---|---|---|
| **D1** | preprocessing confound | is the baseline reading forensics or content? | ~2 GPU-h |
| **E0** | drift asymmetry | does RA-Det's asymmetry hold on this data? | minutes, no GPU |
| **E1** | identity adapter | does the split reproduce the baseline *exactly*? | one eval |
| **E2** | clean teacher | is the clean teacher the mechanism, or self-distillation? | 2 x minutes |
| **E3** | loss ablation | Jacobian weighting vs plain MSE | minutes |
| **E4** | **erasure trade-off** | **does restoring features destroy forensic evidence?** | 6 x seconds |
| **E5** | GRACE-D | does the fused score beat retention 1.0? | one eval |
| **E6** | cached vs live | is the finite epoch set being exploited? | hours |

```
D1 --> P1 stage 0 --> P2 baseline --> P3 caches --> E0
                                          |--> E2 --> E1 --> E5
                                          |      \-> E4
                                          |--> E3
                                          \--> E6
```

**D1, E1 and S0 are gates.** If the baseline does not collapse under degradation
there is no damage to repair. If the identity adapter does not reproduce the
baseline to the last decimal, the trunk/head split is wrong and every later
comparison is against a model nobody benchmarked — which is why E1 now has its
own run config, scored right after the baseline rather than at the end, and why
it stops `run_all.sh` outright when it fails. And every ablation verdict is
quoted in seed-sd, so the five-seed floor (S0) has to exist before E2, E3, E8 or
E9 can be read at all. `run_all.sh` covers all of it — there are no arms left to
run by hand.

### 4.5 Tests

```bash
cd eval_pipeline   && pytest -q     # 26 tests, incl. one end-to-end run on synthetic data
cd ../grace_adapter && pytest -q    # 232 tests
```

The grace suite includes a real end-to-end cache render, a two-stage training
smoke run, and the full PoC path — stage 0 -> cache -> stage 1 -> stage 2 ->
identity check — against a small locally-constructed DINOv3 that needs neither
network nor licence. The one it exists for is
[test_cache_alignment.py](grace_adapter/tests/test_cache_alignment.py): it
renders a real cache and re-runs the trunk live on 20 random indices, clean *and*
degraded. Cache/manifest row misalignment is the highest-risk bug in the project
— it trains, it converges, and it means nothing.

---

## 5. Status and results so far

**Implemented and tested.** The adapter, Jacobian weighting, the full objective,
diagnostics, the discrepancy branch, the degradation schedule, the cache
(writer/reader/spec), EMA, both training stages, two-axis validation, the
configs, `AdaptedDetector`, the DINOv3 proof-of-concept path, and optional W&B
tracking, the ladder and its tap caches. **319 tests pass** across the two
packages (293 `grace_adapter`, 26 `eval_pipeline`).

**Measured.** Stage 0 is complete on the full NTIRE train split (277,643 images),
for both preprocessing arms. Selection is on held-out **images**, by the
unweighted mean of AUC on `ntire_val` and `ntire_val_hard` — averaging matters,
because a head that wins on the easy set by giving up on the hard one must not be
selectable.

| probe | input | selected epoch | `ntire_val` AUC | `ntire_val_hard` AUC | mean |
|---|---|---|---|---|---|
| [`dinov3_ntire`](grace_adapter/checkpoints/probe/dinov3_ntire/head.summary.json) | 224 resize | 36 | **0.9596** | **0.8467** | 0.9032 |
| [`dinov3_ntire_crop`](grace_adapter/checkpoints/probe/dinov3_ntire_crop/head.summary.json) | 224 centre crop at native res | 38 | 0.9431 | 0.7930 | 0.8680 |

All six caches are fully rendered — train, val and val_hard, each in a pooled and
a tapped variant, 27 GB total.

**Adapters: none on disk.** The experiment set is being run again from a clean
tree — one reference arm, one script, every arm covered. A previous round trained
37 stage-1 and 8 stage-2 runs and produced verdicts for E0, E2, E3, E4, E8 and
E9; its best arm closed **87% ± 5%** of the degradation gap on `ntire_val_hard`,
though that gap is only 1.56 AUC points wide, so the result lived in the third
decimal place. Those numbers are kept in `docs/RESULTS.md` as **priors**, not as
records.

**Still not measured: any retention number.** `eval_pipeline/results/` is empty.
Round 1 never produced one because three of the four DINOv3 detector configs
pointed at run ids that did not exist — all three are repointed now, and every
checkpoint path in `configs/` names a run some step of `run_all.sh` produces.
Every retention figure in this README below is still a *target*, not a
measurement.

**The plan, the priors and the pre-registered predictions are in
[`docs/RESULTS.md`](docs/RESULTS.md) §0; the argument the experiments make is
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) §0.** The path is one command:
`bash scripts/run_all.sh` from `grace_adapter/`.

---

## 6. Limitations, and what I would do with more time

### 6.1 The headline result is not in yet

The honest summary of this repository today is that the *instrument* is finished
and the *experiment* is mid-flight. Stage 0 is done, the caches are most of the
way rendered, and every stage downstream of them is minutes or seconds of
compute — but the load-bearing arms (E2's clean teacher, E3's loss ablation)
have not run, so the two claims in section 1 remain claims. That ordering was
deliberate — the harness had to be trustworthy before any number from it meant
anything — but it means the deliverable is a reproducible path rather than a
table.

### 6.2 The proof-of-concept detector is the weakest possible test of GRACE-D

The PoC trunk emits one pooled `(B, 768)` vector, layout `vector`. That buys a
seam that *cannot* be wrong — `DINOv3MLPDetector.forward` literally **is**
`head(trunk(x))`, so E1 is a tautology rather than a nail-biter — and a cache 32x
smaller than RINE's. It costs two things:

- There is no "which encoder block does blur destroy" figure. That needs a
  `layers` split and a per-layer gate, which the code already supports
  ([factory.py](grace_adapter/grace/models/factory.py) picks the `(L, D)` gate
  shape off `FeatureSpec.layout` alone) — it needs a detector, not a rewrite.
- The stage-2 discrepancy head sees **one** drift norm where RINE's split would
  give it 24. **A null result on E5 here is therefore much weaker evidence
  against the discrepancy branch than a null on a `layers` detector would be**,
  and it has to be reported that way.

*With more time:* clone RINE into `third_party/`, verify
`RINESplit._head_forward` against the actual module structure (it is currently
written against documented structure — this is exactly why `verify_split` runs
inside every split's `__init__`), and re-run E4/E5 where they can actually
discriminate. Every `dinov3_*` config already has a `rine_*` twin.

### 6.3 The premise is only approximately true on NTIRE

GRACE exists to repair a head fitted on *clean* images. The NTIRE train set
already mixes in-the-wild transformations — crop, resize, compression, blur —
into **both** classes, and unlike the val set it ships no `is_distorted` flag to
filter them. So "fit on clean data" is only approximately true of any head
trained here, and it cannot be filtered: it is a property of the dataset. For the
same reason `ntire_val_distorted` and `ntire_val_hard` are level-0-only
evaluations — our grid stacked on an unknown prior transform makes both the L0
reference and L1's one-cause attribution untrue.

*With more time:* a dataset whose clean split is verifiably clean, or an explicit
degradation-augmented baseline as a separate detector config with its own
baseline, so that the comparison "adapter vs. just training with augmentation"
becomes measurable rather than assumed away.

### 6.4 Pre-rendering fixes the augmentation set

The cache buys the speed that makes the whole ablation grid affordable, but it
fixes the augmentation at 12 draws per image. The fair objection is that the
adapter might memorise those 12 corruptions rather than learn to undo the
*family* they came from. Three mitigations are built in — a fresh recipe per
(image, epoch) rather than per image, held-out degradation epochs numbered from
`VAL_EPOCH_OFFSET = 10_000` so their `replicate` can never collide with a
training draw, and `source: live` as the direct control — but **E6 is the slow
arm and has not been run**, so "cached is approximately live" is currently an
argument, not a measurement.

### 6.5 D1 is an open question on this dataset

On this project's previous dataset (SID_Set), the resize-fed probe reached 0.9999
val AUC and then held roughly 100% retention through `blur/sigma=2.0`,
`resize/0.25x` and full L3 composition — including a 32x32 round trip, a
resolution at which no generation trace survives. That head had learned
**content**, not forensics, and a detector that never collapses cannot
demonstrate a repair. The cause is upstream of the head: squashing a 1024px
source to 224x224 destroys the traces *before the trunk ever runs*. The crop arm
is the fix, and it is a fix to preprocessing rather than to the head. **That
finding has not been reproduced on NTIRE** — the mechanism belongs to the
preprocessing rather than to that dataset, so the arm is still worth having, but
on NTIRE it is an open question. The crop probe is trained (section 5) and the
ablation run config exists; the two baselines have not been scored against each
other yet.

### 6.6 Other things I would build

- **The ladder adapter** ([models/ladder.py](grace_adapter/grace/models/ladder.py),
  blueprint only). Degradation corrupts early blocks differently from late ones,
  so a single-seam correction is the wrong shape. One forward-compatibility
  decision was already made for it because it is cheap now and expensive to
  retrofit: `CacheSpec` carries an empty `taps` field and `SplitDetector` has a
  `taps()` hook, so the ladder *adds views* to an existing cache rather than
  invalidating its on-disk format.
- **Degradation prompts** ([models/prompts.py](grace_adapter/grace/models/prompts.py),
  blueprint only). A bank of learnable prompts selected by soft attention
  (PromptIR) with the degradation embedding obtained contrastively (AirNet) fits
  the label-free framing better than the current supervised severity scalar — and
  the attention weights are a soft classification of the degradation obtained
  *without* degradation labels, so comparing them against `recipes.parquet` is a
  free confusion-matrix figure.
- **Confidence intervals on retention.** The harness reports point estimates;
  E0's bootstrap CI on the drift asymmetry is the only interval in the project.
  Every headline retention number should carry one before it is quoted.
- **A calibration story.** The frozen head's operating point was calibrated on a
  wider feature cloud than a conditional-mean restorer produces. Nothing in the
  project yet reports what happens to the *threshold* under that mismatch — and
  FPR/FNR at a clean-fixed threshold is what a deployment actually cares about.

---

## 7. References

1. Wang, Wang, Zhang, Owens, Efros. *CNN-Generated Images Are Surprisingly Easy
   to Spot... for Now.* CVPR 2020.
2. Cui et al. *GlobalForge: Towards Robust AI-Generated Image Detection.*
   [arXiv:2607.14684](https://arxiv.org/abs/2607.14684) — RealDeg-Bench's
   seven-operator pool and depth range N in {1..5} independently match this
   harness's L2/L3 design.
3. Wang et al. *RA-Det: Towards Universal Detection of AI-Generated Images via
   Robustness Asymmetry.* [arXiv:2603.01544](https://arxiv.org/abs/2603.01544) —
   the drift-asymmetry premise behind GRACE-D.
4. Simeoni et al. *DINOv3.* The PoC trunk is
   `facebook/dinov3-vits16-pretrain-lvd1689m`, a ViT-S/16 distilled from the
   ViT-7B teacher on LVD-1689M.
5. Hong et al. *WildFake: A Large-scale Challenging Dataset for AI-Generated
   Images Detection.* AAAI 2025.
6. Potlapalli et al. *PromptIR*; Li et al. *AirNet* — the blueprint for the
   degradation prompts in section 6.6.
