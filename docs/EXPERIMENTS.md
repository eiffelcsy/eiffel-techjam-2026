# The experiments

Every experiment this repo implements: what it asks, what it needs, how to run
it, and how to read what comes out. One section each, in the order they become
answerable.

`grace_adapter/README.md` argues *why* the method is shaped the way it is.
`docs/PIPELINE.md` walks the code component by component. This is the operator's
document — the one to have open while running something.

> **Scope.** This document covers the **DINOv3 proof-of-concept path** only. That
> is the arm that runs today, end to end, on a detector assembled inside this
> repo. The model zoo (RINE, B-Free, GAPL) has a parallel set of configs, but its
> splits build on repos you have to clone by hand into `third_party/`, and those
> are not in this tree — so none of its arms can be run yet. Every `dinov3_*`
> config named below has a `rine_*` twin, and when the clones land this document
> applies to them unchanged. See the **Status** section of
> `grace_adapter/README.md`.

---

## Table of contents

1. [How to read any number in this project](#1-how-to-read-any-number-in-this-project)
2. [The index](#2-the-index)
3. [Prerequisites — what must exist before an experiment can run](#3-prerequisites--what-must-exist-before-an-experiment-can-run)
4. [D1 — the preprocessing confound](#4-d1--the-preprocessing-confound)
5. [E0 — drift asymmetry](#5-e0--drift-asymmetry)
6. [E1 — the identity adapter](#6-e1--the-identity-adapter)
7. [E2 — the clean teacher](#7-e2--the-clean-teacher)
8. [E3 — the loss ablations](#8-e3--the-loss-ablations)
9. [E4 — the erasure trade-off](#9-e4--the-erasure-trade-off)
10. [E5 — GRACE-D](#10-e5--grace-d)
11. [E6 — cached versus live](#11-e6--cached-versus-live)
12. [What the PoC cannot answer](#12-what-the-poc-cannot-answer)
13. [Where every number lands](#13-where-every-number-lands)
14. [The whole path in one command](#14-the-whole-path-in-one-command)

---

## 1. How to read any number in this project

Almost everything here reduces to one quantity, **retention**:

```
retention = (auc_degraded − 0.5) / (auc_clean − 0.5)
```

AUC 0.5 is chance, so this asks: *of the detector's skill above chance on clean
images, what fraction survives the degradation?* 1.0 means nothing was lost, 0.0
means the detector is now guessing.

`eval_pipeline` establishes the problem this measures. Detectors of generated
images mostly key on local high-frequency traces — the fine-grained pixel
statistics a generator leaves behind. JPEG, blur and downscaling destroy exactly
those, so retention collapses. GRACE's job is to win back part of that collapse
without fine-tuning the detector at all.

### Two denominators, and they are not interchangeable

| | divides by | written by | use for |
|---|---|---|---|
| harness retention | *each detector's own* clean AUC | `eval_pipeline/scripts/run_eval.py` | describing one detector |
| baseline-normalized retention | the **baseline's** clean AUC | `grace_adapter/scripts/compare.py` | comparing two detectors |

**Use `compare.py` for every GRACE-versus-baseline claim.** Here is why the
harness's own convention is wrong for that comparison. GRACE-D adds an auxiliary
head that reads Δ, the correction the adapter applied. On a clean image there is
nothing to correct, so Δ ≈ 0 and GRACE-D's clean AUC is roughly the baseline's.
On a degraded image Δ is large and informative, so its degraded AUC can go *up*.
Dividing by its own clean AUC would therefore shrink the very improvement being
claimed. Dividing by the **baseline's** clean AUC keeps the comparison honest,
and `> 1.0` then means something concrete: the adapted detector, working on a
degraded image, beats the original detector working on a clean one.

**`compare.py` refuses two result files from different eval sets.** Retention is
only comparable within one dataset. This is why `dinov3_poc_baseline.yaml` and
`dinov3_poc_grace.yaml` list the same `datasets:`. If you change one and not the
other, every comparison downstream breaks.

### The eval set is WildFake, not NTIRE val

`ntire_val` and `ntire_val_hard` are *selection* sets in this project: stage 0
uses them to pick the head's epoch, and stage 1 validates on them at the image
level. Measuring retention there would flatter the baseline and the adapter
alike, because both have already been tuned against those images.

WildFake is held out from all of it — 4998 real COCO val2017 photos and 8843
DALL·E 3 generations, from a generator the head never saw. The NTIRE val numbers
are still reported, but they come out of `checkpoints/grace/<run>/summary.json`
under `validation`, and they are never called retention.

### The degradation sweep is the same 26 conditions everywhere

- **L0** — clean, 1 condition. This is the reference every retention divides by.
- **L1** — one transform at a time, 19 grid points across 11 transforms.
- **L2 / L3** — two, and three-to-five, transforms composed. 3 independent
  re-draws each.

Each degradation is drawn from a seed computed from the image's index, so every
detector and every arm sees byte-identical degraded images. Degradation is also
applied *before* any detector's own preprocessing. Together that means a
difference between two rows is a difference between the models, never a
difference in what they were shown.

---

## 2. The index

| # | Experiment | Asks | Cost | Configs / entry point |
|---|---|---|---|---|
| **D1** | preprocessing confound | is the baseline reading forensics or content? | ~2 GPU-h | `probe/dinov3_ntire_crop.yaml` + `runs/dinov3_poc_baseline_crop.yaml` |
| **E0** | drift asymmetry | does RA-Det's asymmetry hold on this data? | minutes, no GPU | `scripts/analyze_drift.py` |
| **E1** | identity adapter | does the split reproduce the baseline *exactly*? | one eval | `detectors/dinov3+identity.yaml` |
| **E2** | clean teacher | does the clean teacher buy retention, or is it self-distillation? | 2 × minutes | `train/dinov3_clean.yaml` vs `dinov3_degraded.yaml` |
| **E3** | loss ablations | Jacobian vs MSE; ±SW; ±noise | 3 × minutes | `dinov3_plain_mse` / `dinov3_no_sw` / `dinov3_posterior` |
| **E4** | erasure trade-off | **does the adapter destroy forensic evidence as it restores?** | 6 × seconds | stage 2 vs every stage-1 checkpoint |
| **E5** | GRACE-D | does the fused score beat retention 1.0? | one eval | `detectors/dinov3+grace-d.yaml` |
| **E6** | cached vs live | is the finite epoch set being exploited? | hours (the slow arm) | `train/dinov3_live.yaml` |

**Two of these gate the others; the rest are independent.**

- **D1 gates everything.** If the baseline detector does not actually collapse
  under degradation, there is no damage for GRACE to repair.
- **E1 gates everything after it.** If the identity adapter does not reproduce
  the baseline to the last decimal, the trunk/head split is wrong, and every
  later comparison is measuring against a model nobody benchmarked.
- **E0 is not a gate, but read it early.** Both of its outcomes are useful, and
  one of them saves you a day of work on stage 2.
- **E2 through E6 do not depend on each other** and can run in any order.

```
D1 ──> P1 stage 0 ──> P2 baseline ──> P3 caches ──> E0
                                          │
                                          ├──> E2 ──> E1 ──> E5
                                          │      └──> E4
                                          ├──> E3
                                          └──> E6
```

E1 appears after E2 in that graph for a purely practical reason: both are scored
by the same `run_eval.py` command. `dinov3_poc_grace.yaml` loads all three
adapted detectors in one run, and the `+grace` arm needs a trained checkpoint to
already exist.

---

## 3. Prerequisites — what must exist before an experiment can run

Run all commands from `grace_adapter/` unless the prompt says `eval_pipeline/`.
`data/` sits at the repo root rather than inside either package, so a dataset
config's `../data/...` points at the same directory from both.

### P0 — manifests

A manifest is the parquet table listing every image and its label. It is built
once and then treated as fixed.

```bash
cd eval_pipeline
python scripts/build_manifest.py --config configs/datasets/ntire_train.yaml
python scripts/build_manifest.py --config configs/datasets/ntire_val.yaml
python scripts/build_manifest.py --config configs/datasets/ntire_val_hard.yaml
python scripts/build_manifest.py --config configs/datasets/wildfake_coco_dalle3.yaml
```

That is 277,643 / 10,000 / 2,500 / 13,841 rows. `ntire_val.yaml` and
`ntire_val_distorted.yaml` are two splits of one table, so the second command
covers both.

> **Never rebuild a manifest that a cache was rendered against.** The cache
> stores features by row position, and it records a hash of the manifest
> (`manifest_sha`) to prove the rows still line up. Rebuilding changes that hash
> and invalidates every feature in the cache. `poc.sh` skips existing manifests
> rather than overwriting them for this reason.

> `configs/datasets/ntire_train_eval.yaml` is **superseded and now selects zero
> rows.** It used to name shard 5, held back for model selection; shard 5 was
> folded into `split: train` once selection moved to the challenge's own val
> sets. Nothing should reference it.

### P1 — stage 0, the one place a detector is trained

A DINOv3 trunk is a feature extractor with no classifier on top. GRACE splices
itself between a trunk and a head, so the head has to exist first.

```bash
python scripts/train_probe.py configs/probe/dinov3_ntire.yaml
```

This is cheap because the trunk is frozen and the images are never degraded here,
which makes the features constant. So the trunk runs once per image, ever, and
the rest is AdamW on a 400k-parameter head reading those stored features.

Model selection is on held-out **images**, scored as the unweighted mean of AUC
on `ntire_val` and `ntire_val_hard`. Averaging the two matters: a head that wins
on the easy set by giving up entirely on the hard one cannot be selected. The
head is written to whatever `head_checkpoint` path the detector config names,
alongside a `head.summary.json`.

**Clean images only, no degradation augmentation.** This is the premise of the
whole project, not a corner cut. GRACE exists to repair a detector that was
fitted on clean data and then breaks under degradation. A head trained *with*
degradation augmentation would have already solved part of that problem itself,
and every retention number afterwards would be measuring the augmentation instead
of the adapter. If you want that arm, it is a separate detector config with its
own baseline, not a flag on this one.

The backbone `facebook/dinov3-vits16-pretrain-lvd1689m` is licence-gated on the
Hub. Accept the licence on the model page and run `hf auth login` once, or point
`backbone_id` at a mirror you already have — nothing in the code assumes the
official id.

### P2 — the baseline

```bash
cd eval_pipeline
python scripts/run_eval.py --config configs/runs/dinov3_poc_baseline.yaml
```

This produces the denominator that every GRACE arm is compared against, and
nothing else re-derives it. It costs about 360k forward passes: 13,841 images ×
26 conditions.

**Run it before training an adapter.** If retention does not collapse here, then
there is no gap for GRACE to close and the PoC has answered its own question
early — at roughly 8% of what the stage-1 cache costs.

### P3 — the feature caches

The trunk is frozen and a clean image never changes, so its features are always
the same. Compute them once, write them to disk, and the "teacher" the adapter
learns from becomes a lookup instead of a model.

The same reasoning extends to the *degraded* side. Every degradation recipe is
drawn from `stable_seed(index, level, replicate, seed)` — a hash of those four
values, not a running RNG counter. So a degraded view is also a pure function of
(image, condition), and a training epoch is just the `replicate` field under
another name. Epoch 7's degradation of image 412 can be computed right now,
without having run epochs 0 through 6 first, which is what makes rendering every
epoch ahead of time possible.

```bash
python scripts/build_cache.py configs/cache/dinov3.yaml --dry-run   # ALWAYS first
python scripts/build_cache.py configs/cache/dinov3.yaml
python scripts/build_cache.py configs/cache/dinov3_val.yaml
python scripts/build_cache.py configs/cache/dinov3_val_hard.yaml
```

| cache | rows | views | on disk |
|---|---|---|---|
| `cache/dinov3-ntire` | 277,643 | clean + 12 train + 2 held-out | ~6–7 GB |
| `cache_val/dinov3-ntire` | 5,000 | same | ~115 MB |
| `cache_val_hard/dinov3-ntire` | 2,500 | same | ~58 MB |

**The two val caches are not optional.** Every `configs/train/dinov3_*.yaml`
names them under `val_cache_dirs`, and `train_adapter.py` opens them in its first
few statements — so a missing one fails immediately rather than after the whole
run, at the one moment its result was wanted. They each get their own `out_dir`
because `build_cache.py` names its output directory after the *detector* alone,
so two datasets rendered for one detector would otherwise write to the same
place.

The held-out **degradation** views are numbered from `VAL_EPOCH_OFFSET = 10_000`.
Since the epoch number is the `replicate` field, starting them at 10,000
guarantees they can never collide with a training epoch's draw — they are
genuinely disjoint samples from the same distribution.

Rendering is resumable one view at a time (each finished view gets a `.done`
marker), so an interrupted render picks up where it stopped instead of starting
over.

> With `source: cache`, the training loop runs **no trunk forward at all**. One
> step is two reads from a memory-mapped array plus a 2-layer MLP. That is what
> turns E3's three arms and E4's six-point sweep from "in principle" into
> something you run over lunch.

---

## 4. D1 — the preprocessing confound

**Asks:** is the baseline detector reading generation traces, or is it just
reading *content*?

**Why it comes first.** GRACE needs a detector whose accuracy collapses under
degradation, because that collapse is the entire gap the adapter exists to close.
Now consider a head that has instead learned to separate "looks like
prompt-generated imagery" from "looks like a photograph". That is a semantic
distinction, and no transform in the grid destroys semantics — a blurred picture
of a dragon is still a picture of a dragon. Such a head never collapses, its
retention curve is a flat line at 100%, and there is no room for a repair to show
up.

This has already happened once. On this project's previous dataset (SID_Set), the
resize-fed probe reached 0.9999 validation AUC and then held roughly 100%
retention through `gaussian_blur/sigma=2.0`, `resize/scale=0.25` and full L3
composition. Nothing that reads local high-frequency traces survives that. The
cause sits upstream of the head: preprocessing with `size: {224, 224}` and
`default_to_square: true` shrinks a 1024px source by about 4.6× *before the trunk
ever runs*, so the traces are already gone by the time the head could learn them.
The full table is in `eval_pipeline/configs/detectors/dinov3-ntire-crop.yaml`.

**Those numbers are from SID_Set and have not been reproduced on NTIRE.** The
mechanism belongs to the preprocessing rather than to that dataset, so the arm is
still worth having — but on NTIRE it is an open question, not a measurement.

**Run:**

```bash
# stage 0 for the crop arm -- same manifest, same selection sets, same geometry,
# same optimizer, same seed as dinov3_ntire.yaml. `input_mode` is the one variable.
python scripts/train_probe.py configs/probe/dinov3_ntire_crop.yaml

cd ../eval_pipeline
python scripts/run_eval.py --config configs/runs/dinov3_poc_baseline_crop.yaml
python scripts/report.py --results results/
```

**Read:** the two baseline JSONs side by side, at `levels.*.retention` and
`operating_envelope`.

| outcome | what it means | what to do |
|---|---|---|
| crop collapses, resize does not | the resize head took the semantic shortcut | repoint `configs/cache/dinov3.yaml` and every `configs/train/dinov3_*.yaml` at `dinov3-ntire-crop.yaml`, re-render, redo the PoC on the crop head |
| both collapse | preprocessing was not the confound on NTIRE | the resize baseline stands; proceed |
| neither collapses | the **dataset** separates on content | a finding about NTIRE. No adapter fixes it |

The two heads are deliberately not interchangeable. `_assert_head_matches`
refuses to load one into the other, because the trunk sees the image at a
different scale in each mode and the features are not the same thing.

---

## 5. E0 — drift asymmetry

**Asks:** when you degrade an image, do generated images move further in feature
space than real ones do? This is RA-Det's premise, tested on *this* data, before
anything has been trained on it.

**Why it matters.** GRACE-D claims the damage itself carries information.
Concretely: Δ = `adapter(f_deg) − f_deg` is how much the adapter thought it had
to correct, which is its estimate of how far the image drifted. Δ is available at
test time without ever seeing the clean image, and it falls out of a module that
was running anyway.

That claim needs the drift to differ between real and generated images. If it
does not, the discrepancy branch has nothing to read. If it does, then stage 1's
label-free objective is quietly destroying forensic evidence every time its
reconstruction loss improves — which is exactly what E4 goes on to measure.

**This needs the rendered cache**, including at least one finished degraded view,
because the whole analysis compares the `clean` view against an `epoch=NNN` view.
"E0 first" means before anything is *trained*, not before the render.
`FeatureCache` is opened in the first statement of `main`, so a missing cache
fails at once, and a cache with no finished degraded view exits with
`no rendered epochs under <dir>`.

**Run** (minutes, no GPU, no training):

```bash
python scripts/analyze_drift.py \
  --cache    cache/dinov3-ntire \
  --dataset  ../eval_pipeline/configs/datasets/ntire_train.yaml \
  --detector ../eval_pipeline/configs/detectors/dinov3-ntire.yaml \
  --split    grace.splits.dinov3.DINOv3Split \
  --out      results/dinov3_poc_drift.json
```

`--detector` and `--split` are optional. Without them you still get the drift
magnitudes, but not the decomposition into "along the head's decision direction"
versus "orthogonal to it", which needs the model's weights.

**Read** `results/dinov3_poc_drift.json`:

| key | meaning |
|---|---|
| `significant` | the bootstrap confidence interval on the real-versus-fake drift gap excludes zero. **This is the decision.** |
| `asymmetry_ci` | the interval itself. If it straddles zero there is no evidence here, however large the point estimate looks |
| `overall.parallel_fraction` | what fraction of the drift lies along the direction the frozen head is actually sensitive to |
| `by_level`, `by_transform` | where the asymmetry comes from. `by_transform` rows overlap: a composed recipe counts once for each transform in it |

**Both outcomes are useful.**

- `significant: true` → the drift carries forensic signal, the discrepancy branch
  has something to read, and stage 1 is erasing it. Go on to E4 and E5.
- `significant: false` → stage 2 will be weak here. Say so, keep the restoration
  result, and save a day. That is a finding about this dataset, not a refutation
  of RA-Det.

**The parallel/orthogonal split matters as much as the gap.** The frozen head
compresses a whole feature vector into one number, so it only notices movement
along the direction it is sensitive to. Drift that is large but *orthogonal* to
that direction is invisible to it. That is precisely why a separate auxiliary
head reading Δ can find signal the main head cannot — and why the main loss can
keep falling while evidence is being destroyed.

---

## 6. E1 — the identity adapter

**Asks:** with a *null* adapter in place, does `head(adapter(trunk(x)))` give
back exactly what `detector(x)` gives?

**Why.** If it does not, the trunk/head split is wrong, and every comparison
after it is against a model that was never benchmarked. **Run it no matter what
else you are doing.**

On the PoC this is nearly a tautology, and that is the point. `DINOv3MLPDetector.forward`
literally *is* `self.head(self.trunk(x))`, and `DINOv3Split` hands back those two
attributes rather than trying to rebuild them. The arm you debug the pipeline in
should be the one whose seam cannot be wrong. (`verify_split` also runs inside
every split's `__init__`, so a mis-assembled split fails loudly at construction
instead of quietly scoring the wrong model.)

**Run** — it is one of the three detectors in the comparison run:

```bash
cd ../eval_pipeline
python scripts/run_eval.py --config configs/runs/dinov3_poc_grace.yaml
```

```bash
cd ../grace_adapter
R=../eval_pipeline/results       # reused by the compare.py blocks in §7 and §10
python scripts/compare.py \
  --baseline $R/dinov3_poc_baseline__dinov3-ntire__wildfake-coco-dalle3.json \
  --adapted  $R/dinov3_poc_grace__dinov3+identity__wildfake-coco-dalle3.json
```

Result files are named `{run_id}__{detector}__{dataset}.json`. `{dataset}` is the
dataset config's `name:` field — `wildfake-coco-dalle3`, hyphens and all — not
the filename.

**Read:** every `delta` in the printed table must be `+0.0000`. Anything else and
you stop here.

The identity is exact, not approximate. The final projection inside every adapter
block is initialised to zero, so an untrained adapter returns its input bit for
bit, whatever the gate, the noise input or the severity conditioning happen to
be. The same trick means `β = 0` makes GRACE-D *identical* to GRACE at
initialization. Without that guarantee, any change in clean AUC would be
impossible to attribute.

---

## 7. E2 — the clean teacher

**Asks:** is the clean-feature teacher doing the work, or would any smoothing
target produce the same result?

**The ablation is one config key.**

| arm | config | `target_view` | target |
|---|---|---|---|
| **B** (proposed) | `train/dinov3_clean.yaml` | `clean` | `f_clean` from the cache |
| **A** (control) | `train/dinov3_degraded.yaml` | `degraded` | `f_deg.detach()` |

Arm A asks the adapter to reproduce its own input. That is self-distillation with
no information added, and it should achieve nothing. So if arm A matches arm B,
the clean teacher was not the mechanism and something else explains the result.
`tests/test_configs.py::test_arms_differ_only_in_target_view` checks that the two
files differ in exactly one key, so this cannot quietly become a two-variable
comparison.

**Run** (minutes each — no trunk forward inside the loop):

```bash
python scripts/train_adapter.py configs/train/dinov3_clean.yaml      # arm B
python scripts/train_adapter.py configs/train/dinov3_degraded.yaml   # arm A
```

**Read** `checkpoints/grace/dinov3_{clean,degraded}/summary.json`:

```
validation
├── held_out_degradations/epoch_10000..1   unseen CORRUPTIONS, training images
└── held_out_images/<dataset>/epoch_*      unseen IMAGES (ntire_val, ntire_val_hard)
```

These are two different questions and they are reported separately so a single
number cannot hide which one failed. The cache's own held-out epochs use unseen
*corruptions*, but every image they score was trained on, so they say nothing
about generalizing to new images. The val datasets answer that second question.
Each entry carries `cosine_to_clean`, `posterior_spread` and `gate`.

Neither is AUC, deliberately. Retention is measured by the eval harness, on the
eval split, through `grace.detectors.adapted`. What you get here is an in-loop
signal — enough to notice a run that helps L1 while wrecking L3 before it
finishes.

Two more keys are worth watching in `history[]`:

- **`gate`** — how strongly the adapter is applying its correction. It starts at
  `0.018` and should climb to somewhere around 0.1–0.5. Still sitting at the
  initial value means the alignment term never overcame the identity term.
  Saturated at 1.0 means it is over-correcting.
- **`cos_decision`** — `cos(Δ, j)`, the alignment between the correction and the
  head's sensitive direction. Near 0 means the adapter is spending its capacity
  fixing feature directions the head cannot see, so none of that work can move
  AUC.

The headline retention number for arm B comes from the harness, not from here:

```bash
python scripts/compare.py \
  --baseline $R/dinov3_poc_baseline__dinov3-ntire__wildfake-coco-dalle3.json \
  --adapted  $R/dinov3_poc_grace__dinov3+grace__wildfake-coco-dalle3.json
```

For GRACE, `retention_adapted > 1.0` is **impossible by construction**: the best
a restorer can do is recover the clean-image score. If you see it, you are
looking at a GRACE-D run, not a GRACE one.

---

## 8. E3 — the loss ablations

**Asks:** which terms of the objective actually earn their place?

```
L = L_align + λ_sw·L_SW + λ_id·L_identity + λ_kl·L_headKL + λ_sev·L_severity
```

Every term is label-free. Three arms, each **identical to `dinov3_clean.yaml`
apart from the one key being ablated.** Any other difference and it is not an
ablation.

### E3a — Jacobian weighting off (`dinov3_plain_mse.yaml`)

Plain MSE treats every feature dimension as equally worth fixing. The head does
not agree: it collapses the whole feature vector into a single score, so only the
part of the error that lies along the direction it is sensitive to can change
AUC. Everything perpendicular to that is capacity spent on something no AUC will
ever reflect.

The fix is to weight the error by how much the head actually responds to it,
which is the head's gradient with respect to its input:

```
j = ∇_f h(f) |_{f = f_clean}
L_err = (1−ε)·mean[(ĵ·e)²] + ε·mean[e²]        e = f_adapted − f_clean
```

Writing it as a gradient rather than as "project onto the weight vector" is what
lets one implementation cover both linear and MLP heads with no branch — for a
linear head the gradient `j` is just the constant weight vector `w`. And writing
it as a blend between the weighted and unweighted terms means `ε = 1` is
*exactly* `F.mse_loss`.

So `weighting: none` reproduces the GRACE v1 objective **provably**, not
approximately, and `tests/test_losses.py` pins that equivalence. `lam_kl` also
returns to v1's `0.5`, because back then `head_kl` was the only term that knew
anything about the decision.

**Read:** `history[].cos_decision`, next to `dinov3_clean`'s. If this arm sits
near 0 while the weighted run climbs, that is the empirical case for the whole
weighting term.

### E3b + E3c — sliced-Wasserstein off, and posterior sampling on

**These are one experiment split across two configs. Running either alone answers
nothing.**

Start with what a point-wise reconstruction loss actually optimises. Minimising
squared error to a target is solved by predicting the *conditional mean*, and a
mean is always less spread out than the distribution it came from. So the batch
of corrected features ends up as a tighter cloud than a batch of genuinely clean
features would be — and the frozen head's threshold was calibrated on the wider
cloud.

The sliced-Wasserstein term fixes that by matching the two distributions rather
than just their per-image values: project both batches onto random directions,
sort, compare. Because both batches contain the same images, this is a matched
comparison, which is stronger than the usual unpaired form.

Posterior sampling is the other half. The adapter gets an extra noise input `z`,
draws k different corrections, and averages the resulting **logits, not
features**. Averaging logits matters because `E[h(f)] ≠ h(E[f])` for any
nonlinear head, so this is genuine Monte-Carlo averaging over the posterior — and
it is cheap, since k adapter passes cost microseconds against one trunk pass.

> Here is why they are one experiment. Under point-wise losses alone, the best
> thing a stochastic adapter can do is ignore `z` entirely — that is posterior
> collapse, and it is optimal, not a failure. The noise only earns its parameters
> because the SW term rewards reproducing a spread that a conditional mean
> cannot. So `noise_dim > 0` with `lam_sw: 0` buys parameters that provably do
> nothing.

| config | `lam_sw` | `noise_dim` |
|---|---|---|
| `dinov3_no_sw.yaml` | `0.0` | 0 |
| `dinov3_posterior.yaml` | 0.1 (default) | 16 |

**Read:** `validation.*.posterior_spread`, which is the tripwire. If
`dinov3_posterior` reads ~0, the adapter learned to ignore `z`. That is a
reportable negative result about the objective, not a bug to fix.

**Run:**

```bash
python scripts/train_adapter.py configs/train/dinov3_plain_mse.yaml
python scripts/train_adapter.py configs/train/dinov3_no_sw.yaml
python scripts/train_adapter.py configs/train/dinov3_posterior.yaml
```

All three set `wandb.group: e3_losses`, so with `--wandb` the arms land in one
comparison instead of a flat list of runs.

> Scoring the posterior checkpoint *through the harness* needs a detector config
> with a matching `k_eval`. `configs/detectors/dinov3+grace.yaml` pins
> `k_eval: 1`, which is right for the deterministic arms and wrong for this one:
> `AdaptedDetector` only forces `k_eval` down to 1 when the adapter has no noise
> input. The in-loop `validation` block already uses `sampling.k_eval: 8`.

---

## 9. E4 — the erasure trade-off

**Asks: does restoring features destroy the forensic evidence they carried?**
This is the sharpest question in the project, and it meets the obvious critique
of the whole approach head-on.

**The argument.** E0 says generated images drift further under perturbation than
real ones do. Stage 1 trains the adapter to undo drift. So an adapter that gets
better at its job is, by construction, removing more drift from the fakes than
from the reals — which is the same as removing the thing that distinguished them.
Its reconstruction loss falls the whole time. If this is happening, the auxiliary
head's standalone AUC should *fall* as stage 1 improves.

**The design that makes it testable.** Stage 2 never touches the adapter; it
trains only the auxiliary head and the fusion weight β. That is why GRACE and
GRACE-D ship the same adapter weights bit for bit, and why "the adapter is
trained without labels" stays literally true. It also means you can train stage 2
against *each* stage-1 checkpoint in turn, with the adapter as the only thing
that varies.

`dinov3_clean.yaml` sets `checkpoint_every: 2`, so a 12-epoch run leaves six
intermediate checkpoints. Stage 2 takes **seconds**, which is what makes sweeping
all six practical rather than theoretical.

**Run** (bash / Git Bash):

```bash
for ck in checkpoints/grace/dinov3_clean/step_*.pt; do
  python scripts/train_discrepancy.py configs/train/dinov3_discrepancy.yaml \
    --adapter "$ck" --run-id "e4_$(basename "$ck" .pt)" \
    --wandb --wandb-group e4_erasure
done
```

PowerShell:

```powershell
Get-ChildItem checkpoints/grace/dinov3_clean/step_*.pt | ForEach-Object {
  python scripts/train_discrepancy.py configs/train/dinov3_discrepancy.yaml `
    --adapter $_.FullName --run-id "e4_$($_.BaseName)" `
    --wandb --wandb-group e4_erasure
}
```

**Read** `checkpoints/grace/e4_step_*/summary.json`, key
`validation.epoch_*.auc_aux`, plotted against stage-1 progress. Stage 2 records
`adapter_checkpoint` in its run config, and that is the x-axis of the figure.

Three AUCs are reported per epoch, and what matters is the relationship between
them:

| key | reading |
|---|---|
| `auc_main` | the frozen head on adapted features — what GRACE alone achieves |
| `auc_aux` | the auxiliary head on Δ alone. **Above chance = drift carries signal.** This is the E4 curve |
| `auc_fused` | `main + β·aux`. Above `auc_main` = Δ carries signal the main head was not already using |

A **falling `auc_aux` as stage 1 improves** is direct evidence that restoration
erases forensic evidence, and plotting retention against drift preservation is
the figure that shows it. A flat curve means the two objectives do not conflict
on this data: a simpler story, and a less interesting result.

---

## 10. E5 — GRACE-D

**Asks:** does the fused score break the restoration ceiling?

```
GRACE     logit = head(adapter(trunk(x)))                        label-free
GRACE-D   logit = head(adapter(trunk(x))) + β·aux(Δ, severity)   + labels
                                            Δ = adapter(f_deg) − f_deg
```

There is a hard ceiling on restoration: the best a perfect restorer can do is
recover the score the detector would have given the clean image, i.e. retention
1.0. A fused score that reads Δ can go past it, because *how much damage the
image took* is information the clean image never contained. RA-Det gets a similar
signal, but only by running a second forward pass on a deliberately perturbed
copy; here it falls out of a module that was already running.

**Run:**

```bash
python scripts/train_discrepancy.py configs/train/dinov3_discrepancy.yaml

cd ../eval_pipeline
python scripts/run_eval.py --config configs/runs/dinov3_poc_grace.yaml
```

```bash
cd ../grace_adapter
python scripts/compare.py \
  --baseline $R/dinov3_poc_baseline__dinov3-ntire__wildfake-coco-dalle3.json \
  --adapted  $R/dinov3_poc_grace__dinov3+grace-d__wildfake-coco-dalle3.json
```

**Read:** `exceeds_clean_ceiling` in the comparison output, and
`retention_adapted` at each level. `> 1.0` is the headline claim. Only the
discrepancy branch can produce it, so `compare.py` prints a warning telling you
to confirm the run really is GRACE-D before reporting the number.

`configs/detectors/dinov3+grace.yaml` and `dinov3+grace-d.yaml` must name the
**same** adapter checkpoint. Otherwise the comparison changes two things at once
and neither can be attributed.
`tests/test_configs.py::test_grace_and_grace_d_share_an_adapter` enforces it.

**The PoC tests this branch in its weakest possible form.** The DINOv3 split
produces one pooled vector, so the auxiliary head sees a single drift magnitude.
RINE's `layers` split would give it 24, one per encoder block. A null result here
is therefore much weaker evidence against the branch than a null result on a
`layers` detector would be, and it should be reported that way.

---

## 11. E6 — cached versus live

**Asks:** is stage 1 exploiting the fact that the pre-rendered augmentations are
a finite set?

Pre-rendering fixes the augmentation at 12 draws per image. The fair objection is
that the adapter might be memorising those 12 specific corruptions rather than
learning to undo the degradation *family* they were drawn from. `source: live`
settles it: same schedule, same `grid_file`, same level weights, but a fresh
recipe sampled every step and the trunk running inside the loop. That is the
direct control.

Two other defences are already built in: a fresh recipe per (image, epoch) rather
than one per image, and the held-out degradation epochs numbered from 10,000.

**Run** — this is the one arm that is *not* fast. Cached steps skip the trunk
entirely; this one pays a full DINOv3 forward per image per step. Expect hours
where the other arms take minutes.

```bash
python scripts/train_adapter.py configs/train/dinov3_live.yaml
# ...or cap it, since it is a control:
python scripts/train_adapter.py configs/train/dinov3_live.yaml --epochs 4
```

**Read:** `checkpoints/grace/dinov3_live/summary.json` against
`dinov3_clean/summary.json`, on the same `validation` keys.

- cached ≈ live → the finite epoch set is not being exploited, and every cached
  number in E2–E5 stands.
- a gap → the cached runs are partly memorising specific corruptions, and
  `n_epochs` in `configs/cache/dinov3.yaml` has to go up, which means a
  re-render.

`batch_size` drops from 64 to 32 in this config because it now counts **images**
rather than cached feature vectors: the trunk is back in the loop, so the memory
cost is the detector's rather than the adapter's. Stage-1 validation still reads
pre-rendered caches either way.

---

## 12. What the PoC cannot answer

The PoC trunk is `DINOv3 ViT-S/16 (distilled), pooled → (B, 768)`, feature layout
`vector`. That buys a seam that cannot be wrong and a cache 32× smaller than
RINE's. Here is what it costs:

| | answerable here | why |
|---|---|---|
| D1 preprocessing | **yes** | it is a property of this detector |
| E0 drift asymmetry | **yes** | a magnitude comparison, layout-agnostic |
| E1 identity | yes, trivially | the seam is a construction, not a reconstruction |
| E2 clean teacher | **yes** | needs no particular layout |
| E3 loss ablations | **yes** | all terms operate on the last axis |
| E4 erasure trade-off | partially | Δ is one norm here, not a per-block profile |
| E5 GRACE-D | weakly | the auxiliary head's weakest possible input |
| per-block damage figure | **no** | needs a `layers` split |

E2 and E3 are the load-bearing arms, because they decide whether the *objective*
works at all — and they run here in seconds. That is the whole argument for the
PoC: the experiments that genuinely need a published detector keep waiting on its
clone, but they become the only thing waiting.

Switching to a `layers` split is a small change when it is wanted: emit per-block
CLS tokens as `(B, 12, 384)` and give the head a learned weighting over them.
Nothing downstream needs touching, because `grace.models.factory` picks the
`(L, D)` gate shape off `FeatureSpec.layout` alone.

---

## 13. Where every number lands

| artifact | written by | holds |
|---|---|---|
| `grace_adapter/checkpoints/probe/<run>/head.summary.json` | `train_probe.py` | P1: per-epoch train AUC, per-val-set AUC/acc, `selected_epoch` |
| `grace_adapter/results/dinov3_poc_drift.json` | `analyze_drift.py` | E0: `significant`, `asymmetry_ci`, `parallel_fraction`, by level and transform |
| `grace_adapter/checkpoints/grace/<run>/summary.json` | `train_adapter.py` | E2/E3/E6: `history` (gate, `cos_decision`, loss terms), `validation` on both held-out axes |
| `grace_adapter/checkpoints/grace/<run>/{ema,last}.pt` | `train_adapter.py` | the adapter. `ema.pt` is what the detector configs load |
| `grace_adapter/checkpoints/grace/<run>/step_*.pt` | `train_adapter.py` (`checkpoint_every`) | E4's x-axis |
| `grace_adapter/checkpoints/grace/<run>/summary.json` (stage 2) | `train_discrepancy.py` | E4/E5: `beta`, `auc_main`/`auc_aux`/`auc_fused` |
| `eval_pipeline/results/{run_id}__{detector}__{dataset}.json` | `run_eval.py` | D1/E1/E5: per-level and per-condition AUC, retention, `operating_envelope` |
| stdout + optional `--out` JSON | `compare.py` | the baseline-normalized retention table and `exceeds_clean_ceiling` |

**W&B is never the record.** The `summary.json` next to the checkpoints is the
source of truth, and it is written whether or not anything was tracked — so two
people comparing results compare files, not screenshots. Tracking is off by
default and never fails a run: a dead network warns once and continues. Steps are
logged on the training-step axis, passed explicitly, so two runs with different
`log_every` remain comparable. Set `--wandb-group` to the experiment id and every
arm of a sweep lands in one comparison.

```bash
python scripts/train_adapter.py configs/train/dinov3_clean.yaml --wandb --wandb-group e2_teacher
python scripts/train_adapter.py configs/train/dinov3_clean.yaml --wandb --wandb-offline   # no outbound network
```

---

## 14. The whole path in one command

```bash
cd grace_adapter
bash scripts/poc.sh              # P0 -> P1 -> P2 -> P3 -> E0 -> E2 -> E5 -> E1
bash scripts/poc.sh --smoke      # 2 epochs, 2 cache views -- minutes, proves wiring
WANDB=1 bash scripts/poc.sh      # every stage tracked under one group
```

Eight steps, and every one is idempotent: existing manifests are skipped, the
cache resumes one view at a time, and re-running a training stage overwrites its
own run directory and nothing else. When it finishes it prints what to read, in
order.

`poc.sh` does **not** cover D1, E3, E4 or E6. Those are deliberate extra arms
rather than part of the main spine — run them from §4, §8, §9 and §11 above.

---

## Gotchas that bite an experiment specifically

- **Cache/dataset row misalignment is the highest-risk bug in the project.** If
  the cache's rows stop lining up with the manifest's, training still runs, the
  loss still falls, and the result means nothing.
  `tests/test_cache_alignment.py` guards it by rendering a real cache and then
  re-running the trunk live on 20 random indices, clean *and* degraded.
- **Never shuffle before caching, and never rebuild a manifest afterwards.** Four
  fingerprints (`manifest_sha`, `schedule_sha`, `detector_sha`,
  `preprocess_sha`) are checked when a cache is opened, and the error names which
  one moved.
- **Re-fitting the probe does not invalidate a cache.** The cached features come
  from the trunk, and the trunk never sees the head. `detector_sha` hashes the
  config, whose head path names weights those features never touched.
- **Unknown config keys raise rather than being ignored.** A typo that silently
  falls back to the default objective costs a day and looks like a negative
  result.
- **`checkpoint: null` in a detector config means identity**, i.e. exactly the
  base detector. That is E1's arm, not a broken config.
- **Changing `schedule.seed` invalidates every degraded view** in a cache.
- **`compare.py` needs both JSONs from the same eval set.** If you add a dataset
  to one run config, add it to the other.
