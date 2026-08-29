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

> **The "Prior round" boxes.** Several sections open with a quoted result. Those
> are from an earlier round whose checkpoints are **no longer on disk** — the
> experiment set is being run again from a clean tree, against a single reference
> arm, on the current objective. They are kept because a prediction is worth more
> than a blank page: each one says what to expect and what would be surprising,
> and two of them (E3's readout and E4's direction) came back contrary to what
> the section around them argues. Read them as priors, not as records. The
> record is [`docs/RESULTS.md`](RESULTS.md).

---

## Table of contents

0. [The argument these experiments make](#0-the-argument-these-experiments-make)
1. [How to read any number in this project](#1-how-to-read-any-number-in-this-project)
2. [The index](#2-the-index)
3. [Prerequisites — what must exist before an experiment can run](#3-prerequisites--what-must-exist-before-an-experiment-can-run)
4. [D1 — the preprocessing confound](#4-d1--the-preprocessing-confound)
5. [E0 — drift geometry, and the ceiling](#5-e0--drift-geometry-and-the-ceiling)
6. [E1 — the identity adapter](#6-e1--the-identity-adapter)
7. [S0 — the seed floor](#7-s0--the-seed-floor)
8. [E2 — the clean teacher](#8-e2--the-clean-teacher)
9. [E3 — the loss ablation](#9-e3--the-loss-ablation)
10. [E4 — the erasure trade-off](#10-e4--the-erasure-trade-off)
11. [E5 — GRACE-D](#11-e5--grace-d)
12. [E6 — cached versus live](#12-e6--cached-versus-live)
13. [E7 — the ladder](#13-e7--the-ladder)
14. [E8 — the gate](#14-e8--the-gate)
15. [E9 — the hyperparameter sweeps](#15-e9--the-hyperparameter-sweeps)
16. [What the PoC cannot answer](#16-what-the-poc-cannot-answer)
17. [Where every number lands](#17-where-every-number-lands)
18. [The whole path in one command](#18-the-whole-path-in-one-command)

---

## 0. The argument these experiments make

Twenty-two steps is not a narrative. This is the narrative, and every arm below
exists to supply one line of it.

> **A frozen detector loses accuracy under degradation. A ~0.4M-parameter
> residual adapter, trained without labels, wins part of it back — and the size
> of "part" is not a mystery to be discovered after the fact. It is bounded by a
> geometric quantity you can compute before training anything.**

The five moves, in order:

**1. There is a gap, and it is the gap we think it is.** Retention has to
collapse on the eval set (**P2**), and it has to collapse because the detector
was reading generation traces rather than image *content* — a head that separates
"looks generated" from "looks photographed" is making a semantic judgement, and
no blur destroys semantics. **D1** tells those two apart, by re-running the
identical detector with one preprocessing key changed. If neither arm collapses,
the finding is about the dataset and no adapter is relevant. That is the one
outcome that ends the project, so it is measured first, at about 8% of what the
feature cache costs.

**2. The instrument is exact.** **E1** puts a *null* adapter in the seam and
demands that `head(adapter(trunk(x)))` reproduce `detector(x)` to the last
decimal. It is nearly a tautology on this detector, and that is the point — the
arm you debug a pipeline in should be the one whose seam cannot be wrong. It is a
gate rather than a result: if it fails, every later comparison is against a model
nobody benchmarked, so it runs before the cache render and stops the pipeline
outright.

**3. Most of the damage was never repairable, and we can say how much.** This is
the move that makes the rest of the project honest. The frozen head compresses a
768-dimensional feature vector into one number, so it can only notice movement
along the single direction it is sensitive to. Drift orthogonal to that direction
is invisible to it — correcting it perfectly would not move AUC by a thousandth.
**E0** measures exactly that split (`parallel_fraction`) from the rendered cache,
with no adapter in existence. It converts "GRACE recovered X% of the gap" from a
number floating in space into a number with a denominator: *of the damage the
head could in principle have noticed*, how much came back.

E0 also answers RA-Det's question — do generated images drift *further* under
perturbation than real ones? — which decides whether the discrepancy branch
(move 5) has anything to read at all.

**4. The gain is real, it is small, and we know which part of the objective
produced it.** **S0** trains the reference arm five times and reports the spread,
which is the unit every later verdict is quoted in: two arms closer together than
about twice the seed spread are the same arm, and saying so is the difference
between an ablation table and a list of coincidences. Then three one-key
ablations attribute the gain:

- **E2** removes the clean teacher (`target_view: degraded`), leaving an adapter
  asked to reproduce its own input. No information is added, so it should gain
  nothing. If it matches the real arm, the teacher was not the mechanism.
- **E3** removes the Jacobian weighting (`weighting: none`), which is provably
  exactly plain MSE. Weighting the error by how much the head actually responds
  to it is the objective's one non-obvious idea; this is whether it pays.
- **E8** asks whether the gate — reported everywhere as "the adapter learning to
  apply its correction" — is learning at all, or whether decoupled weight decay
  is pulling its logit toward zero on its own.

  **E9** then sweeps the remaining knobs against the S0 floor, and a *saturated*
  capacity axis is a result **for** the method: GRACE claims the evidence is
  displaced rather than destroyed, and a displacement should be undoable by a
  small correction. If the gap only closed with a large adapter, the claim was
  false.

**5. What it costs, and what it leaves on the table.** Two hard questions close
the argument.

- **E4** is the sharpest critique of the whole approach, and it is aimed at us.
  If fakes drift further than reals (E0), and the adapter is trained to undo
  drift, then an adapter getting *better* at its job is removing more of what
  distinguished fakes from reals — destroying forensic evidence while its
  reconstruction loss falls. It is testable only because stage 2 never touches
  the adapter, so stage 2 can be trained against each stage-1 checkpoint in turn
  with the adapter as the only thing that varies.
- **E5** tries to break the ceiling. Restoration cannot exceed retention 1.0 — the
  best a restorer can do is recover the clean-image score. A score that also
  reads Δ, the correction the adapter proposed, *can*, because how much damage an
  image took is information the clean image never contained. The β sweep turns a
  null here into a specific claim: if no weighting of the auxiliary logit helps,
  the logit is **redundant with the main head**, which is a different and more
  interesting statement than "uninformative".

**Two controls guard the whole thing.** **E6** re-runs the reference arm with
degradation sampled live instead of read from the pre-rendered cache — the direct
answer to "the adapter memorised your twelve corruptions". **E7** asks the one
architectural question the PoC can pose: the plain adapter is nearly blind to
*which* trunk stage the damage entered at, so does telling it — via per-block
taps — recover more?

**What shape the answer is likely to take.** The honest headline here is bounded
rather than triumphant: a small, real, label-free gain, quoted against a ceiling
that was known in advance, with a clear account of which term produced it and
which two extensions did not. That is a better result than an unqualified number,
and it is why E0 runs third rather than last.

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

Ordered as `scripts/run_all.sh` runs them. The step numbers are that script's:
`bash scripts/run_all.sh --list` prints the same list, and `--from N` resumes at
one.

| step | # | Experiment | Asks | Cost | Configs / entry point |
|---|---|---|---|---|---|
| 1–2 | **P0/P1** | manifests, stage 0 | — | ~2–4 GPU-h | `probe/dinov3_ntire{,_crop}.yaml` |
| 3 | **P2** | the baseline | does retention actually collapse? | ~1–2 GPU-h | `runs/dinov3_poc_baseline.yaml` |
| 4 | **D1** | preprocessing confound | is the baseline reading forensics or *content*? | ~1–2 GPU-h | `runs/dinov3_poc_baseline_crop.yaml` |
| 5 | **E1** | identity adapter — **GATE** | does the split reproduce the baseline *exactly*? | one eval | `runs/dinov3_poc_identity.yaml` |
| 6 | **P3** | feature caches | — | hours, ~7 GB | `cache/dinov3{,_val,_val_hard}.yaml` |
| 7 | **E0** | drift geometry — **the ceiling** | how much of the drift can the head even see? | minutes, no GPU | `scripts/analyze_drift.py` |
| 8 | **S0** | the seed floor | how big is a difference that means nothing? | 5 × minutes | `train/dinov3_clean.yaml` × 5 seeds |
| 9–10 | **E5a/b** | stage 2, and the β sweep | does Δ carry signal, and can any β use it? | seconds + minutes | `train/dinov3_discrepancy.yaml`, `scripts/sweep_beta.py` |
| 11 | **E5c** | **the headline** | what retention came back? | one eval | `runs/dinov3_poc_grace.yaml` |
| 12 | **E2** | clean teacher | is the teacher the mechanism, or self-distillation? | minutes | `train/dinov3_degraded.yaml` |
| 13 | **E3** | loss ablation | Jacobian weighting vs plain MSE | minutes | `train/dinov3_plain_mse.yaml` |
| 14 | **E8** | the gate | is it learning, or is it being decayed open? | minutes | `train/dinov3_gate_nodecay.yaml` |
| 15–16 | **E9** | hyperparameter sweeps | is any knob outside the seed floor? | 6 × minutes | `train/dinov3_sweep_*.yaml` |
| 17 | **E4** | erasure trade-off | **does restoring destroy the evidence it restores?** | 6 × seconds | stage 2 vs every stage-1 checkpoint |
| 18–21 | **E7** | ladder / taps | does knowing *where* the damage entered recover more? | ~21 GB + minutes | `train/dinov3_ladder_final.yaml` |
| 22 | **E6** | cached vs live | is the finite epoch set being exploited? | **hours** — the slow arm | `train/dinov3_live.yaml` |

> **Findings live in [`docs/RESULTS.md`](RESULTS.md).** This document is how to
> run each arm; that one is what came back. It also carries the previous round's
> results and the predictions they imply, which is worth reading before you spend
> the GPU time — several arms have a known expected answer, and two of them came
> back the opposite way round from what the sections below predict.

### What gates what

**Three of these gate the others; the rest are independent.** The script enforces
the order, but the reasons are worth knowing when running an arm by hand.

- **D1 gates everything.** If the detector does not actually collapse under
  degradation — or collapses for the wrong reason — there is no damage for GRACE
  to repair and no adapter changes that.
- **E1 gates everything after it**, and it is the only *hard stop* in the script.
  If the null adapter does not reproduce the baseline to the last decimal, the
  trunk/head split is wrong and every later comparison is against a model nobody
  benchmarked. `compare.py --assert-identity` exits non-zero, and `set -e` in
  `run_all.sh` turns that into a full stop.
- **S0 gates every ablation verdict.** E2, E3, E8 and E9 are all "is arm X
  different from the reference?", and that question is unanswerable without the
  seed spread. Run it before reading any of them, not after.
- **E0 is not a gate, but read it before the retention numbers, not after.**
  `parallel_fraction` is the denominator that makes the headline interpretable,
  and one of its outcomes saves a day of work on stage 2.
- **E2, E3, E6, E8 and E9 do not depend on each other** and can run in any order.

```
P0 ─> P1 ─> P2 ─┬─> D1                          (is there a gap, and is it real?)
                │
                └─> E1 ══ GATE ══> P3 ─> E0     (exact instrument, known ceiling)
                                          │
                                          v
                                    S0 (5 seeds) ────────> the noise floor
                                          │
                    ┌─────────────────────┼──────────────────────┐
                    v                     v                      v
              E5a ─> E5b ─> E5c     E2 / E3 / E8 / E9      E7a ─> E7b ─> E7c ─> E7d
              (Δ, β, headline)      (attribution)          (the ladder)
                    │
                    └─> E4  (needs S0's step_*.pt AND E5a's stage-2 recipe)
                                                           E6  (slow control, last)
```

E5 sits before E2/E3 in the script for one practical reason: `dinov3_poc_grace.yaml`
scores `+grace` and `+grace-d` in a single pass over the eval set, and the
`+grace-d` arm needs a stage-2 checkpoint to exist. Training stage 2 first buys
the headline number several hours earlier, at no cost — stage 2 is seconds.

E1 does **not** sit in that run any more. It has its own run config, scored right
after the baseline, so that the check which decides whether any of this is
measurable fires *before* the work it guards rather than after all of it.

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
> and invalidates every feature in the cache. `run_all.sh` skips existing manifests
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
> turns E3's ablation and E4's six-point sweep from "in principle" into
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

## 5. E0 — drift geometry, and the ceiling

> **Prior round: significant, and small enough to bound the whole project.**
> Asymmetry 0.0051 on drifts of ~0.12, CI `[0.0041, 0.0062]` — it excludes zero
> because n = 277k, not because the effect is large. The number that matters is
> `parallel_fraction = 0.0298`: **97% of the drift is orthogonal to the direction
> the frozen head can see**, so 97% of any correction cannot move AUC. Carried by
> `gaussian_noise` and `resize`; `center_crop` is *negative*. See
> [RESULTS.md §4](RESULTS.md#4-findings-one-per-experiment).

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
R=../eval_pipeline/results       # reused by the compare.py blocks in §8 and §11
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
bit, whatever the gate or the severity conditioning happen to
be. The same trick means `β = 0` makes GRACE-D *identical* to GRACE at
initialization. Without that guarantee, any change in clean AUC would be
impossible to attribute.

---

## 7. S0 — the seed floor

**Asks:** how large does a difference have to be before it means anything?

**Why it comes before every ablation.** E2, E3, E8 and E9 all reduce to "is arm X
different from the reference?", and none of them can be answered by comparing two
numbers. Every one of those arms differs from the reference in one config key
*and* in nothing else — same data, same schedule, same epochs — so the only other
thing that could move a metric is the seed. Measure that first and every later
comparison has a yardstick; skip it and the ablation table is a list of
coincidences ranked by size.

The failure this prevents has already happened once here: a four-point geometry
sweep spanning 0.0009 AUC was read as a ranking, on a metric whose seed spread
turned out to be comparable to the whole span. Picking a winner from it was
reading noise.

**Two different error bars, and they answer different questions.**

| | measures | the right yardstick for |
|---|---|---|
| **seed spread** (this arm) | how much a metric moves when only `--seed` changes | *"is configuration A better than configuration B?"* |
| **sampling SE** (Hanley–McNeil, ~0.0081 on `ntire_val_hard`) | how precise an AUC is on a set this size | *"how precise is the number I am publishing?"* |

The seed spread is much the narrower of the two, and it is the correct one for
comparing arms trained on identical data. The sampling SE is the one to quote
next to the headline retention figure. Reporting only the second makes every
ablation look like noise; reporting only the first makes the headline look more
precise than it is.

**Run** (five stage-1 runs, minutes each — the seed replicates are CLI overrides
rather than five near-identical config files, which is what `--seed` and
`--run-id` are for):

```bash
python scripts/train_adapter.py configs/train/dinov3_clean.yaml
for s in 1 2 3 4; do
  python scripts/train_adapter.py configs/train/dinov3_clean.yaml \
    --seed "$s" --run-id "dinov3_clean_s$s"
done

python scripts/seed_stats.py dinov3_clean 'dinov3_clean_s*'
```

**Read:** the `sd` column on `held_out_images/ntire_val_hard`, metric
`auc_adapted`. That number is the unit every verdict in §8, §9, §14 and §15 is
quoted in. As a rule of thumb, two arms closer than about **2 sd** are the same
arm and should be reported as such rather than ranked.

`seed_stats.py --vs <run>` does the comparison directly, printing the gap next to
the spread:

```bash
python scripts/seed_stats.py dinov3_clean 'dinov3_clean_s*' --vs dinov3_plain_mse
```

**This run is also the reference arm itself.** `dinov3_clean` at seed 0 is the
checkpoint `configs/detectors/dinov3+grace.yaml` and `dinov3+grace-d.yaml` load,
and it is the one every ablation is one key away from. Its `step_*.pt` files are
E4's x-axis. So this step is not overhead attached to the ablations — it is the
main result, run five times.

## 8. E2 — the clean teacher

> **Prior round: confirmed, cleanly.** Arm A (`dinov3_degraded`) gains **−0.0000**
> AUC on both val sets — exactly nothing, as predicted. Arm B gains +0.0035
> (`ntire_val`) and +0.0126 (`ntire_val_hard`). The clean teacher is the
> mechanism. This is the one unambiguous positive result in the project.
>
> **Why it is being re-run.** Those numbers came from an objective that still
> carried the since-removed `lam_sw` and `lam_id` terms, and the pair was not
> symmetric in them: they were **1.65% of arm B's objective** and **0.000% of
> arm A's** — under `target_view: degraded` the adapter reproduces its input, so
> the distribution-matching and anchor terms are trivially satisfied (8e-11).
> Arm B therefore carried ~2% of objective that arm A did not, and the
> comparison was not in effect a one-key change.
>
> The conclusion should survive, because arm A gained *exactly zero* and a 2%
> term cannot move a floor — but "confirmed under a superseded objective" is
> not a reportable statement, and re-running it costs minutes. Expect the same
> answer; the point is being able to say it without a footnote.

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
Each entry carries `cosine_to_clean` and `gate`.

Neither is AUC, deliberately. Retention is measured by the eval harness, on the
eval split, through `grace.detectors.adapted`. What you get here is an in-loop
signal — enough to notice a run that helps L1 while wrecking L3 before it
finishes.

Two more keys are worth watching in `history[]`:

- **`gate`** — how strongly the adapter is applying its correction. ~~It starts
  at `0.018` and should climb to somewhere around 0.1–0.5.~~ **Superseded by E8.**
  On this data the gate barely leaves its init (0.018 → 0.021), and *all* of that
  movement is decoupled weight decay: with `decay_gate: false` it **falls** to
  0.0164. `gate` is not a health signal and the objective's net pull on it is
  downward. See [§14](#14-e8--the-gate).
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

## 9. E3 — the loss ablation

> **Prior round: the weighting wins on AUC, but not for the reason below, and the
> readout this section prescribes does not work.** `dinov3_plain_mse` finished at
> `cos_decision` **0.1072** — *twice* the weighted arm's 0.0655, not "near 0".
> Across the three arms `cos_decision` is **anti-correlated** with the outcome:
> the ratio-16 arm (which is now the reference) had the lowest alignment
> (0.0488) and the highest gain (+0.0148 hard, vs plain MSE's +0.0103). Read
> AUC, not `cos_decision`. The "Read:" paragraph at the end of this section is
> superseded.
>
> **Why it is being re-run.** Both arms predated the `lam_sw` / `lam_id`
> removal. The ablation itself stayed clean — those terms were 1.65% of the
> weighted arm's objective and 1.49% of plain MSE's, near-symmetric across the
> pair — but the *magnitudes* come from an objective that no longer exists, and
> the weighted arm sat at ratio 4 rather than the reference's 16. Both arms now
> carry the reference weights, so the pair is a true one-key change.

**Asks:** does the Jacobian weighting actually earn its place?

```
L = L_align + λ_kl·L_headKL + λ_sev·L_severity
```

Every term is label-free. One arm, **identical to `dinov3_clean.yaml` apart from
the one key being ablated.** Any other difference and it is not an ablation.

### Jacobian weighting off (`dinov3_plain_mse.yaml`)

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

So `weighting: none` makes `L_err` **provably** plain `F.mse_loss`, not
approximately, and `tests/test_losses.py` pins that equivalence.

This arm ablates the weighting **and nothing else** — every other loss key is
`dinov3_clean.yaml`'s, written out verbatim rather than left to the `LossConfig`
defaults, which differ from the tuned baseline in three keys (`w_err`, `w_cos`,
`eps_iso`). It is therefore not "the GRACE v1 objective": v1 also ran
`lam_kl: 0.5` and the since-removed `lam_sw` / `n_proj` / `lam_id` terms, and
adopting them here would confound the one key under test. A v1-objective baseline
is a legitimate thing to want, but it is a *separate arm*, not this one — and it
is no longer expressible in this config schema at all.

~~**Read:** `history[].cos_decision`, next to `dinov3_clean`'s. If this arm sits
near 0 while the weighted run climbs, that is the empirical case for the whole
weighting term.~~ **Superseded — see the result box at the top of this section.**
Plain MSE finished at *twice* the weighted arm's `cos_decision`. Read hard-split
Δ AUC instead, against the 0.00071 seed sd.

> `weighting: none` also switches off the `j` the loss uses, so `loop.py` builds
> a diagnostic Jacobian on logging steps when the loss did not need one.
> Without that, `cos_decision` would be missing from precisely the arm this
> figure is about.

**Run:**

```bash
python scripts/train_adapter.py configs/train/dinov3_plain_mse.yaml
```

It sets `wandb.group: e3_losses`, so with `--wandb` it lands alongside
`dinov3_clean` in one comparison instead of a flat list of runs.

---

## 10. E4 — the erasure trade-off

> **Prior round: the hypothesis is refuted on this data.** `auc_aux`'s deviation from
> chance **rises monotonically** with stage-1 progress — 0.029 → 0.310 on
> `ntire_val`, 0.001 → 0.138 on `ntire_val_hard`, across all six checkpoints.
> Restoration is *concentrating* the forensic signal here, not erasing it, so the
> sharpest critique of the approach does not land. The curve below is real but
> points the other way; the aux head's polarity is unstable (AUC dips below 0.5
> at steps 5,420 and 7,588), so plot |AUC − 0.5|, not AUC.
>
> **All six x-axis points are pre-`lam_sw`/`lam_id`-removal** (`dinov3_clean`'s
> checkpoints). Stage 2 itself is current. Since this experiment is *about* stage
> 1's objective, it is the arm where that matters most — re-run it after E2/E3.

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

## 11. E5 — GRACE-D

> **Prior round: no, and a β sweep says no weighting would.** Stage 2 trains
> fine and `auc_aux` reaches 0.855 standalone, so Δ *does* carry signal. But
> `auc_fused − auc_main` sits between −0.0008 and +0.0006 across every stage-2
> run, and `scripts/sweep_beta.py` bounds the fusion at **0.9917 retention over
> all β**, with the two val sets peaking at *opposite signs*. The aux logit is
> redundant with the main head, not uninformative. The harness run below is still
> needed for the record, but on the prior evidence it will not produce
> `exceeds_clean_ceiling`.
>
> `dinov3+grace-d.yaml` now names `checkpoints/grace/dinov3_disc/discrepancy.pt`,
> which is exactly what `configs/train/dinov3_discrepancy.yaml` writes, so the
> arm loads as soon as stage 2 has run. It was previously pointed at a run id
> that never existed, which is why no harness number was ever produced for it.

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

## 12. E6 — cached versus live

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

`batch_size` drops from 256 to 32 in this config because it now counts **images**
rather than cached feature vectors: the trunk is back in the loop, so the memory
cost is the detector's rather than the adapter's. Stage-1 validation still reads
pre-rendered caches either way.

---

## 13. E7 — the ladder

**Asks:** the plain adapter has to infer, from the pooled seam alone, which trunk
stage the damage entered at, and it is nearly blind to that (0.376 vs 0.896
nine-way transform ID; see `DEFAULT_TAP_BLOCKS` in `grace/splits/dinov3.py`). If
displaced evidence is recoverable at all, does knowing *where* it was displaced
from recover more of it?

**Prerequisite:** the tap caches, a separate render of about 21 GB. The cache
stores raw `(K, 768)` taps, so `tap_dim` costs stage-1 time and nothing else —
every rung of the sweep reads the same bytes. `tap_blocks` is the key that
*would* force a re-render, which is why it lives in `split_args` and is recorded
in the cache spec rather than restated per run.

```bash
python scripts/build_cache.py configs/cache/dinov3_taps.yaml
python scripts/build_cache.py configs/cache/dinov3_val_taps.yaml
python scripts/build_cache.py configs/cache/dinov3_val_hard_taps.yaml

# the ladder arm, at the reference arm's loss weights and the reference's epochs
python scripts/train_adapter.py configs/train/dinov3_ladder_final.yaml

# the tap_dim sweep. 64 IS dinov3_ladder_final, so there are three other rungs
python scripts/train_adapter.py configs/train/dinov3_sweep_tapdim_32.yaml
python scripts/train_adapter.py configs/train/dinov3_sweep_tapdim_128.yaml
python scripts/train_adapter.py configs/train/dinov3_sweep_tapdim_256.yaml

# stage 2, with and without the per-tap drift profile as an input
python scripts/train_discrepancy.py configs/train/dinov3_discrepancy_ladder.yaml
python scripts/train_discrepancy.py configs/train/dinov3_discrepancy_ladder_taps.yaml

# and through the harness, on the same denominator as every other arm
cd ../eval_pipeline && python scripts/run_eval.py --config configs/runs/dinov3_poc_ladder.yaml
```

**Read:** `tap_gate/block*` in `history` — how much the correction leans on each
block. That per-layer profile is the figure this arm exists to make, and it is
what a RINE `layers` split would have given for free. Read it knowing E8's
result: weight decay moves a gate on its own, so a `tap_gate` sitting near its
0.018 init is not evidence either way.

`dinov3_discrepancy_ladder_taps.yaml` sets `use_taps: true`, which feeds the
auxiliary head one log1p'd drift norm *per tapped block* instead of the single
pooled norm it otherwise sees. On a `vector` seam that is the difference between
the discrepancy branch's weakest possible form and something a null result would
actually say something about — see [§16](#16-what-the-poc-cannot-answer).

> **Status: inconclusive, and not yet a fair test.** The ladder runs on disk
> (`dinov3_ladder_e4`, `dinov3_sweep_tapdim_*`) stopped at 4,336 steps against
> the plain arm's 13,008, so their 66–71% hard gap-closed cannot be read against
> the plain arm's 87% — it is a third of the training, not a worse architecture.
> **`dinov3_ladder_final.yaml` is the arm that settles it and has not been run.**
>
> What did come back: `tap_gate` finished at 0.0188 / 0.0221 / 0.0213 / 0.0265 /
> 0.0199 across blocks 0/2/4/6/9 — all within a hair of the 0.018 init, and per
> E8 that drift is weight decay. No localization signal yet. `tap_dim` across
> {32, 64, 128, 256} spans 0.0114–0.0118 hard Δ AUC, i.e. nothing.

**`dinov3_ladder_final.yaml` is the only ladder config in the tree, and that is
deliberate.** A previous round also carried `dinov3_ladder.yaml` at the old loss
weights, and every ladder checkpoint that actually got trained came from a
4-epoch run being read against a 12-epoch plain arm. That comparison was never
legitimate, and the simplest way to keep it legitimate is for the config that
makes it possible not to exist. `epochs`, `seed` and the loss block here are
copied from `dinov3_clean.yaml` and must stay copied from it.

---

## 14. E8 — the gate

**Asks:** the gate is reported everywhere as evidence the adapter is "learning to
apply its correction". Is it learning, or is decoupled weight decay pulling
`gate_logit` toward zero (and so the sigmoid toward 0.5)?

```bash
python scripts/train_adapter.py configs/train/dinov3_gate_nodecay.yaml
```

`decay_gate: false` exempts `gate_logit` from weight decay and touches nothing
else. A global `weight_decay: 0.0` would also change the MLP and confound it.

**Read:** `gate` in `history`, against **`dinov3_clean`, the reference arm**, as
the control. `decay_gate` defaults to `true` and `gate_init` to `-4.0`, so the
reference *is* the decay-on, init-(−4) arm — there is no `dinov3_sweep_gate_-4`
config and adding one would only create a second name for the same run.

```bash
python scripts/seed_stats.py dinov3_clean --vs dinov3_gate_nodecay
```

> **Prior round: it was the optimizer.**
>
> | run | gate first → last |
> |---|---|
> | the reference arm (decay on) | 0.01799 → 0.02089 (**+0.0029**) |
> | `dinov3_gate_nodecay` (decay off) | 0.01799 → **0.01645** (**−0.0015**) |
>
> With decay off the gate **falls**. The objective's net pull on the correction
> magnitude is *downward* — the alignment term wants a smaller correction than
> the init. Corroborated by `dinov3_sweep_gate_-3`, which starts 2.6× higher and
> moves *less* (+0.0021), because a larger gate is decayed back harder.
>
> **Consequence: `gate` is not a health signal.** §8's guidance to expect it to
> "climb to somewhere around 0.1–0.5" is wrong on this data, and any reading of a
> rising gate as evidence of learning — including `tap_gate` in E7 — is measuring
> AdamW.

---

## 15. E9 — the hyperparameter sweeps

**Asks:** is any knob outside seed noise?

The unit of measurement is five-seed replicates of one config: hard-split Δ AUC
has **sd 0.00071**. Two arms closer than ~0.0015 are the same arm.

Six arms, one file each, every one of them `dinov3_clean.yaml` with **exactly one
key** changed:

```bash
# the loss-weight ratio -- the reference is ratio 16, so there is no wratio_16
python scripts/train_adapter.py configs/train/dinov3_sweep_wratio_0.25.yaml
python scripts/train_adapter.py configs/train/dinov3_sweep_wratio_1.yaml
python scripts/train_adapter.py configs/train/dinov3_sweep_wratio_4.yaml

# capacity, and the gate init -- the reference is n_blocks 3 / bottleneck 128 /
# gate_init -4, so those three rungs are the reference itself
python scripts/train_adapter.py configs/train/dinov3_sweep_nblocks_2.yaml
python scripts/train_adapter.py configs/train/dinov3_sweep_bottleneck_256.yaml
python scripts/train_adapter.py 'configs/train/dinov3_sweep_gate_-3.yaml'

# and read every one of them against the S0 family, not against a single run
python scripts/seed_stats.py dinov3_clean 'dinov3_clean_s*' \
  --vs dinov3_sweep_wratio_0.25 dinov3_sweep_wratio_1 dinov3_sweep_wratio_4 \
       dinov3_sweep_nblocks_2 dinov3_sweep_bottleneck_256 'dinov3_sweep_gate_-3'
```

The `tap_dim` axis in the table below belongs to E7 and is run there — it needs
the tap caches.

> **Prior round: everything is saturated except the loss-weight ratio, and that one
> flips sign between splits.**
>
> | axis | arms | hard Δ AUC range | verdict |
> |---|---|---|---|
> | `w_err`/`w_cos` ratio | 0.25, 1, 4, 16 | 0.0122–0.0149 | the only live axis |
> | bottleneck | 128, 256 | 0.0125–0.0126 | saturated |
> | `n_blocks` | 2, 3 | 0.0119–0.0126 | saturated |
> | `tap_dim` | 32, 64, 128, 256 | 0.0114–0.0118 | saturated |
> | gate init | −4, −3 | 0.0120–0.0132 | immaterial (E8) |
>
> On `ntire_val` the ratio preference **reverses** (ratio 4 gives +0.00324 vs
> ratio 16's +0.00287). Both directions are backed by five-seed families, so this
> is a measured split-dependence, not noise. Report it as one.

**The reference arm is a rung of three of these sweeps, and is not duplicated as
a separate config.** It is ratio 16 on the loss axis, `n_blocks: 3` /
`bottleneck: 128` on the capacity axes, and `gate_init: -4.0`. A previous round
carried `dinov3_sweep_wratio_16.yaml` *and* `dinov3_clean_final.yaml` as separate
files with identical contents, and reporting them as two arms would have been a
straightforward error. One config, one run, one row.

**Caveat to state when this family is reported.** The wratio rungs vary the ratio
but not the total weight (2.5 / 2.0 / 2.5 / 4.25), so scale and ratio are mildly
confounded across the family, and `align`'s share against `head_kl` and
`lam_sev` moves with it. That is how the sweep is defined and how its result
should be quoted; a scale-controlled version is a separate arm, not a fix to this
one.

---

## 16. What the PoC cannot answer

The PoC trunk is `DINOv3 ViT-S/16 (distilled), pooled → (B, 768)`, feature layout
`vector`. That buys a seam that cannot be wrong and a cache 32× smaller than
RINE's. Here is what it costs:

| | answerable here | why |
|---|---|---|
| D1 preprocessing | **yes** | it is a property of this detector |
| E0 drift asymmetry | **yes** | a magnitude comparison, layout-agnostic |
| E1 identity | yes, trivially | the seam is a construction, not a reconstruction |
| E2 clean teacher | **yes** | needs no particular layout |
| E3 loss ablation | **yes** | every term operates on the last axis |
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

## 17. Where every number lands

| artifact | written by | holds |
|---|---|---|
| `grace_adapter/checkpoints/probe/<run>/head.summary.json` | `train_probe.py` | P1: per-epoch train AUC, per-val-set AUC/acc, `selected_epoch` |
| `grace_adapter/results/dinov3_poc_drift.json` | `analyze_drift.py` | E0: `significant`, `asymmetry_ci`, `parallel_fraction`, by level and transform |
| `grace_adapter/checkpoints/grace/<run>/summary.json` | `train_adapter.py` | E2/E3/E6: `history` (gate, `cos_decision`, loss terms), `validation` on both held-out axes (alignment + `auc_*`/`acc_*`/`retention` per view), `val_history` (the same axes per epoch, when `val_every` is set) |
| `grace_adapter/checkpoints/grace/<run>/{ema,last}.pt` | `train_adapter.py` | the adapter. `ema.pt` is what the detector configs load |
| `grace_adapter/checkpoints/grace/<run>/step_*.pt` | `train_adapter.py` (`checkpoint_every`) | E4's x-axis |
| `grace_adapter/checkpoints/grace/<run>/summary.json` (stage 2) | `train_discrepancy.py` | E4/E5: `beta`, `auc_main`/`auc_aux`/`auc_fused` |
| `eval_pipeline/results/{run_id}__{detector}__{dataset}.json` | `run_eval.py` | D1/E1/E5: per-level and per-condition AUC, retention, `operating_envelope` |
| `grace_adapter/results/dinov3_beta_sweep.json` | `sweep_beta.py` | E5: `auc_fused` and retention at every β, i.e. the bound on the fusion over *all* weightings |
| `grace_adapter/results/compare_*.json` | `compare.py` | the baseline-normalized retention table and `exceeds_clean_ceiling`, per arm |
| stdout | `seed_stats.py` | S0: mean / sd / range per axis and metric, and the `--vs` gap against that spread |

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

## 18. The whole path in one command

```bash
cd grace_adapter
bash scripts/run_all.sh                # every experiment, in order
bash scripts/run_all.sh --list         # print the 22 steps and exit
bash scripts/run_all.sh --smoke        # 2 epochs, 2 cache views -- proves wiring
bash scripts/run_all.sh --from 8       # resume at step 8
bash scripts/run_all.sh --only 17      # just E4
bash scripts/run_all.sh --skip-slow    # everything except E6, the live control
WANDB=1 bash scripts/run_all.sh        # every stage tracked under one group
```

**Twenty-two steps, and every one of them is idempotent.** Manifests are skipped
rather than rebuilt (rebuilding one invalidates every feature cached against it),
the cache resumes at view granularity, a training run is skipped if its
`summary.json` exists, and a harness run is skipped if its result JSON exists. So
an interrupted run is resumed by re-running the same command; `--force` re-does
everything anyway.

It covers **every arm in this document** — D1, E0 through E9, S0, and both
prerequisites. There is no second script and no "extra arms" list to run by hand.

### The one hard stop

Step 5 is E1, and it runs `compare.py --assert-identity`, which exits non-zero
unless the null adapter reproduces the baseline exactly. Under `set -e` that ends
the run. It is the only place the script refuses to continue, and it is
positioned before the cache render — the most expensive step — precisely so that
a wrong trunk/head split costs one eval pass rather than a day.

### Order of information, not order of convenience

The first seven steps cost real GPU time and train nothing GRACE touches. They
are there because each can end the project:

| step | can end the project by showing |
|---|---|
| 3 (P2) | retention does not collapse — there is no gap to close |
| 4 (D1) | neither arm collapses — the dataset separates on *content* |
| 5 (E1) | the seam is wrong — nothing downstream is a measurement |
| 7 (E0) | the drift the head can see is ~0 — the ceiling is the finding |

Step 8 (S0) then produces the yardstick, and steps 9–11 produce the headline. By
step 11 you know whether there was a problem, whether the instrument is sound,
what the ceiling was, how big a real difference has to be, and what came back.
Everything after that explains it.

### Reading the output

The script prints a closing map of what to read and in what order — the argument
from [§0](#0-the-argument-these-experiments-make), pointed at the files it just
wrote. Read that, not the file list.


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
