# Experiments

What has been run, what is planned, and why each one exists. This is the reasoning
behind the numbers, not the numbers themselves — results live in `results/`,
`checkpoints/`, and the `wandb/` logs.

The project tests one claim: a detector fit on *clean* images collapses under
degradation (JPEG, blur, resize), and that collapse is repairable by a small,
label-free module that corrects the detector's internal feature vector back
toward what a clean image would produce (`GRACE`), optionally supplemented by a
frequency branch that re-reads the image in a basis the resize destroyed.

Read this document together with `PIPELINE.md` (what the model *is*) and
`DATA.md` (what it was trained on). The experiment arc below is ordered the way
the work was meant to be done: establish the denominator, gate the pipeline,
fit the head, then repair it — and only then ablate the repair.

> **Scope note.** Stage 2 — the frequency enricher and all of its ablations
> (`dinov3_enrich*`, the band / mask / position / top-k / anchor / aux-fuse
> sweeps, and the native-resolution frequency read) — is intentionally *not*
> documented here yet. It will be added once the set of ablations worth
> reporting is settled.

---

## 0. The overarching structure

Every experiment is one of a few kinds, and the ordering matters:

1. **Data audits** — descriptive statistics of the corpus, not model runs. They
   set the training protocol (crop range) and establish the floors a detector
   must beat before its number means anything.
2. **Stage 0 — probe** — fit the frozen DINOv3 trunk's classification head on
   clean features. This is the only thing that "trains a detector"; everything
   after operates on top of it.
3. **Baselines** — the unadapted detector's retention curve. The denominator
   every GRACE arm is compared against.
4. **Gates** — cheap runs that must pass before downstream work is trusted
   (the identity adapter, the falsification checks).
5. **Stage 1 — GRACE adapter** — the reference arm and its ablations.
6. **Stage 2 — frequency enricher** — deferred (see scope note).

### Shared evaluation protocol

Unless noted, a "retention curve" is measured on the held-out benchmark
`wildfake_coco_dalle3` (COCO val2017 reals + DALL-E 3 fakes — neither stratum
appears in training), over the standard degradation sweep (all four severity
levels, all eleven transforms), with retention defined as the degraded AUC
normalised by the detector's *own* clean AUC. See
`eval/configs/runs/dinov3_poc_baseline.yaml` for the definition and
`scripts/compare.py` for why retention is normalised against the *baseline's*
clean AUC rather than each arm's own.

One evaluation arm is used:

- **crop200** — a 200×200 window at native pixel scale. In distribution for a
  multi-scale-trained head, and the informative arm. Every real in the benchmark
  is exactly 200×200, so nothing is upsampled and the dimension shortcut is
  0.5 by construction.

---

## 1. Data audits

### `audit_sizes` — the crop range carries no label

`python scripts/audit_sizes.py --config load_data/configs/datasets/wildfake_train.yaml`

Multi-scale cropping exists to remove the resolution shortcut (every benchmark
real is 200×200, every fake is 1024px+). But cropping removes it only if the
*crop size* doesn't reintroduce it: a window can't be larger than its source,
so if one class's sources are systematically smaller, the realized crop size
becomes a classifier. This script simulates the draw and reports, for each
candidate `s_max`, the realized-size AUC (`E-cropsize`), recommending the
largest range that stays at chance.

- **Rationale**: the crop range is not a hyperparameter to pick from a plan; it
  is read off the corpus. The result (`recommended_s_max`) is written into every
  crop-bearing config (checked by `tests/test_configs.py`), producing the
  `s_max: 256` you see throughout. Ran on `wildfake_train` for the training
  protocol; the benchmark is characterized separately to fix arm (a)'s size.

### `shortcut_baseline` — the container floor (`E-shortcut`)

`python scripts/shortcut_baseline.py --dataset load_data/configs/datasets/wildfake_coco_dalle3.yaml`

Fits a logistic regression on four numbers per image — width, height, byte size,
bits-per-pixel — without decoding a pixel. It reports what the *file* alone
gives away.

- **Rationale**: on the raw benchmark, `max(w, h)` alone scores AUC 1.0000, so
  any detector reported without this number beside it is reporting an unknown
  mixture of forensics and file metadata. The dimension features die under both
  evaluation arms (every image gets identical dimensions); bits-per-pixel
  survives into the pixels and is the channel a frequency branch is most likely
  to read by accident. This number is the floor a real result must clear.

---

## 2. Stage 0 — the probe (fitting the head)

A frozen DINOv3 ViT-S/16 has no classifier, so GRACE has no "seam" to adapt
until one is fit. Each probe run trains a small MLP head on clean features only
(no degradation augmentation), then selects an epoch on the held-out validation
split.

### `dinov3_wildfake_multiscale` — the head everything downstream loads

`python scripts/main/train_probe.py train/configs/probe/dinov3_wildfake_multiscale.yaml`

The crop-era head, fit under the multi-scale protocol (128–256px windows,
`input_mode: multiscale`). This is the head the stage-1 adapter and the cache
render all load.

- **Rationale**: `s_max` is deliberately left blank in the config and filled
  from `audit_sizes` — a window larger than a source can't be taken, so an
  unaudited range scored `E-cropsize` 0.9895 on the benchmark, almost exactly
  the shortcut the crop was introduced to remove.

---

## 3. Baselines — the denominator

### `dinov3_poc_baseline` — unadapted retention (multiscale arm)

`python scripts/main/run_eval.py --config eval/configs/runs/dinov3_poc_baseline.yaml`

The unadapted detector's retention curve on the benchmark. This is the number
every GRACE arm is compared against, and it comes out of this file alone so
nothing re-derives it.

- **Rationale**: run *before* training an adapter. If retention doesn't collapse
  here, there is no gap for GRACE to close and the PoC has answered its own
  question early (~300k forwards vs ~4.2M for the stage-1 cache).

### `dinov3_poc_baseline_arms` — the crop200 evaluation arm

`python scripts/main/run_eval.py --config eval/configs/runs/dinov3_poc_baseline_arms.yaml`

The same unadapted detector scored on the `crop200` arm — a 200×200 window at
native pixel scale, in distribution for a multi-scale-trained head, and the
informative arm for frequency. (The `r512` robustness arm was dropped: it was
out of distribution and spectrally confounded, and the informative arm answers
the question on its own.)

- **Rationale**: this run decides the project's first real measurement: if
  retention doesn't collapse on `crop200`, the frequency branch has nothing to
  enrich and the finding is reported *before* anything is trained.

---

## 4. `dinov3_poc_identity` — the null adapter (E1)

`python scripts/main/run_eval.py --config eval/configs/runs/dinov3_poc_identity.yaml`
(read with `scripts/compare.py --assert-identity`)

The null adapter (`checkpoint: null`) — the trunk/head split wired up with no
trained weights — scored on its own. It must reproduce
`dinov3_poc_baseline` to the last decimal.

- **Rationale**: E1 is a *gate* on the whole project. It asks whether
  `head(adapter(trunk(x)))` with a null adapter equals `detector(x)` exactly;
  if not, the trunk/head split is wrong and every adapted number afterwards is
  measured against a model nobody benchmarked. A gate has to fire before the
  work it guards, so it lives in its own run and is scored right after the
  baseline, before the cache render.

---

## 5. Stage 1 — the GRACE adapter

The adapter is trained label-free: it maps degraded trunk features back toward
clean ones through a frozen head, with a small MLP "severity head" estimating
how far each image has drifted. See `PIPELINE.md` for the mechanism.

### `dinov3_multiscale` — the reference arm

`python scripts/main/train_adapter.py train/configs/train/dinov3_multiscale.yaml`

The reference arm under the multi-scale protocol, trained on windowed pixels at
native scale (128–256px). Every stage-1 ablation in this family is this file
with one key changed, and every detector config loads this run's `ema.pt`
(`checkpoints/grace/dinov3_multiscale/ema.pt`). The pre-multiscale resize arm
(`dinov3_clean`) was dropped with the detectors it trained on.

- **Rationale for the re-render**: the trunk now sees 128–256px windows at
  native pixel scale — a different feature space, not a harder version of the
  same one. An adapter trained on whole-image features has learned a mapping
  between pixels nothing downstream will ever show it again.
- **Parameter budget**: `bottleneck: 128` (~0.4M params against a 21M frozen
  trunk) *is* the claim — if the gap only closed with a large adapter, the
  evidence was merely displaced, not repaired.

### `dinov3_poc_grace` — the adapted detector vs the baseline

`python scripts/main/run_eval.py --config eval/configs/runs/dinov3_poc_grace.yaml`

`head(adapter(trunk(x)))` scored under the same conditions as the baseline, with
retention normalized by the *baseline's* clean AUC (`compare.py` refuses two
result files from different eval sets, so the dataset list is pinned to the
baseline's).

- **Rationale**: restoration can't exceed the clean-feature score, so retention
  ≤ 1.0 by construction — the ceiling the frequency branch exists to exceed.

---

## 6. Stage-1 ablations

Each ablation is `dinov3_multiscale` with exactly one key changed, so a diff
against the reference shows the single variable. They exist to answer
"does the mechanism earn its complexity, or is the repair just noise?"

### Loss — `dinov3_plain_mse` (Jacobian weighting off)

`python scripts/main/train_adapter.py train/configs/train/dinov3_plain_mse.yaml`

`loss.weighting: none` makes the error term exactly `F.mse_loss` (pinned by
`tests/test_losses.py`), against the reference's Jacobian-weighted error through
the frozen head.

- **Rationale**: the Jacobian weighting is a real design choice, not a detail —
  it re-weights the feature error by how much each feature moves the head's
  decision. If dropping it costs nothing, the weighting was never doing work.

### Capacity — `dinov3_sweep_bottleneck_256` and `dinov3_sweep_nblocks_2`

`python scripts/main/train_adapter.py train/configs/train/dinov3_sweep_bottleneck_256.yaml`
`python scripts/main/train_adapter.py train/configs/train/dinov3_sweep_nblocks_2.yaml`

Doubling the bottleneck (256 vs 128) and dropping from three residual blocks to
two, respectively.

- **Rationale**: the parameter budget is the claim (see §5). These two runs test
  whether the reference is at the small end of a curve that still has slope, or
  at a plateau. If the larger adapter gains nothing, the small one isn't a
  concession.

### Gate — `dinov3_sweep_gate_-3` and `dinov3_gate_nodecay`

`python scripts/main/train_adapter.py train/configs/train/dinov3_sweep_gate_-3.yaml`
`python scripts/main/train_adapter.py train/configs/train/dinov3_gate_nodecay.yaml`

Two distinct questions about the gate:
- `gate_init -3` (sigmoid ≈ 0.047) vs the reference's `-4` (≈ 0.018) — read
  `grad_norm`/`gate` over the first epoch, not final AUC, because the claim is
  about the early transient.
- `decay_gate: false` exempts `gate_logit` from weight decay, leaving everything
  else untouched.

- **Rationale**: the adapter starts as the identity and is supposed to learn to
  open the gate. Both runs check the alternative explanation — that the gate is
  merely being *pushed* open by weight decay or initialization rather than
  *learning* to open in response to the data. If the result is the same, the
  mechanism is a bookkeeping artifact.

### Loss ratio — `dinov3_sweep_wratio_{0.25,1,4}` (16 is the reference)

`python scripts/main/train_adapter.py train/configs/train/dinov3_sweep_wratio_0.25.yaml` (and `_1`, `_4`)

Sweeps the `w_err / w_cos` ratio over 0.25 / 1 / 4, with 16 being
`dinov3_multiscale` itself.

- **Rationale**: the ratio balances error magnitude against direction
  (cosine). The reference sits at the tuned point from earlier work; the sweep
  checks whether it's actually optimal on this corpus or merely inherited.

---

## 7. Seed variance — `seed_stats`

`python scripts/seed_stats.py dinov3_multiscale_final_s*`

Aggregates repeated runs into mean / std / CI and says whether a gap is real.

- **Rationale**: every number in the project comes from `seed: 0`, which is fine
  for a deterministic reproduction check but says nothing about how much a
  metric moves when only the seed changes. A sweep can't be read without that
  noise floor — an earlier geometry sweep spanned 0.0009 AUC on a set whose
  marginal standard error is 0.0081, which is picking a winner out of noise.
  `seed_stats` measures seed variance directly (for comparing configurations);
  the Hanley-McNeil sampling error of the val set itself is the right yardstick
  for the number you finally report.

---

## Pending — Stage 2 (frequency enricher)

To be documented separately. Includes the base enricher (`dinov3_enrich`), the
native-resolution frequency read (`dinov3_enrich_nativefreq`), and the ablation
family (band count, mask/position/top-k, anchor, aux-fuse, head fine-tuning).
Left out until the set of results worth reporting is fixed.
