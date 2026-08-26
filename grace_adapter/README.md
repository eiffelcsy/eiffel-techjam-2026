# GRACE — Gated Residual Adapter for Clean-feature Estimation

A tiny adapter, trained without labels, that maps a frozen detector's features of
a **degraded** image back onto its features of the **clean** image — and a second,
supervised variant that reads the correction itself as forensic evidence.

```
GRACE     logit = head( adapter(trunk(x)) )                        label-free
GRACE-D   logit = head( adapter(trunk(x)) ) + β · aux(Δ, severity)  + labels
                                              Δ = adapter(f_deg) − f_deg
```

The detector is never fine-tuned, never re-trained, and never even loaded
differently. The adapter is spliced at its trunk/head seam and the adapted model
re-enters the sibling evaluation harness (`../eval_pipeline`) as an ordinary
`FrozenDetector`, so the baseline and the adapted numbers come out of the same
code path, the same conditions, and the same result schema.

---

## 1. The two claims

`../eval_pipeline` measures the problem: detectors lean on local high-frequency
traces, and JPEG, blur and resize destroy exactly those. Retention —
`(auc_deg − 0.5) / (auc_clean − 0.5)` — is the number that collapses.

**GRACE's claim is that the evidence is displaced, not destroyed.** The degraded
features still carry the discriminative signal; it has moved somewhere the frozen
head does not look. If so, a ~2M-parameter residual correction recovers a real
slice of the retention gap. If a large adapter turns out to be needed, the claim
is false and the result is far less interesting — which is why the parameter
budget is a design constraint, not a convenience.

**GRACE-D's claim is that the damage is itself a signal.** RA-Det found that
generated images drift further in embedding space under perturbation than real
ones do. An adapter trained purely to *erase* drift is therefore destroying
forensic evidence while its reconstruction loss falls, and doing so
asymmetrically. So keep the quantity the adapter already computes: Δ is its
estimate of the drift, available at test time without the clean image, as a
by-product of a module that was running anyway. RA-Det needs a second forward
pass on a deliberately perturbed image to get the same signal.

This also breaks the restoration ceiling. A perfect restorer can at best recover
the clean-image score — retention 1.0. A fused score that reads Δ can exceed it,
because the *magnitude of the damage* is information the clean image does not
contain.

| | Adapter | Aux head | Labels | Ceiling |
|---|---|---|---|---|
| **GRACE** | trained | — | none | retention ≤ 1.0 |
| **GRACE-D** | *bit-identical, frozen* | trained | yes | may exceed 1.0 |

Stage 2 never touches the adapter, so both variants ship the same weights and
"the adapter is trained without labels" stays literally true. That separation is
not tidiness — it is what makes the erasure question testable (§7, E4).

---

## 2. Why the teacher is free

The trunk is frozen and the clean image never changes, so **the clean features
are constant**. Compute them once, ever, to disk. The teacher then never appears
in the training loop at all; it is a lookup.

## 3. The extension: cache the degraded side too

The same argument applies one step further out.

`pipeline.degrade.conditions.Condition` draws every recipe from
`stable_seed(index, level, replicate, seed)` — a blake2b hash, never a global RNG
counter — so a degraded view is *also* a pure function of (image, condition). And
the harness already has a field whose entire purpose is "an independent re-draw
over the same images": `replicate`.

```
epoch  ≡  replicate
```

Epoch 7's degradation of image 412 is computable now, on any machine, without
having run epochs 0–6 — the precondition for rendering every epoch **offline,
ahead of time** and storing the degraded features next to the clean ones.

> In `source: cache` mode the training loop contains **no trunk forward at all**.
> A step is two memmap reads and a 2-layer MLP.

That changes which experiments are affordable, not just how fast one is. Stage 2
in particular costs *seconds*, which is what makes the E4 sweep over stage-1
checkpoints practical rather than theoretical.

### 3.1 What it costs

Per image, per view, float16 (one view = clean, or one epoch):

| Detector | Layout | Shape | Bytes/image/view | 13 views |
|---|---|---|---|---|
| GAPL | `vector` | (1024,) | 2 KB | 26 KB |
| B-Free | `tokens` | (5, 768) | 7.5 KB | 98 KB |
| RINE | `layers` | (24, 1024) | 48 KB | **624 KB** |

At 100k training images: GAPL ≈ 2.6 GB, B-Free ≈ 10 GB, RINE ≈ **62 GB**. Always
run `build_cache.py --dry-run` first.

Compute, for `R` runs over `E` epochs and `N` images: degrading in the loop costs
`R × N × E` trunk forwards, every run; pre-rendering costs `N × (E + 1)`, once,
resumably. Break-even is the first run.

### 3.2 What it costs you

Pre-rendering fixes the augmentation set at `E` draws per image. Three
mitigations, all built in:

1. `E ≥ 8`, with a fresh recipe per (image, epoch) — not per image.
2. **Held-out degradations.** Validation epochs are numbered from
   `VAL_EPOCH_OFFSET = 10_000`, so their `replicate` can never collide with a
   training epoch's. Disjoint draws from the same distribution.
3. **`source: live`** (`configs/train/rine_live.yaml`) degrades in the loop with
   the same schedule and is the direct control. Cached ≈ live means the finite
   epoch set is not being exploited.

---

## 4. The objective

```
L = L_align + λ_sw·L_SW + λ_id·L_identity + λ_kl·L_headKL + λ_sev·L_severity
```

Every term is label-free. Only stage 2's BCE uses image labels.

### 4.1 Spend capacity where it changes the decision

Plain MSE treats every feature direction as equally worth fixing. The head does
not: it maps features to one scalar, so only the error inside its sensitive
subspace can move AUC, and everything orthogonal is capacity spent on nothing an
AUC can see. Its Jacobian is a gradient vector the shape of the feature:

```
j = ∇_f h(f) |_{f = f_clean}
L_err = (1−ε)·mean[(ĵ·e)²]  +  ε·mean[e²]          e = f_adapted − f_clean
```

**For a linear head `j` is exactly the constant `w`, so one implementation covers
both linear and MLP heads with no branch** — the reason to express it this way
rather than as "project onto w". Written as a blend so `ε = 1` is *exactly*
`F.mse_loss`: the plain-MSE ablation is one config key and provably the same
objective GRACE v1 had (`tests/test_losses.py`).

`head_kl` is the exact version of the first term — a finite difference through
the real head rather than a first-order expansion — but it only ever sees the
scalar. It is kept and demoted to `λ_kl = 0.1`.

**The diagnostic that motivates this empirically**: `cos(Δ, j)`, logged every 50
steps. If a plain-MSE run sits near 0, the adapter is spending nearly all its
capacity on directions the head cannot see.

### 4.2 Distribution matching and posterior sampling are one feature

Point-wise alignment is satisfied by a **conditional mean**, which is
systematically under-dispersed: the corrected batch forms a tighter cloud than
real clean features do, and the frozen head's operating point was calibrated on
the wider one. Sliced-Wasserstein fixes that — random projections, sort, L2; six
lines, one hyperparameter, no adversarial stability risk. Because both batches
hold the same images it is a *matched* comparison, stronger than the usual form.

Posterior sampling gives the adapter a noise input `z`, draws k corrections, and
averages **logits, not features** — `E[h(f)] ≠ h(E[f])` for any nonlinear head,
so this is cheap Monte-Carlo posterior averaging, k adapter passes (microseconds)
against one trunk pass.

> **These are not two features.** Under point-wise reconstruction losses alone the
> optimal stochastic policy is to ignore `z` — posterior collapse. Noise earns its
> keep *only* because the SW term rewards matching the spread that a conditional
> mean under-disperses. Shipping `noise_dim > 0` with `lam_sw: 0` buys parameters
> that do nothing, which is why `rine_no_sw.yaml` and `rine_posterior.yaml` are
> run as a pair. `posterior_spread` in the validation block is the tripwire; if it
> reads ~0 that is a reportable negative result about the objective, not a bug.

### 4.3 Severity is free, and does not cost the label-free claim

Transform grids are ordered mild → severe, so a step's severity is its
parameter's normalised rank within its own grid, combined with composition depth.
The degradation sampler already knows both, so the target is written into
`recipes.parquet` at render time at zero marginal cost. **The labels are the
sampler's own metadata, not image labels.**

The scalar FiLM this feeds is deliberately thin — it is what `models/prompts.py`
(FUTURE) replaces with a soft mixture over learnable degradation prompts.

---

## 5. The adapter — one class, because layout is a gate shape

```
y = f + g ⊙ MLP(LN(f)),    g = sigmoid(gate_logit),    gate_logit init −4
```

Everything operates on the last axis, so `(B, D)`, `(B, T, D)` and `(B, L, D)` run
through the same code with the same weights shared across the group axis. Only
the gate shape differs — `(D,)` for vector and tokens, `(L, D)` for layers — and
the factory picks it. There is no class hierarchy and nothing else in the project
branches on layout.

The `(L, D)` gate is also the interpretability output: mean it over `D` and you
have how much correction each encoder block needs, per degradation.

**Identity at initialization, exactly.** The last projection of every block is
zero-initialised, so the adapter returns its input bit-for-bit whatever the gate,
the noise, or the severity conditioning happen to be — and the same
zero-initialisation trick makes `β = 0` mean GRACE-D *is* GRACE at init. Without
this, a clean-AUC change is unattributable.

Log `gate().mean()`. It should climb off 0.018 and plateau around 0.1–0.5.
Saturating at 1.0 is over-correction; sitting at init means the alignment term is
too weak against the identity term.

---

## 6. Package map

Laid out like the sibling harness: a descriptive project directory holding a
short importable package, so `eval_pipeline/` → `import pipeline` and
`grace_adapter/` → `import grace`. Every path below is relative to
`grace_adapter/grace/`.

```
grace/
├── splits/          the trunk/head seam, added AROUND detectors
│   ├── base.py        FeatureSpec, SplitDetector, assert_frozen
│   ├── verify.py      head(trunk(x)) == detector(x), checked at construction
│   ├── rine.py bfree.py gapl.py    reconstruct a seam inside a vendored repo
│   └── dinov3.py      delegates to a seam that is already there            ← §8
├── probe/train.py   stage 0 — fit the PoC detector's own head. PoC ONLY.  ← §8.2
├── cache/
│   ├── schedule.py    (index, epoch) -> degradation, as a pure function  ← §3
│   ├── spec.py        four fingerprints + the FUTURE `taps` field
│   ├── writer.py      offline render: clean view + one view per epoch
│   └── reader.py      memmap random access, per-worker, by manifest index
├── models/
│   ├── adapter.py     GatedResidualAdapter (+ noise, + severity FiLM)
│   ├── severity.py    SeverityHead — target is free from recipes.parquet
│   ├── discrepancy.py DiscrepancyHead + FusedHead                        ← §1
│   ├── factory.py     the only layout branch, three lines
│   ├── ladder.py      FUTURE — blueprint only
│   └── prompts.py     FUTURE — blueprint only
├── train/
│   ├── weighting.py   head_gradient, decision_weighted_error             ← §4.1
│   ├── losses.py      alignment, sliced-Wasserstein, identity, KL, severity
│   ├── diagnostics.py cos(Δ,j), drift asymmetry, posterior spread
│   ├── data.py        CachedPairDataset | LivePairDataset — one config flag
│   ├── tracker.py     W&B as a null object — off by default, never fatal   ← §9
│   └── loop.py        stage 1 (label-free) and stage 2 (supervised)
└── detectors/adapted.py   AdaptedDetector — a FrozenDetector, for the harness
```

Configs, four kinds, one directory each:

```
configs/
├── probe/        PoC only — fit a frozen detector's own classification head
├── cache/        what to render, for which detector
├── train/        one stage-1 or stage-2 run against a cache
├── detectors/    the ADAPTED detector, in the HARNESS's config shape
└── defaults.yaml annotated reference: every key with its default. Never loaded.
```

Detectors and datasets are **never redefined here**. Configs reference
`../eval_pipeline/configs/` by path, so each is described in exactly one place and
GRACE cannot drift from what was benchmarked. The PoC detector obeys the same
rule: `DINOv3MLPDetector` lives in `pipeline/detectors/dinov3.py` with the rest
of the zoo, and `grace/` contributes only its seam and the script that fits its
head.

---

## 7. The pipeline, and the experiments

```bash
# 1. render, once per detector -- resumable at view granularity
python scripts/build_cache.py configs/cache/rine.yaml --dry-run
python scripts/build_cache.py configs/cache/rine.yaml

# 2. the premise, before anything is TRAINED on it -- reads the cache, no GPU
python scripts/analyze_drift.py --cache cache/rine --dataset ../eval_pipeline/configs/datasets/ntire_train.yaml

# 3. stage 1, minutes per run
python scripts/train_adapter.py configs/train/rine_clean.yaml

# 4. stage 2, seconds per run
python scripts/train_discrepancy.py configs/train/rine_discrepancy.yaml

# 5. score through the SAME harness that produced the baseline
cd ../eval_pipeline && python scripts/run_eval.py --config configs/runs/<run>.yaml
python scripts/compare.py --baseline <baseline.json> --adapted <adapted.json>
```

**The render comes first, and E0 is second.** `analyze_drift.py` opens the cache
in its first statement and reads clean *and* degraded features out of it; it
exits with `no rendered epochs under <dir>` if no degraded view has been
finalized. "E0 first" means *before anything is trained* — the render is the
step E0 itself is built on, not one of the things it decides.

| # | Arm | Config | Asks |
|---|---|---|---|
| E0 | drift analysis | `scripts/analyze_drift.py` | does RA-Det's asymmetry hold here? |
| E1 | identity | `detectors/rine+identity.yaml` | does the split reproduce Day 1 *exactly*? |
| E2 | A vs B | `train/rine_degraded.yaml` / `rine_clean.yaml` | does the clean teacher buy retention? |
| E3 | loss ablations | `rine_plain_mse` / `rine_no_sw` / `rine_posterior` | Jacobian vs MSE; ±SW; ±noise |
| **E4** | **erasure trade-off** | stage 2 vs every stage-1 checkpoint | **does the adapter destroy evidence?** |
| E5 | GRACE-D | `detectors/rine+grace-d.yaml` | does the fused score beat retention 1.0? |
| E6 | cached vs live | `train/rine_live.yaml` | is the finite epoch set being exploited? |

**E0 is the first thing to look at, and both outcomes are useful.** Asymmetry present → the
discrepancy branch has signal and the label-free objective is knowingly discarding
it. Asymmetry absent → stage 2 will be weak here; say so, keep the restoration
result, and save a day. The parallel/orthogonal decomposition matters as much as
the gap: drift that is large but orthogonal to the decision direction is invisible
to the frozen head, which is exactly why an auxiliary head can recover signal the
main head cannot — and why the main loss can fall while evidence is destroyed.

**E4 answers the critique directly.** `rine_clean.yaml` sets `checkpoint_every: 2`;
train stage 2 against each intermediate checkpoint and plot the auxiliary head's
standalone AUC against stage-1 progress:

```bash
for ck in checkpoints/grace/rine_clean/step_*.pt; do
  python scripts/train_discrepancy.py configs/train/rine_discrepancy.yaml \
    --adapter "$ck" --run-id "e4_$(basename "$ck" .pt)"
done
```

A falling `auc_aux` as stage 1 improves is direct evidence that restoring features
erases forensic evidence, and the retention-versus-drift-preservation curve is the
figure. It costs almost nothing because stage 2 is seconds.

Every arm above has a PoC twin: swap `rine` for `dinov3` in the config name and
it runs today, in seconds, on a detector whose seam cannot be wrong. See §8.4 for
which of these the PoC can actually answer and which it can only gesture at.

**Run E1 first regardless.** If the identity adapter does not reproduce the Day-1
JSON to the last decimal, the split is wrong and everything downstream compares
against a model that was never benchmarked.

`scripts/compare.py` reports retention against the **baseline's** clean AUC, so
`> 1.0` means what it should — the harness's own retention divides by each
detector's own clean AUC, which for GRACE-D would hide exactly the effect being
claimed.

---

---

## 8. The proof-of-concept path: DINOv3 ViT-S/16 + an MLP head

Everything above describes adapting *somebody else's* detector, and section 11
says why none of it can be run yet: the three zoo splits compose modules from
repos cloned by hand under `third_party/`, and `RINESplit._head_forward` is
written against documented structure rather than against a clone. Until those
exist, a retention number could come from a wrongly composed head and nothing in
the curve would say so.

So there is a fourth detector, built here rather than downloaded, whose only job
is to make the seam not a question:

```
trunk  frozen DINOv3 ViT-S/16 (distilled), pooled  -> (B, 768)   layout "vector"
GRACE  the adapter, spliced at that seam           -> (B, 768)
head   LayerNorm -> Linear -> GELU -> Linear       -> (B,)  one logit
```

`head(trunk(x)) == detector(x)` holds because `DINOv3MLPDetector.forward` *is*
`self.head(self.trunk(x))`, and `DINOv3Split` delegates to both rather than
reconstructing either. E1 — the identity adapter reproducing the baseline
exactly — becomes a tautology instead of a nail-biter, which is precisely what
you want from the arm you debug the rest of the pipeline in.

### 8.1 Why this backbone, and what it costs

`facebook/dinov3-vits16-pretrain-lvd1689m` is the ViT-S/16 **distilled from the
ViT-7B teacher** on LVD-1689M, which is why a 21M-parameter trunk carries
features worth correcting at all. It is a licence-gated Hub repo: accept it on
the model page and `hf auth login` once, or point `backbone_id` at a mirror you
already have — nothing in the code assumes the official id.

Pooling is `cls+patchmean` (768-d), DINOv3's own linear-probe recipe. It also
matters for *this* project specifically: generation traces are a local
high-frequency phenomenon, so a detector reading only the CLS token would lean on
semantics and lose accuracy for reasons unrelated to the artefacts JPEG and blur
destroy — exactly the confound the retention curve must not contain.

| | RINE | DINOv3 PoC |
|---|---|---|
| layout | `layers` (24, 1024) | `vector` (768,) |
| cache | 48 KB/image/view | **1.5 KB**/image/view |
| 15 views x 1200 images | 864 MB | **27 MB** |
| per-layer damage profile | yes — the figure | no |

**The cost is real and worth stating.** A `vector` split has one gate vector, so
there is no "which encoder block does blur destroy" plot, and the stage-2
discrepancy head sees **one** drift norm where RINE's would see 24. The PoC
therefore tests GRACE-D in its weakest form, and a null result there is much
weaker evidence against the branch than a null result on a `layers` detector
would be. Report it that way.

The `layers` variant is a small change when it is wanted — emit per-block CLS
tokens as `(B, 12, 384)` and give the head an importance weighting over them.
Nothing downstream of the split needs touching: `grace.models.factory` picks the
`(L, D)` gate off `FeatureSpec.layout` alone.

### 8.2 Stage 0 — the one place a detector is trained

A DINOv3 trunk has no classifier, and GRACE cannot adapt a seam whose head does
not exist yet. `scripts/train_probe.py` fits one, once, and it is the only script
in the project that trains a detector.

```bash
python scripts/train_probe.py configs/probe/dinov3_ntire.yaml
```

Two passes and a 400k-parameter fit: one trunk forward per image ever (the trunk
is frozen and the images are not degraded, so the features are constant), then
AdamW on the head against those features. Seconds. Same argument as section 2,
one scale down.

**Clean images only, no augmentation.** This is the premise, not a shortcut. The
whole claim concerns a detector fit on clean data whose accuracy collapses under
degradation; a head trained with degradation augmentation would have partly
solved the problem GRACE exists to solve, and every retention number downstream
would be measuring the augmentation instead of the adapter. If you want that arm,
it is a separate detector config and a separate baseline, not a flag.

Model selection is on held-out **images**, by AUC. A 768-in/512-hidden head
reaches training AUC 1.0 within a few epochs on a PoC-sized manifest, and the
last epoch is not the one to ship — an overfit head has a near-arbitrary
Jacobian, and section 4.1's weighting differentiates through it to decide where
the adapter spends its capacity.

The head is written to the path the **detector config** names in
`args.head_checkpoint`, so the file the probe produces and the file the detector
loads are one string in one place. The trunk never sees the head, so re-fitting
the probe does not invalidate a rendered cache: `detector_sha` hashes the config,
and the head path in it names weights the cached features never saw.

### 8.3 Running it

```bash
bash scripts/poc.sh              # the whole path
bash scripts/poc.sh --smoke      # 2 epochs, 2 cache views -- minutes, proves wiring
WANDB=1 bash scripts/poc.sh      # every stage tracked under one group
```

or by hand:

```bash
# 0. dataset -- NTIRE shards 0-4 (train) + shard 5 (selection) into ONE manifest
cd ../eval_pipeline && python scripts/build_manifest.py --config configs/datasets/ntire_train.yaml

# 1. stage 0 -- fit the head on clean features
cd ../grace_adapter && python scripts/train_probe.py configs/probe/dinov3_ntire.yaml

# 2. the baseline this is all measured against -- run it BEFORE training anything
cd ../eval_pipeline && python scripts/run_eval.py --config configs/runs/dinov3_poc_baseline.yaml

# 3. render the cache
cd ../grace_adapter
python scripts/build_cache.py       configs/cache/dinov3.yaml --dry-run
python scripts/build_cache.py       configs/cache/dinov3.yaml

# 4. E0 -- AFTER the render (it compares the clean view against a degraded one),
#    but before anything is trained
python scripts/analyze_drift.py --cache cache/dinov3-ntire \
  --dataset  ../eval_pipeline/configs/datasets/ntire_train.yaml \
  --detector ../eval_pipeline/configs/detectors/dinov3-ntire.yaml \
  --split    grace.splits.dinov3.DINOv3Split

# 5. stage 1 (both arms), 6. stage 2
python scripts/train_adapter.py     configs/train/dinov3_clean.yaml       # arm B
python scripts/train_adapter.py     configs/train/dinov3_degraded.yaml    # arm A
python scripts/train_discrepancy.py configs/train/dinov3_discrepancy.yaml

# 7. score all three arms through the SAME harness as the baseline
cd ../eval_pipeline && python scripts/run_eval.py --config configs/runs/dinov3_poc_grace.yaml
```

`configs/datasets/ntire_train.yaml` puts **two** splits in one manifest, which no
other dataset config does — evaluation needs one held-out set, but stage 0 needs
a training set drawn from the same distribution and provably disjoint from it.
`ConcatSource` gives each child its own image subdirectory, and that is
load-bearing rather than tidy: `HFImageDatasetSource` names files by position
within its split, so both splits start at `images/00000000_0.png` and without the
prefix the second would overwrite the first, leaving a manifest whose train and
validation rows point at the same pixels.

### 8.4 What the PoC can and cannot answer

| | answerable here | why |
|---|---|---|
| E1 identity | yes, trivially | the seam is a construction, not a reconstruction |
| E2 arm A vs arm B | **yes** | the clean-teacher ablation needs no particular layout |
| E3 loss ablations | **yes** | all four terms operate on the last axis |
| E4 erasure trade-off | partially | Δ is one norm here, not a per-block profile |
| E5 GRACE-D | weakly | the auxiliary head's weakest input |
| E6 cached vs live | yes | `source: live` is layout-agnostic |
| per-block damage figure | **no** | needs a `layers` split |

E2 and E3 are the load-bearing ones and they run here in seconds. That is the
argument for the PoC: the experiments that decide whether the *objective* works
become cheap, and the ones that need a real published detector stay waiting on
its clone — rather than everything waiting on it.

---

## 9. Tracking (Weights & Biases), optional

Off by default. All three training stages log through
`grace.train.tracker`, which is a null object when disabled — so there is no
`if wandb is not None` anywhere in `loop.py` and no code path that only runs on
somebody's machine.

```bash
python scripts/train_adapter.py configs/train/dinov3_clean.yaml \
  --wandb --wandb-group e2_teacher
```

or `wandb: {enabled: true, group: e2_teacher}` in the run config. `--wandb-offline`
writes to `./wandb/` for a node with no outbound network.

Three properties the rest of the package depends on:

* **W&B is never the record.** `summary.json` next to the checkpoints stays the
  source of truth, written whether or not anything was tracked. Two people
  comparing results compare files, not screenshots.
* **W&B never fails a run.** A dead network, an expired key or a missing package
  warns once and continues untracked. The one exception is `enabled: true` with
  the package absent, which is a configuration error and is raised at second zero
  rather than after the GPU time.
* **The step axis is the training step**, passed explicitly. W&B's implicit
  counter would interleave the 50-step diagnostics with the end-of-epoch rows and
  make two runs with different `log_every` incomparable.

`group` is what makes the sweeps in section 7 legible — set it to the experiment
id and every arm lands in one comparison rather than in a flat list of forty
runs. For E4 specifically, stage 2 logs `adapter_checkpoint` as run config, which
is the x-axis of the erasure figure.

---

## 10. Future additions — blueprint only

**`models/ladder.py`** — ladder / multi-seam adapter. Hook 4–6 intermediate
blocks, project, fuse into the correction at the seam (side-tuning / LST).
Degradation corrupts early blocks differently from late ones; the RINE per-layer
gate already anticipates this for one detector, and generalising it makes "which
blocks need correction under blur vs. compression" a headline interpretability
result rather than a footnote.

> **One forward-compatibility decision was made now**, because it is cheap now and
> expensive to retrofit: `CacheSpec` carries a `taps` field and `SplitDetector`
> has a `taps()` hook, both empty. The ladder then *adds views* to an existing
> cache rather than invalidating its on-disk format.

**`models/prompts.py`** — degradation prompts. A bank of learnable prompts
selected by soft attention (PromptIR), with the degradation embedding obtained
contrastively and without labels (AirNet) — which fits the label-free framing
better than the supervised severity scalar does. Same parameter budget, and
"degradation prompts for feature-space restoration" is a cleaner novelty statement
than "FiLM on a gate". The attention weights are a soft classification of the
degradation obtained without degradation labels; comparing them against the known
recipe in `recipes.parquet` is a free confusion-matrix figure.

---

## 11. The one upstream change

`Condition.sample_recipe` short-circuited on `if self.level < 2`, because eval's
L1 conditions carry an explicit fixed `steps` — the 19-point OFAT grid. Training
needs L1 to mean *one randomly drawn transform*. The guard is now:

```python
if not self.grid:            # was: if self.level < 2
    return Recipe(self.steps)
```

All four eval behaviours are unchanged (L0 has neither `steps` nor `grid`; L1 has
`steps`, no `grid`; L2/L3 have a `grid`), and `LEVELS[1]["n_transforms"] == (1, 1)`
already said one. **Made**, and covered by
`tests/test_schedule.py::test_eval_conditions_are_unaffected_by_that_change`; the
harness's own 17 tests still pass.

## 12. Gotchas

- **Cache/dataset index alignment** is the highest-risk bug in the project. It
  trains, it converges, it means nothing. `tests/test_cache_alignment.py` renders a
  real cache and re-runs the trunk live on 20 random indices, clean *and* degraded.
- **Never shuffle before caching. Never rebuild the manifest afterwards.** Four
  fingerprints are asserted at load and the error names *which* one moved.
- **Preprocessing must be deterministic.** `sha_preprocess` runs the transform
  twice on a probe and fails at startup rather than 40 GB later.
- **The trunk stays in `eval()` every step**, asserted inside the loop by
  `assert_frozen`, not once at startup.
- **Cast cached fp16 to fp32 before any loss.** fp16 MSE on unnormalized ViT
  features underflows to zero and trains nothing.
- **`head` must be differentiable w.r.t. its input.** The Jacobian weighting takes
  a gradient at the clean features; a head wrapped in `no_grad` silently disables
  it. Parameters stay frozen; only the input needs a graph.
- Open memmaps **per worker**, never inherited across a fork.

## 13. Status

Implemented and tested: the adapter, weighting, losses, diagnostics, discrepancy
branch, schedule, cache (writer/reader/spec), both training stages, the configs,
`AdaptedDetector`, the DINOv3 proof-of-concept path (§8) and optional W&B
tracking (§9). 203 tests pass, including a real end-to-end render, a two-stage
training smoke run, and the full PoC path — stage 0 → cache → stage 1 → stage 2 →
identity check — against a small locally-constructed DINOv3.

**Runnable today:** the PoC path, `bash scripts/poc.sh`. It needs the DINOv3
backbone, which is a licence-gated Hub repo (§8.1), and nothing else. Its tests
need neither: they build a 2-layer DINOv3 from a local config, so the wiring is
covered with no network and no licence.

**Not verified:** `splits/rine.py`, `splits/bfree.py` and `splits/gapl.py` compose
modules from repos cloned by hand under `third_party/`, which are not in this tree.
`RINESplit._head_forward` is written against the documented upstream structure and
**must be checked against the clone**. This is why `verify_split` runs in every
split's `__init__`: a wrong composition fails immediately and loudly, listing the
trainable modules it found, instead of scoring a model that was never benchmarked.
B-Free and GAPL raise `NotImplementedError` in `trunk` pending their clones.

That gap is the reason §8 exists. The experiments that decide whether the
*objective* works — E2's clean-teacher ablation and E3's loss ablations — need no
particular detector, and on the PoC they cost seconds. The experiments that need
a published detector still wait on its clone, but they are now the only thing
waiting.
