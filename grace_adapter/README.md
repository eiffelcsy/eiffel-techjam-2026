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
`FrozenDetector` — so the baseline and the adapted numbers come out of the same
code path, the same conditions, and the same result schema.

---

## 1. The two claims

`../eval_pipeline` measures the problem: detectors lean on local high-frequency
traces, and JPEG, blur and resize destroy exactly those. Retention —
`(auc_deg − 0.5) / (auc_clean − 0.5)` — is the number that collapses.

**GRACE's claim is that the evidence is displaced, not destroyed.** The degraded
features still carry the discriminative signal; it has moved somewhere the frozen
head does not look. If so, a ~0.4M-parameter residual correction recovers a real
slice of the retention gap. If a large adapter turns out to be needed, the claim
is false and the result is far less interesting — which is why the parameter
budget is a design constraint, not a convenience.

**GRACE-D's claim is that the damage is itself a signal.** RA-Det found that
generated images drift further in embedding space under perturbation than real
ones do. An adapter trained purely to *erase* drift is therefore destroying
forensic evidence while its reconstruction loss falls, and doing so
asymmetrically. So keep the quantity the adapter already computes: Δ is its
estimate of the drift, available at test time without the clean image, as a
by-product of a module that was running anyway.

This also breaks the restoration ceiling. A perfect restorer can at best recover
the clean-image score — retention 1.0. A fused score reading Δ can exceed it,
because the *magnitude of the damage* is information the clean image does not
contain.

| | Adapter | Aux head | Labels | Ceiling |
|---|---|---|---|---|
| **GRACE** | trained | — | none | retention ≤ 1.0 |
| **GRACE-D** | *bit-identical, frozen* | trained | yes | may exceed 1.0 |

Stage 2 never touches the adapter, so both variants ship the same weights and
"the adapter is trained without labels" stays literally true. That separation is
what makes the erasure question (§6, E4) testable.

---

## 2. Why the training loop has no trunk in it

The trunk is frozen and the clean image never changes, so **the clean features
are constant**: compute them once, to disk, and the teacher becomes a lookup.

The same argument extends to the degraded side.
`pipeline.degrade.conditions.Condition` draws every recipe from
`stable_seed(index, level, replicate, seed)` — a blake2b hash, never a global RNG
counter — so a degraded view is *also* a pure function of (image, condition). And
the harness already has a field whose entire purpose is "an independent re-draw
over the same images": `replicate`.

```
epoch  ≡  replicate
```

Epoch 7's degradation of image 412 is computable now, without having run epochs
0–6 — the precondition for rendering every epoch **offline, ahead of time**.

> In `source: cache` mode the training loop contains **no trunk forward at all**.
> A step is two memmap reads and a 2-layer MLP.

That changes which experiments are affordable, not just how fast one is. Stage 2
costs *seconds*, which is what makes the E4 sweep over stage-1 checkpoints
practical rather than theoretical.

### What it costs

Per image, per view, float16 (one view = clean, or one epoch):

| Detector | Layout | Shape | Bytes/image/view | 15 views |
|---|---|---|---|---|
| DINOv3 (PoC) | `vector` | (768,) | 1.5 KB | 23 KB |
| GAPL | `vector` | (1024,) | 2 KB | 30 KB |
| B-Free | `tokens` | (5, 768) | 7.5 KB | 113 KB |
| RINE | `layers` | (24, 1024) | 48 KB | **720 KB** |

The PoC train cache is the full 277,643-image NTIRE train split at 15 views
(clean + 12 training epochs + 2 held-out): ~6.3 GB. RINE at that size would be
~200 GB. Always run `build_cache.py --dry-run` first.

Compute, for `R` runs over `E` epochs and `N` images: degrading in the loop costs
`R × N × E` trunk forwards, every run; pre-rendering costs `N × (E + 1)`, once,
resumably. Break-even is the first run.

### What it costs you

Pre-rendering fixes the augmentation set at `E` draws per image. Three
mitigations, all built in:

1. `E ≥ 8`, with a fresh recipe per (image, epoch) — not per image.
2. **Held-out degradations.** Validation epochs are numbered from
   `VAL_EPOCH_OFFSET = 10_000`, so their `replicate` can never collide with a
   training epoch's. Disjoint draws from the same distribution.
3. **`source: live`** (`configs/train/*_live.yaml`) degrades in the loop with the
   same schedule and is the direct control. Cached ≈ live means the finite epoch
   set is not being exploited.

---

## 3. The objective

```
L = L_align + λ_sw·L_SW + λ_id·L_identity + λ_kl·L_headKL + λ_sev·L_severity
```

Every term is label-free. Only stage 2's BCE uses image labels.

**Spend capacity where it changes the decision.** Plain MSE treats every feature
direction as equally worth fixing; the head does not. It maps features to one
scalar, so only error inside its sensitive subspace can move AUC. Its Jacobian is
a gradient vector the shape of the feature:

```
j = ∇_f h(f) |_{f = f_clean}
L_err = (1−ε)·mean[(ĵ·e)²]  +  ε·mean[e²]          e = f_adapted − f_clean
```

For a linear head `j` is exactly the constant `w`, so one implementation covers
both linear and MLP heads with no branch. Written as a blend so `ε = 1` is
*exactly* `F.mse_loss`: the plain-MSE ablation is one config key and provably the
same objective GRACE v1 had (`tests/test_losses.py`). `head_kl` is the exact
version of the first term — a finite difference through the real head rather than
a first-order expansion — but it only ever sees the scalar, so it is kept and
demoted to `λ_kl = 0.1`.

The diagnostic that motivates this empirically is `cos(Δ, j)`, logged every 50
steps. If a plain-MSE run sits near 0, the adapter is spending nearly all its
capacity on directions the head cannot see.

**Distribution matching and posterior sampling are one feature.** Point-wise
alignment is satisfied by a conditional mean, which is systematically
under-dispersed: the corrected batch forms a tighter cloud than real clean
features do, and the frozen head's operating point was calibrated on the wider
one. Sliced-Wasserstein fixes that — random projections, sort, L2; one
hyperparameter, no adversarial stability risk, and a *matched* comparison since
both batches hold the same images. Posterior sampling gives the adapter a noise
input `z`, draws k corrections, and averages **logits, not features** —
`E[h(f)] ≠ h(E[f])` for any nonlinear head.

> Under point-wise reconstruction losses alone the optimal stochastic policy is
> to ignore `z` — posterior collapse. Noise earns its keep *only* because the SW
> term rewards matching the spread a conditional mean under-disperses. Shipping
> `noise_dim > 0` with `lam_sw: 0` buys parameters that do nothing, which is why
> `*_no_sw.yaml` and `*_posterior.yaml` run as a pair. `posterior_spread` in the
> validation block is the tripwire; ~0 is a reportable negative result about the
> objective, not a bug.

**Severity is free and does not cost the label-free claim.** Transform grids are
ordered mild → severe, so a step's severity is its parameter's normalised rank
within its own grid, combined with composition depth. The degradation sampler
already knows both, so the target is written into `recipes.parquet` at render
time. The labels are the sampler's own metadata, not image labels.

---

## 4. The adapter — one class, because layout is a gate shape

```
y = f + g ⊙ MLP(LN(f)),    g = sigmoid(gate_logit),    gate_logit init −4
```

Everything operates on the last axis, so `(B, D)`, `(B, T, D)` and `(B, L, D)`
run through the same code with the same weights shared across the group axis.
Only the gate shape differs — `(D,)` for vector and tokens, `(L, D)` for layers —
and the factory picks it. Nothing else in the project branches on layout.

The `(L, D)` gate is also the interpretability output: mean it over `D` and you
have how much correction each encoder block needs, per degradation.

**Identity at initialization, exactly.** The last projection of every block is
zero-initialised, so the adapter returns its input bit-for-bit whatever the gate,
the noise, or the severity conditioning happen to be — and the same trick makes
`β = 0` mean GRACE-D *is* GRACE at init. Without this, a clean-AUC change is
unattributable.

Log `gate().mean()`. It should climb off 0.018 and plateau around 0.1–0.5.
Saturating at 1.0 is over-correction; sitting at init means the alignment term is
too weak against the identity term.

---

## 5. Package map

Laid out like the sibling harness: a descriptive project directory holding a
short importable package, so `eval_pipeline/` → `import pipeline` and
`grace_adapter/` → `import grace`.

```
grace/
├── splits/          the trunk/head seam, added AROUND detectors
│   ├── base.py        FeatureSpec, SplitDetector, assert_frozen
│   ├── verify.py      head(trunk(x)) == detector(x), checked at construction
│   ├── rine.py bfree.py gapl.py    reconstruct a seam inside a vendored repo
│   └── dinov3.py      delegates to a seam that is already there            ← §7
├── probe/train.py   stage 0 — fit the PoC detector's own head. PoC ONLY.
├── cache/
│   ├── schedule.py    (index, epoch) -> degradation, as a pure function    ← §2
│   ├── spec.py        four fingerprints + the FUTURE `taps` field
│   ├── writer.py      offline render: clean view + one view per epoch
│   └── reader.py      memmap random access, per-worker, by manifest index
├── models/
│   ├── adapter.py     GatedResidualAdapter (+ noise, + severity FiLM)
│   ├── severity.py    SeverityHead — target is free from recipes.parquet
│   ├── discrepancy.py DiscrepancyHead + FusedHead                          ← §1
│   ├── factory.py     the only layout branch, three lines
│   ├── ladder.py      FUTURE — blueprint only
│   └── prompts.py     FUTURE — blueprint only
├── train/
│   ├── weighting.py   head_gradient, decision_weighted_error               ← §3
│   ├── losses.py      alignment, sliced-Wasserstein, identity, KL, severity
│   ├── diagnostics.py cos(Δ,j), drift asymmetry, posterior spread
│   ├── data.py        CachedPairDataset | LivePairDataset — one config flag
│   ├── ema.py         EMA shadow weights; every run ships raw + ema.pt
│   ├── tracker.py     W&B as a null object — off by default, never fatal   ← §8
│   └── loop.py        stage 1 (label-free), stage 2 (supervised), validate
└── detectors/adapted.py   AdaptedDetector — a FrozenDetector, for the harness
```

Configs, four kinds, one directory each:

```
configs/
├── probe/        PoC only — fit a frozen detector's own classification head
├── cache/        what to render, for which detector, to which out_dir
├── train/        one stage-1 or stage-2 run against a cache
├── detectors/    the ADAPTED detector, in the HARNESS's config shape
└── defaults.yaml annotated reference: every key with its default. Never loaded.
```

Detectors and datasets are **never redefined here**. Configs reference
`../eval_pipeline/configs/` by path, so each is described in exactly one place
and GRACE cannot drift from what was benchmarked. The PoC detector obeys the same
rule: `DINOv3MLPDetector` lives in `pipeline/detectors/dinov3.py` with the rest
of the zoo, and `grace/` contributes only its seam and the script that fits its
head.

**Stage-1 validation has two axes, reported separately** in `summary.json`:

- `held_out_degradations` — the cache's val epochs (numbered from 10000). Unseen
  corruptions over images that *were* trained on, so it cannot speak to
  generalization across images.
- `held_out_images/<name>` — whole datasets the adapter never saw, named by the
  parallel `val_datasets` / `val_cache_dirs` lists. Each needs its own rendered
  cache root, because `build_cache.py` derives its root from the detector name
  alone and two datasets under one `out_dir` would collide. The *images* are
  what is held out here, so the degradation draw need not also be: a
  training-numbered epoch is the right corruption to score, applied to rows the
  adapter never saw.

Both run once at the end of a run. Set `val_every: N` to also run them every N
epochs, appended to `summary.json` as `val_history` -- a list of
`{epoch, step, ...the same two axes}` rows -- and logged to W&B under `val/` at
the step they were measured. `validation` still holds the finished adapter
either way, so nothing downstream has to ask what schedule a run used. Matching
`val_every` to `checkpoint_every` pairs every E4 checkpoint with the held-out
numbers it scored when it was written. Validation forks the RNG, so turning it
on does not perturb a single training draw.

**A mid-run pass scores one epoch per image cache** -- the draw training just
finished -- while the end-of-run pass sweeps them all. The degradation axis is
always both val epochs. That asymmetry is a cost decision: on the PoC caches a
full sweep is 14 passes per val set, and the epochs of an image cache are
fourteen reads of one axis rather than fourteen different questions.

**Each row carries alignment *and* detection metrics.** Alignment:
`cosine_to_clean`, `gate`, `posterior_spread`. Detection, through the frozen
head, for three views — `degraded` (the input, what the detector scores without
GRACE), `adapted` (the adapter's output) and `clean` (the ceiling): `auc_*`,
`acc_*`, `f1_*`, plus `retention` = `(auc_adapted - 0.5) / (auc_clean - 0.5)`.
`threshold` is picked on the clean view and applied unchanged to the other two,
which is `pipeline.eval.metrics`' rule and is what exposes calibration drift
that AUC hides.

Those detection numbers are computed by importing the harness's own metric
functions, not by reimplementing them — but they are still the *in-loop* signal,
measured on cached features. Reported retention is the harness's, on the eval
split, through `grace.detectors.adapted`.

---

## 6. The pipeline, and the experiments

> The operator's version of this section — every arm with its prerequisites, its
> exact commands, the keys to read out of each artifact and how to interpret them
> — is **`../docs/EXPERIMENTS.md`**. What follows is the map.

```bash
# 1. render, once per detector and per dataset -- resumable at shard granularity
python scripts/build_cache.py configs/cache/<name>.yaml --dry-run
python scripts/build_cache.py configs/cache/<name>.yaml

# 2. the premise, before anything is TRAINED on it -- reads the cache, no GPU
python scripts/analyze_drift.py --cache cache/<detector> --dataset <dataset.yaml>

# 3. stage 1, minutes per run
python scripts/train_adapter.py configs/train/<name>.yaml

# 4. stage 2, seconds per run
python scripts/train_discrepancy.py configs/train/<name>_discrepancy.yaml

# 5. score through the SAME harness that produced the baseline
cd ../eval_pipeline && python scripts/run_eval.py --config configs/runs/<run>.yaml
python scripts/compare.py --baseline <baseline.json> --adapted <adapted.json>
```

**The render comes first, and E0 is second.** `analyze_drift.py` opens the cache
in its first statement and reads clean *and* degraded features out of it; it
exits with `no rendered epochs under <dir>` if no degraded view has been
finalized. "E0 first" means *before anything is trained*.

| # | Arm | Config | Asks |
|---|---|---|---|
| E0 | drift analysis | `scripts/analyze_drift.py` | does RA-Det's asymmetry hold here? |
| E1 | identity | `detectors/*+identity.yaml` | does the split reproduce the baseline *exactly*? |
| E2 | A vs B | `train/*_degraded.yaml` / `*_clean.yaml` | does the clean teacher buy retention? |
| E3 | loss ablations | `*_plain_mse` / `*_no_sw` / `*_posterior` | Jacobian vs MSE; ±SW; ±noise |
| **E4** | **erasure trade-off** | stage 2 vs every stage-1 checkpoint | **does the adapter destroy evidence?** |
| E5 | GRACE-D | `detectors/*+grace-d.yaml` | does the fused score beat retention 1.0? |
| E6 | cached vs live | `train/*_live.yaml` | is the finite epoch set being exploited? |

Every arm has both a `rine_` and a `dinov3_` config under `configs/train/`. Only
the `dinov3_` ones run today (§10); each ablation differs from its `*_clean.yaml`
baseline in exactly one key, repeated verbatim rather than inherited, because an
ablation that silently differs in a second key is not an ablation.

**Run E1 first regardless.** If the identity adapter does not reproduce the
baseline JSON to the last decimal, the split is wrong and everything downstream
compares against a model that was never benchmarked.

**E0's two outcomes are both useful.** Asymmetry present → the discrepancy branch
has signal and the label-free objective is knowingly discarding it. Asymmetry
absent → stage 2 will be weak here; say so, keep the restoration result, and save
a day. The parallel/orthogonal decomposition matters as much as the gap: drift
that is large but orthogonal to the decision direction is invisible to the frozen
head, which is exactly why an auxiliary head can recover signal the main head
cannot — and why the main loss can fall while evidence is destroyed.

**E4 answers the critique directly.** `*_clean.yaml` sets `checkpoint_every: 2`;
train stage 2 against each intermediate checkpoint and plot the auxiliary head's
standalone AUC against stage-1 progress:

```bash
for ck in checkpoints/grace/dinov3_clean/step_*.pt; do
  python scripts/train_discrepancy.py configs/train/dinov3_discrepancy.yaml \
    --adapter "$ck" --run-id "e4_$(basename "$ck" .pt)"
done
```

A falling `auc_aux` as stage 1 improves is direct evidence that restoring
features erases forensic evidence, and the retention-versus-drift-preservation
curve is the figure. It costs almost nothing because stage 2 is seconds.

`scripts/compare.py` reports retention against the **baseline's** clean AUC, so
`> 1.0` means what it should — the harness's own retention divides by each
detector's own clean AUC, which for GRACE-D would hide exactly the effect being
claimed.

---

## 7. The proof-of-concept path: DINOv3 ViT-S/16 + an MLP head

Everything above describes adapting *somebody else's* detector, and §10 says why
none of it can be run yet: the three zoo splits compose modules from repos cloned
by hand under `third_party/`, and `RINESplit._head_forward` is written against
documented structure rather than against a clone. Until those exist, a retention
number could come from a wrongly composed head and nothing in the curve would say
so.

So there is a fourth detector, built here rather than downloaded, whose only job
is to make the seam not a question:

```
trunk  frozen DINOv3 ViT-S/16 (distilled), pooled  -> (B, 768)   layout "vector"
GRACE  the adapter, spliced at that seam           -> (B, 768)
head   LayerNorm -> Linear -> GELU -> Linear       -> (B,)  one logit
```

`head(trunk(x)) == detector(x)` holds because `DINOv3MLPDetector.forward` *is*
`self.head(self.trunk(x))`, and `DINOv3Split` delegates to both rather than
reconstructing either. E1 becomes a tautology instead of a nail-biter — precisely
what you want from the arm you debug the rest of the pipeline in.

The backbone is `facebook/dinov3-vits16-pretrain-lvd1689m`, the ViT-S/16
distilled from the ViT-7B teacher on LVD-1689M, which is why a 21M-parameter
trunk carries features worth correcting at all. It is a licence-gated Hub repo:
accept it on the model page and `hf auth login` once, or point `backbone_id` at
a mirror. Pooling is `cls+patchmean` (768-d), DINOv3's own linear-probe recipe —
and the right choice here specifically, since generation traces are a local
high-frequency phenomenon and a CLS-only detector would lean on semantics.

`../eval_pipeline/configs/detectors/dinov3-ntire-crop.yaml` is the second
preprocessing arm (`input_mode: crop`, a 224 window at native resolution rather
than a whole-image squash to 224). See the harness README: on the previous
dataset a `resize` probe learned content rather than forensics, and a detector
that never collapses cannot demonstrate a repair.

**The cost of a `vector` layout is real and worth stating.** One gate vector
means no "which encoder block does blur destroy" plot, and the stage-2
discrepancy head sees **one** drift norm where RINE's `layers` split would see
24. The PoC therefore tests GRACE-D in its weakest form, and a null result there
is much weaker evidence against the branch than a null on a `layers` detector
would be. Report it that way.

| | answerable on the PoC | why |
|---|---|---|
| E1 identity | yes, trivially | the seam is a construction, not a reconstruction |
| E2 arm A vs arm B | **yes** | the clean-teacher ablation needs no particular layout |
| E3 loss ablations | **yes** | all four terms operate on the last axis |
| E4 erasure trade-off | partially | Δ is one norm here, not a per-block profile |
| E5 GRACE-D | weakly | the auxiliary head's weakest input |
| E6 cached vs live | yes | `source: live` is layout-agnostic |
| per-block damage figure | **no** | needs a `layers` split |

E2 and E3 are the load-bearing ones and they run here cheaply. That is the
argument for the PoC: the experiments that decide whether the *objective* works
become affordable, and only the ones needing a real published detector stay
waiting on its clone.

### 7.1 Stage 0 — the one place a detector is trained

A DINOv3 trunk has no classifier, and GRACE cannot adapt a seam whose head does
not exist yet. `scripts/train_probe.py` fits one, once.

```bash
python scripts/train_probe.py configs/probe/dinov3_ntire.yaml
```

Two passes and a ~400k-parameter fit: one trunk forward per image ever (the trunk
is frozen and the images are not degraded, so the features are constant), then
AdamW on the head against those features.

**Clean images only, no augmentation.** This is the premise, not a shortcut. A
head trained with degradation augmentation would have partly solved the problem
GRACE exists to solve, and every retention number downstream would measure the
augmentation instead of the adapter. If you want that arm, it is a separate
detector config and a separate baseline, not a flag.

> **Caveat on NTIRE specifically.** The NTIRE train set already mixes
> in-the-wild transformations (crop, resize, compression, blur) into both
> classes and, unlike the val set, ships no `is_distorted` flag to filter them
> out. "Fit on clean data" is therefore only approximately true of any head
> trained against it. There is no way to filter it from the table — it is a
> property of the dataset.

Model selection is on held-out **images**, by AUC, over the challenge's own val
sets (`ntire_val` + `ntire_val_hard`) — the selection scalar is the unweighted
mean of their AUCs, and each is also reported alone, because a head that wins the
mean by collapsing on the hard set is the exact failure this project is about.
That is sound here because the *reported* benchmark is WildFake, so nothing
selected on NTIRE val is ever reported against NTIRE val.

The head is written to the path the **detector config** names in
`args.head_checkpoint`, so the file the probe produces and the file the detector
loads are one string in one place. Re-fitting the probe does not invalidate a
rendered cache: `detector_sha` hashes the config, and the head path in it names
weights the cached features never saw.

### 7.2 Running it

```bash
bash scripts/poc.sh              # the whole path
bash scripts/poc.sh --smoke      # 2 epochs, 2 cache views -- minutes, proves wiring
WANDB=1 bash scripts/poc.sh      # every stage tracked under one group
```

Eight steps, each idempotent and each annotated in the script itself: four
manifests → stage 0 → the WildFake baseline → the train cache and the two
validation caches → E0 → both stage-1 arms → stage 2 → score all three arms
through the harness. Run them by hand from `poc.sh` if you want to stop between
stages; it is the executable version of §6's sequence with the PoC's configs
filled in, and it prints what to read, in order, when it finishes.

Two things about that sequence are load-bearing:

- **The validation caches are not optional.** `configs/train/dinov3_*.yaml` name
  `cache_val/dinov3-ntire` and `cache_val_hard/dinov3-ntire` in
  `val_cache_dirs`, and `train_adapter.py` opens them in its first few statements
  so a missing one fails at second zero rather than after the run.
- **The baseline and the arms must score the same dataset** — both run configs
  name WildFake and nothing else. `compare.py` refuses two result files from
  different eval sets, because it normalizes retention by the *baseline's* clean
  AUC and that is only comparable on one set. NTIRE val is a selection set here,
  so a retention curve measured there would flatter baseline and adapter alike;
  it is still reported for stage 1, in
  `checkpoints/grace/<run>/summary.json` under `validation`.

`configs/runs/dinov3_poc_baseline_crop.yaml` is the preprocessing ablation: the
same baseline with the crop-fed head (§7). Read the two against each other —
GRACE needs a detector whose accuracy *collapses* under degradation, and a head
that reads content rather than generation traces gives a flat line at 100%
retention with no room for a repair.

---

## 8. Tracking (Weights & Biases), optional

Off by default in code; the shipped `dinov3_*` train configs enable it. All three
stages log through `grace.train.tracker`, which is a null object when disabled —
so there is no `if wandb is not None` anywhere in `loop.py`.

```bash
python scripts/train_adapter.py configs/train/dinov3_clean.yaml \
  --wandb --wandb-group e2_teacher      # or --no-wandb to force it off
```

`--wandb-offline` writes to `./wandb/` for a node with no outbound network.
Naming a project or group implies tracking.

Three properties the rest of the package depends on:

* **W&B is never the record.** `summary.json` next to the checkpoints stays the
  source of truth, written whether or not anything was tracked.
* **W&B never fails a run.** A dead network, an expired key or a missing package
  warns once and continues untracked. The one exception is `enabled: true` with
  the package absent, which is a configuration error raised at second zero.
* **The step axis is the training step**, passed explicitly, so two runs with
  different `log_every` stay comparable.

`group` is what makes the sweeps in §6 legible. For E4 specifically, stage 2 logs
`adapter_checkpoint` as run config, which is the x-axis of the erasure figure.

---

## 9. Future additions — blueprint only

Both raise `NotImplementedError` with the design in the module docstring.

- **`models/ladder.py`** — ladder / multi-seam adapter. Hook 4–6 intermediate
  blocks, project, fuse into the correction at the seam (side-tuning / LST),
  since degradation corrupts early blocks differently from late ones.
- **`models/prompts.py`** — degradation prompts. A bank of learnable prompts
  selected by soft attention (PromptIR), with the degradation embedding obtained
  contrastively and without labels (AirNet) — a better fit for the label-free
  framing than the supervised severity scalar. The attention weights are a soft
  classification of the degradation obtained *without* degradation labels, so
  comparing them against `recipes.parquet` is a free confusion-matrix figure.

> **One forward-compatibility decision was made now**, because it is cheap now
> and expensive to retrofit: `CacheSpec` carries a `taps` field and
> `SplitDetector` has a `taps()` hook, both empty. The ladder then *adds views*
> to an existing cache rather than invalidating its on-disk format.

---

## 10. Status

**Implemented and tested.** The adapter, weighting, losses, diagnostics,
discrepancy branch, schedule, cache (writer/reader/spec), EMA, both training
stages, the two-axis validation, the configs, `AdaptedDetector`, the DINOv3
proof-of-concept path and optional W&B tracking. **224 tests pass**, including a
real end-to-end render, a two-stage training smoke run, and the full PoC path —
stage 0 → cache → stage 1 → stage 2 → identity check — against a small
locally-constructed DINOv3 that needs neither network nor licence.

**Run so far.** Stage 0 is done on the full NTIRE train split (277,643 images):
the selected head is epoch 36, at **0.9596 AUC on `ntire_val`** and **0.8467 on
`ntire_val_hard`** (mean 0.9032) — see
`checkpoints/probe/dinov3_ntire/head.summary.json`. Both validation caches
(`cache_val/`, `cache_val_hard/`, 15 views each) are rendered. The train cache is
still rendering. No adapter has been trained yet, and no harness results exist
yet — so every retention number in this README is a target, not a measurement.

**Known gaps.**

- `splits/rine.py`, `splits/bfree.py` and `splits/gapl.py` compose modules from
  repos cloned by hand under `third_party/`, which are not in this tree.
  `RINESplit._head_forward` is written against the documented upstream structure
  and **must be checked against the clone**. This is why `verify_split` runs in
  every split's `__init__`: a wrong composition fails immediately and loudly,
  listing the trainable modules it found, instead of scoring a model that was
  never benchmarked. B-Free and GAPL raise `NotImplementedError` in `trunk`
  pending their clones.

That last gap is the reason §7 exists. The experiments that decide whether the
*objective* works — E2's clean-teacher ablation and E3's loss ablations — need no
particular detector; the ones that need a published detector are now the only
thing waiting on a clone.

## 11. The one upstream change

`Condition.sample_recipe` short-circuited on `if self.level < 2`, because eval's
L1 conditions carry an explicit fixed `steps` — the 19-point OFAT grid. Training
needs L1 to mean *one randomly drawn transform*, so the guard is now
`if not self.grid:` instead. All four eval behaviours are unchanged (L0 has
neither `steps` nor `grid`; L1 has `steps`, no `grid`; L2/L3 have a `grid`), and
`LEVELS[1]["n_transforms"] == (1, 1)` already said one. Covered by
`tests/test_schedule.py::test_eval_conditions_are_unaffected_by_that_change`; the
harness's own 26 tests still pass.

## 12. Gotchas

- **Cache/dataset index alignment** is the highest-risk bug in the project. It
  trains, it converges, it means nothing. `tests/test_cache_alignment.py` renders
  a real cache and re-runs the trunk live on 20 random indices, clean *and*
  degraded.
- **Never shuffle before caching. Never rebuild the manifest afterwards.** Four
  fingerprints are asserted at load and the error names *which* one moved.
- **One `out_dir` per dataset.** `build_cache.py` derives its root as
  `{out_dir}/{detector_name}`, so a second dataset rendered for the same detector
  under the same `out_dir` is rejected on `manifest_sha`.
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
