# The GRACE-freq pipeline, input to output

This document traces the **final GRACE-freq detector** — adapter + frequency
enricher, spliced onto a frozen DINOv3 probe — end to end: what happens to one
image at inference, what produced the three checkpoints that computation
depends on, and the exact commands that build all of it from a raw corpus. It
describes the code as it exists in this tree today; see
[Status against this tree](#status-against-this-tree) for what has and has not
actually been run.

Two other documents this one assumes but does not repeat: the crop/multiscale
protocol (`pipeline.degrade.crop`) and the degradation grid
(`eval_pipeline/configs/degradations.yaml`). Read their module docstrings for
the "why"; this document is the "what, in what order."

## 1. The one-line formula

```
GRACE        logit = head( adapter(trunk(x)) )
GRACE-freq   logit = head( enricher( adapter(trunk(x)), dct(x) ) )
```

`x` is one image. `trunk` is a frozen DINOv3 ViT-S/16, pooled to a 768-d
vector. `adapter` is a ~2M-parameter gated residual MLP trained label-free to
map a degraded feature toward its clean counterpart. `dct(x)` is a patch-DCT
read of the same window `x` at *native pixel resolution*, computed
independently of the trunk. `enricher` is a cross-attention module that fuses
the DCT tokens into the adapter's output. `head` is the detector's own frozen
MLP classifier, untouched since stage 0.

GRACE-freq is implemented by one class, `grace.detectors.fused.FusedDetector`
([fused.py](../grace_adapter/grace/detectors/fused.py)), named in a detector
config like any other harness detector.

## 2. Inference: one image through the fused detector

```
                 ┌──────────────────────────────────────────────┐
                 │              AIGCDataset worker               │
  PIL image  ──► │  degrade (native res) ──► window (input_mode) │
                 │       │                          │             │
                 │       │                    ┌─────┴─────┐       │
                 │       ▼                    ▼           ▼       │
                 │  preprocess_fn()      FreqExtract  (same img)  │
                 │  (224² normalize)     (patch-DCT, native px)   │
                 └───────┬───────────────────────┬────────────────┘
                         │ x: (3,224,224)         │ aux: (196,192)
                         ▼                        │
                 trunk (frozen DINOv3)             │
                         │ f: (768,)               │
                         ▼                        │
                 adapter(f, severity, taps)        │
                         │ f_corrected: (768,)      │
                         ▼                        │
                 enricher(f_corrected, aux, severity) ◄──┘
                         │ fused: (768,)
                         ▼
                 head (frozen MLP)
                         │
                         ▼
                      logit
```

### 2.1 The two reads

Every other detector in this harness sees the image once. GRACE-freq sees it
twice, because the trunk's 224px normalized tensor has already destroyed the
information the frequency branch needs:

- **The spatial read.** `AdaptedDetector`/`FusedDetector.preprocess` resizes
  or crops to 224×224 and normalizes — whatever the base detector's
  `input_mode` says (see §2.2).
- **The frequency read.** `FusedDetector.aux_fn()` returns a
  `pipeline.freq.view.FreqExtract` callable — five ints and two strings, no
  model, picklable — that the dataset (`pipeline.data.dataset.AIGCDataset`)
  calls on the *same degraded PIL image*, before the 224 squash, in the same
  DataLoader worker as the spatial preprocessing
  ([dataset.py](../eval_pipeline/pipeline/data/dataset.py)). Both reads
  happen once per image, in the worker process, while the GPU is free.

The dataset packages the two tensors into one `Inputs` namedtuple
(`x`, `aux`) that `collate` stacks and `.to(device)` moves as a unit. A bare
tensor reaching `FusedDetector.forward` is refused (`TypeError`) rather than
silently scored against zeros.

### 2.2 The window invariant

The spatial branch and the frequency branch **must read the same pixels**, or
the enricher is attending over a different picture than it is correcting.
They choose that window in different places, which is the one thing in this
design that is easy to break silently:

| context | window chosen | how the two branches agree |
|---|---|---|
| training (cache render) | `multiscale_crop`, in the dataset, seeded on `(index, epoch)` | one variable — both branches are handed the identical crop result |
| eval, arm **crop200** | inside `_CropResizePreprocess`, from `input_mode: crop200` | `_freq_geometry()` reads the *same* `VIEWS` table the preprocessing reads: `("crop", 200)` |
| eval, arm **r512** | inside `_ResamplePreprocess`, from `input_mode: resample512` | `_freq_geometry()` reads `("resample", 512)` from the same table |

`VIEWS` lives once, in
[`pipeline/detectors/dinov3.py`](../eval_pipeline/pipeline/detectors/dinov3.py):

```python
VIEWS = {
    "crop200":    ("crop", 200),
    "resample512": ("resample", 512),
}
```

`FusedDetector.__init__` derives `(geometry, size)` from the base detector's
`input_mode` via `_freq_geometry()` and builds `FreqExtract` from it — there
is no second config key that states the window, because a second key is a
second place for the two branches to disagree.
`tests/test_fused_split.py` / `tests/test_freq_view.py` pin this identity.

### 2.3 The spatial branch: trunk → adapter

`self.split.trunk_with_taps(x)` runs the frozen DINOv3 ViT-S/16
(`grace.splits.dinov3.DINOv3Split`) and pools tokens to `f: (768,)` via
`cls+patchmean` (CLS token concatenated with the mean of patch tokens;
registers dropped). Taps (intermediate hidden states, for the ladder variant)
are read here too but are `None` for the plain adapter this document
describes.

`self.adapter` — a `GatedResidualAdapter`
([adapter.py](../grace_adapter/grace/models/adapter.py)) — computes:

```
y = f + g ⊙ MLP(LN(f)),   g = sigmoid(gate_logit)
```

stacked over `n_blocks` residual blocks (bottleneck 256, 2 blocks by
default), with an optional severity-conditioned FiLM on the gate logit. The
severity scalar comes from `self.severity_head(f)` — a small MLP trained
alongside the adapter to *predict* corruption severity from the degraded
feature alone, since no recipe metadata exists at inference. Every block's
final projection is zero-initialized, so an untrained (or checkpoint-less)
adapter returns `f` bit-for-bit — this is what makes `checkpoint: null` (E1)
and the identity gate a tautology rather than an approximation.

### 2.4 The frequency branch: patch-DCT

`pipeline.freq.dct.extract_freq`
([dct.py](../eval_pipeline/pipeline/freq/dct.py)) is a pure numpy function,
torch-free, run identically at render time and at eval time:

```
window (S×S, native px)
  → patch_dct        per 8×8 block, per RGB channel, orthonormal 2D DCT-II
  → |coefficients|    magnitude only — phase does not survive pooling
  → cell_pool         adaptive-average to a fixed 14×14 cell grid
  → radial_order      reorder each block's 64 coefficients by spatial frequency
  → log1p             compress dynamic range for fp16 storage
  → (196, 192) float32
```

- **8×8 patch** — JPEG's own block size, so JPEG block-boundary artefacts
  land on coefficients this basis actually resolves.
- **14×14 cell grid** — fixed regardless of the crop's native size (128–512
  px in this project), so one extraction shape serves every scale the crop
  draw produces, and 196 cells line up 1:1 with DINOv3's 196 patch tokens at
  224.
- **Radial ordering** makes a "frequency band" a *contiguous slice* of the
  192-wide coefficient axis (channel-major: channel `c`'s coefficient `j`
  sits at `c·64 + j`), which is what lets `band_masks` be clean rectangles
  and what lets the top-k coefficient ablation mean "the k lowest
  frequencies" rather than "the k first in raster order."

### 2.5 The enricher: cross-attention fusion

`grace.models.frequency.FrequencyEnricher`
([frequency.py](../grace_adapter/grace/models/frequency.py)):

```
fused = f_corrected + Σ_b  gate_b ⊙ BandExpert_b(freq_tokens, query=f_corrected)
```

- **Two band experts** by default (`n_bands: 2`) — one for high frequency,
  one for low. Each owns its own learnable soft mask over the 192-wide
  coefficient axis (initialized near-hard: `sigmoid(±4)`, complementary —
  the two masks sum to exactly 1 at every coefficient at init), its own
  LayerNorm, its own K/V projections (masking is applied to the *tokens*,
  before mixing — masking attention logits would select *cells*, not
  *frequencies*), a shared query projection from the 768-d seam, standard
  multi-head cross-attention (4 heads, `d_model=256`), and a
  **zero-initialized output projection**.
- **Two experts because damage moves the spectrum in opposite directions**:
  blur/downscale destroy high-frequency energy, noise adds it, JPEG does
  both while shifting energy onto block-aligned coefficients. One expert
  with one gate cannot open for one and shut for the other; two can.
- **Position** is a separable 2D embedding (`row_emb + col_emb`, `2·grid·d`
  params instead of `grid²·d`), added to the *keys* only.
- **Gates** are per-band, per-channel, FiLM-conditioned on the same
  predicted severity scalar the adapter uses.
- **Identity at initialization, exactly**: every expert's output projection
  is zero, so `fused == f_corrected` bit-for-bit at step 0, for *any*
  frequency tokens and *any* severity — no LayerNorm sits on the residual
  sum, because that would rescale `f_corrected` and feed the frozen head a
  space it was never fit on.

### 2.6 The head

`self.split.head(fused)` — the detector's own frozen `LayerNorm → MLP →
logit` (`pipeline.detectors.dinov3.ProbeHead`), fit once in stage 0 on clean
images and never touched by stage 1 or (in the default, frozen-head arm) by
stage 2. This is what makes a GRACE-freq gain attributable to the *features*:
the classifier boundary never moves.

### 2.7 Why GRACE-freq can exceed retention 1.0

The adapter can only rearrange what the trunk kept, and the trunk was fed a
224px resize — whatever that destroyed is gone from its output at *any*
adapter capacity, so GRACE alone is bounded by the clean-feature score
(retention ≤ 1.0 by construction). The DCT branch reads the pixels the trunk
threw away, at native scale, directly — it is not restoring a summary, so it
is not bounded by the summary's ceiling. Passing retention 1.0 is the
strongest available evidence that the frequency branch supplies information
the adapter provably cannot.

## 3. Where the three checkpoints come from

GRACE-freq's inference path depends on exactly three trained artifacts, and
their training order is fixed: stage 0 must exist before stage 1 can
differentiate through the head; stage 1 must be a *finished, frozen* artifact
before stage 2 begins (both GRACE-D and GRACE-freq ship the identical stage-1
adapter weights, which is what keeps "the adapter is trained label-free"
literally true of the shipped model).

```
stage 0 (labels)         stage 1 (label-free)          stage 2 (labels, adapter frozen)
────────────────         ─────────────────────         ────────────────────────────────
train_probe.py           train_adapter.py               train_enrich.py
clean images only    →   f_deg → f_clean (cached)   →   BCE(head(fused), label)
                                                          + λ·‖fused − f_corrected‖
      │                          │                              │
      ▼                          ▼                              ▼
  head.pt                   ema.pt                        enricher.pt
(frozen classifier)     (frozen after this)          (the only thing that trains here)
```

### 3.1 Stage 0 — the probe head

`grace/scripts/train_probe.py` against
[`configs/probe/dinov3_wildfake_multiscale.yaml`](../grace_adapter/configs/probe/dinov3_wildfake_multiscale.yaml).
Fits `ProbeHead` (`LayerNorm → MLP → 1 logit`) on the frozen DINOv3 trunk's
pooled features, **clean images only, multi-scale-crop windows** (same
`CropConfig` the whole pipeline shares). No degradation augmentation — that
is the premise, not an oversight: a head trained under augmentation would
have already solved part of the problem GRACE exists to repair, and its
retention curve would say nothing. Writes
`checkpoints/probe/dinov3_wildfake_multiscale/head.pt`, stamped with
`input_mode: multiscale` so `_assert_head_matches` can refuse loading it into
a detector fed a different pixel scale. This one head is loaded by all three
`input_mode`s used downstream (`multiscale`, `crop200`, `resample512`) —
`HEAD_COMPATIBILITY` permits exactly that fan-out, because a
multi-scale-trained head is *meant* to be scored on both fixed eval arms.

### 3.2 The spatial + frequency feature cache

`grace/scripts/build_cache.py` against
[`configs/cache/dinov3_multiscale.yaml`](../grace_adapter/configs/cache/dinov3_multiscale.yaml)
(train split) and
[`dinov3_multiscale_val.yaml`](../grace_adapter/configs/cache/dinov3_multiscale_val.yaml)
(held-out images). One render, one decode-degrade-crop pass per image,
producing every view of it at once
([writer.py](../grace_adapter/grace/cache/writer.py)):

- **clean** + **12 degraded epochs** + **2 held-out epochs** of the pooled
  768-d spatial feature (`f_deg`, `f_clean`) — fp16 memmap shards.
- The **frequency view** (`freq_deg`, `(196,192)` fp16) rides along in the
  same pass rather than a second render: same detector, same dataset, same
  schedule, same crop seed, so a separate `cache_freq/` would just
  re-decode/re-degrade/re-crop the identical 50k images to recompute
  spatial features that are bit-identical here.
- `recipes.parquet` records each row's degradation label, transforms, and
  severity — the label-free severity target stage 1 trains against.
- `index.npy` is shared across every view in the directory, so row *r* means
  the same image in every view — what makes `(f_deg, f_clean, freq_deg)` for
  one training step a single aligned lookup.

Every render-time knob that changes the *meaning* of a cached byte — the
manifest, the degradation schedule, the detector's preprocessing, the crop
protocol (`crop_sha`), and (new for this branch) the DCT protocol
(`freq_sha`, from `patch`/`grid`/`channels`/`radial`/`norm`) — is
fingerprinted in `spec.json` and asserted on every load
([cache/spec.py](../grace_adapter/grace/cache/spec.py)). View count (how many
epochs are rendered) is resumable and cheap to extend later; the coefficient
set is not — changing `patch`/`grid`/`radial` invalidates every frequency
byte on disk, which is why the frequency protocol ships at its safest
setting (full radially-ordered coefficients) and the *view count* is where
render cost is actually saved (5 epochs for a standalone freq cache; the
merged multiscale+freq cache in this pipeline gets the full 12+2 for free by
riding along).

### 3.3 Stage 1 — the adapter (label-free)

`grace/scripts/train_adapter.py` against
[`configs/train/dinov3_multiscale.yaml`](../grace_adapter/configs/train/dinov3_multiscale.yaml)
(the crop-era reference arm — analogous to `dinov3_clean.yaml` in the
pre-crop era; see [[grace-reference-arm]]). Trains
`GatedResidualAdapter(f_deg) → f_clean` by gradient descent on

```
L = w_err · L_err(adapter(f_deg), f_clean; Jacobian-weighted)
  + w_cos · L_cos
  + λ_sev · L_severity
```

reading only cached features — no image decode, no trunk forward, in
`source: cache` mode (the default). `w_err`/`w_cos` weight a Jacobian-weighted
reconstruction term against a cosine term; the Jacobian weighting scales
per-dimension error by how much the frozen head's decision actually depends
on that dimension (`grace.train.weighting.head_gradient`), which is what
makes the objective care about *decision-relevant* drift rather than
treating all 768 dimensions as equally worth fixing. `L_severity` trains the
auxiliary severity head. AdamW; the gate logit is optionally exempted from
weight decay (`decay_gate: false`) — see [[gate-drift-is-weight-decay]] for
why that matters when reading the trained gate as a health signal. Writes
`checkpoints/grace/dinov3_multiscale/{last,ema}.pt` — an EMA copy of the
weights plus `feature_spec` and `adapter_cfg`, self-contained enough that the
eval harness loads it with no reference to the training run. From here on,
the adapter never trains again: both `train_discrepancy.py` and
`train_enrich.py` load and freeze it (`requires_grad_(False)`,
`split.assert_frozen()` checked every step).

### 3.4 Stage 2 — the frequency enricher (supervised, adapter frozen)

`grace/scripts/train_enrich.py` against
[`configs/train/dinov3_enrich.yaml`](../grace_adapter/configs/train/dinov3_enrich.yaml),
pointed at `adapter_checkpoint: checkpoints/grace/dinov3_multiscale/ema.pt`
and the frequency cache from §3.2. Per step, reads `f_deg`, `freq_deg`,
`label`, `severity` — **no `f_clean`**, because there is no restoration
target here, this is enrichment against real labels:

```python
f_corrected = adapter(f_deg, severity=pred_sev).detach()      # frozen
fused       = enricher(f_corrected, freq_deg, pred_sev)
loss        = BCE(head(fused), label) + lam_anchor * ‖fused - f_corrected‖
```

`lam_anchor` (default 0.1) is the term that keeps this an *enrichment*
rather than a supervised re-fit of the seam — without it, nothing stops the
enricher walking the features to wherever the frozen head happens to
separate this corpus, which would be a content classifier wearing the
adapter's output as a starting point (`lam_anchor: 0` is ablation E15).

**Validation runs before the first optimizer step** (`validation.step_0` in
`summary.json`): at that point every expert's output projection is still
zero, so `fused == f_corrected` bit-for-bit and `auc_fused` must equal
`auc_corrected` — the plain `+grace` arm — to the last decimal.
`train_enrich.py` exits non-zero if this does not hold; it is the same
identity `after_freq.sh` step 8 (E10) checks from the eval-harness side.

Two arms always ship: **`finetune_head: false`** (default — the head stays
the one the baseline was measured with, so a gain is attributable to the
features) and **`finetune_head: true`** (E14 — trains a `deepcopy` of the
head alongside the enricher; the frozen head's ceiling is governed by
`parallel_fraction`, the fraction of feature drift lying along the direction
the frozen head can even see — see §5). Writes
`checkpoints/grace/dinov3_enrich{,_ft}/enricher.pt`: enricher weights, both
`feature_spec` and `freq_spec`, both `EnricherConfig` and `FreqConfig` (both
configs are needed to rebuild — a checkpoint carrying only the enricher's own
config could load cleanly against a cache rendered at a *different* DCT
protocol and attend over the wrong frequencies in silence), plus the
fine-tuned head's state dict when `finetune_head` was set — the head travels
*with* the enricher it was fit alongside, because a detector config naming
one without the other would score a head against features it never saw.

## 4. The eval-time detector configs

Six small YAML files under
[`grace_adapter/configs/detectors/`](../grace_adapter/configs/detectors/),
naming the three checkpoints above by path, differing from each other by
exactly one key. Each is a `target:` dotted-path detector the harness
(`eval_pipeline`) loads and scores identically to every other detector in the
tree.

| config | class | swept key vs. its control |
|---|---|---|
| `dinov3-crop200+grace.yaml` | `AdaptedDetector` | — (adapter only; E12's control) |
| `dinov3-crop200+grace-freq-null.yaml` | `FusedDetector` | `enricher: null` — fresh module, must equal the row above to the last decimal (E10) |
| `dinov3-crop200+grace-freq.yaml` | `FusedDetector` | `enricher: .../dinov3_enrich/enricher.pt` — the trained module (E12) |
| `dinov3-r512+grace.yaml` | `AdaptedDetector` | same three rows, base = `dinov3-wildfake-r512.yaml` instead of `crop200` |
| `dinov3-r512+grace-freq-null.yaml` | `FusedDetector` | " |
| `dinov3-r512+grace-freq.yaml` | `FusedDetector` | " |

All six load the **identical adapter checkpoint**
(`checkpoints/grace/dinov3_multiscale/ema.pt`) — stage 1 is never re-trained
per eval arm — and differ only in `base` (which of the two fixed eval
windows, §5) and `enricher`. This is what makes the six-row, two-column
result table in §5 a controlled comparison: every difference between rows is
exactly the swept key and nothing else.

`FusedDetector.__init__` ([fused.py](../grace_adapter/grace/detectors/fused.py))
assembles the whole graph described in §2 from these four inputs
(`base`, `split`, `checkpoint`, `enricher`) plus the derived window geometry.

## 5. The two evaluation arms

WildFake ships its COCO "real" half pre-resized to a uniform 200×200 while
its DALL-E-3 "fake" half is native ~1024px output, so `max(w, h)` alone
separates the classes at AUC ≈ 1.0000 on the raw benchmark. Any model shown
whole images reads that dimension shortcut instead of content, and a
frequency branch reads it hardest. The fix is two **fixed-size** windows, so
within either arm every image has identical dimensions and the shortcut is
0.5 *by construction* rather than by post-hoc normalization:

- **`crop200`** — centre 200×200 window at native pixel scale (the largest
  square every image in the corpus can natively supply — reals pass through
  untouched, nothing is upsampled). In-distribution for the multi-scale
  trained head; **the informative arm for the frequency branch**.
- **`resample512`** — whole image squashed to 512×512. Out-of-distribution
  (training never squashes a whole image) and spectrally confounded — it
  upsamples the 200×200 reals 2.56× while downsampling the fakes 2×, so the
  reals carry near-zero energy above 200-Nyquist by construction. Read as a
  robustness check, not as evidence about generation traces.

Both arms load the *same* multiscale-trained probe head
(`HEAD_COMPATIBILITY["multiscale"] ⊇ {"crop200", "resample512"}`) and the
same adapter checkpoint; only `input_mode` differs, which is also what
`_freq_geometry()` reads to keep the DCT branch aligned per §2.2.

The headline run,
[`eval_pipeline/configs/runs/dinov3_poc_freq.yaml`](../eval_pipeline/configs/runs/dinov3_poc_freq.yaml),
scores all six detector configs against `wildfake_coco_dalle3` (13,841
images) in one invocation of `run_eval.py`, so the degradation condition
lattice (26 conditions: clean + 19 single-transform + composed pairs/triples)
is built once and every row sees byte-identical degraded images.

## 6. The evaluation harness loop

`pipeline.eval.runner.evaluate_detector`
([runner.py](../eval_pipeline/pipeline/eval/runner.py)) is generic over every
detector in the tree, GRACE-freq included:

1. Score the **clean** condition first — its scores fix the classification
   threshold (max-F1 on clean) and the retention denominator (`auc_clean`).
2. Score every one of 26 degradation conditions (L1 single-transform grid,
   L2/L3 composed), each drawn from the *same* `AIGCDataset`, with
   `aux=detector.aux_fn()` wired through so the frequency branch gets its
   second read for every condition, not just clean.
3. Per condition: AUC, `retention = (auc_deg − 0.5) / (auc_clean − 0.5)`
   ([metrics.py](../eval_pipeline/pipeline/eval/metrics.py)), and an error
   breakdown (FP/FN, precision/recall/F1) at the clean-fixed threshold.
4. Aggregate by transform (mean/worst per grid) and by level (L1/L2/L3
   pooled, with bootstrap CIs and an "interaction gap" against the
   independence prediction from L1 marginals).
5. Write `results/{run_id}__{detector}__{dataset}.json` — the schema
   documented at the top of `runner.py`.

For GRACE-freq specifically, `detector.aux_fn()` returns the `FreqExtract`
from §2.2 rather than `None`, which is the one branch point in this loop
that differs at all between a plain detector and the fused one.

## 7. Reproducing it from a raw corpus

Three driver scripts, run in order, each idempotent (skips any step whose
output already exists) and each stopping hard on its own falsification gate
rather than continuing to produce numbers against a broken premise:

```
scripts/after_fetch.sh   --write-range     # fetch → manifest → crop-range audit
scripts/after_audit.sh                      # refit stage 0 → 32×32 gate → P2′
scripts/after_freq.sh                       # E0-freq → caches → stage 1 → stage 2 → headline run → E10
```

### 7.1 `after_fetch.sh` — the corpus and the crop-range gate

Waits for `fetch_wildfake_train.py`, verifies the corpus is complete
(60,000 images), builds the manifest once
(`data/wildfake_train/manifest.parquet` — never rebuilt afterward, since
every cache checks its hash), then runs
`eval_pipeline/scripts/audit_sizes.py` to compute the largest crop side
every image in the corpus can natively supply
(`results/audit_sizes__wildfake-train.json` →
`recommended_s_max`). **This number has no safe default** — drawing crops
beyond what a class's images can natively supply makes the realized crop
size itself class-conditional (measured: an unaudited 128–512 range scores
an E-cropsize shortcut AUC of 0.9895). `--write-range` writes the audited
`s_max` into the seven configs that carry the crop protocol:

```
configs/probe/dinov3_wildfake_multiscale.yaml
configs/cache/dinov3_multiscale.yaml
configs/cache/dinov3_multiscale_val.yaml
configs/cache/wildfake_freq.yaml
configs/cache/wildfake_freq_val.yaml
configs/train/dinov3_multiscale.yaml
configs/train/dinov3_enrich.yaml
```

See [[crop-range-gates-everything]] — as of the last check this gate was
still open (`s_max` unwritten) and nothing downstream of it can run.

### 7.2 `after_audit.sh` — from an audited range to the first measurement

1. **E-shortcut** — the dimension/container shortcut on both eval arms,
   before any model runs; expected ≈ chance by construction.
2. **P1′** — refit stage 0 (§3.1) on multi-scale crops.
3. **The 32×32 round trip** (**hard stop**) — the head must fall toward
   chance when its input is destroyed by a 32×32 downscale-and-back. A head
   that *survives* this is reading content, not generation traces, and there
   is no damage left for GRACE or the frequency branch to repair.
4. **P2′** — the unadapted baseline's retention curve on both eval arms.
   This is the project's first real measurement: if retention does not
   collapse here, GRACE has nothing to repair and the frequency branch has
   nothing to enrich, established before a single adapter trains.

### 7.3 `after_freq.sh` — the frequency branch itself, 9 steps

1. **E0-freq** (**hard stop**) — do the DCT bands beat the spectral-rolloff
   floor on `wildfake_train_val`? WildFake's reals are COCO downsamples and
   its fakes are not, so the spectrum carries a resampling signature
   unrelated to generation traces; this must be cleared *before* the ~31 GB
   frequency cache is rendered, and it is measured only on the validation
   split, never on the reported benchmark (choosing band geometry against a
   test-set number would be test-set selection).
2. **P3′** — render the crop-era spatial cache, train + val (§3.2, spatial
   only).
3. **S0′** — stage 1, five seeds (§3.3) — establishes the run-to-run noise
   floor every later ablation is read against.
4. **E0-drift** — `parallel_fraction`: the share of feature drift lying
   along the one direction the frozen head can see. Everything orthogonal is
   unrepairable *by construction*, whatever stage 2 learns — quote every
   gain in step 7 against this, not against 1.0.
5. **Render the frequency cache**, train + val (§3.2, adds the DCT view;
   ~31 GB, since a rendered DCT view is ~49× the size of the spatial
   feature it rides beside).
6. **Stage 2** — both enricher arms, frozen head and fine-tuned head (§3.4).
7. **The headline run** — all six detector configs (§4) × both eval arms
   (§5), one `run_eval.py` invocation (§6).
8. **E10** (**hard stop**) — `scripts/compare.py --assert-identity` between
   each `+grace` row and its `+grace-freq-null` row, on both arms. Must
   match to the last decimal (§3.4); if it does not, the enricher is not
   spliced where it claims to be and every number in step 7 is measured
   against a model nobody benchmarked.
9. **Ablations** (skippable with `--skip-ablations`) — E11 (`--n-bands 1`,
   single branch instead of HF/LF), E13 (`--top-k 16` and `--no-pos-emb`),
   E15 (`--lam-anchor 0`) — each a one-key CLI override against the step-6
   reference, read against step 3's five-seed noise floor.

## 8. Reading the result

Per eval arm, three rows from the headline run form the comparison:

```
+grace              retention ≤ 1.0 by construction (restoration only)
+grace-freq-null    must equal the row above exactly (E10, checked in step 8)
+grace-freq         E12: the number that matters, and only a result if the
                    gap from +grace exceeds twice the five-seed spread (S0′)
```

Read `crop200` as the number — it is in-distribution for the multiscale head
and the informative arm for frequency. Read `r512` as a robustness check
only. Read every gain against `parallel_fraction` (step 4 above), not
against 1.0: a small frozen-head ceiling is what makes the fine-tuned-head
arm (E14) the honest headline rather than an afterthought, at the cost of
giving up the frozen head's provenance.

## 9. Status against this tree

This document describes what the code does, not what has been measured. Per
memory recorded alongside this pipeline (see linked notes for detail):

- [[crop-range-gates-everything]] — every config above that carries
  `crop.s_max` refuses to load until `after_fetch.sh --write-range` runs
  against a completed corpus.
- [[grace-eval-harness-blockers]] — no retention number in this project is a
  verified measurement until the P2/P2′ baseline has actually run end to end
  and produced a result JSON.
- [[gate-drift-is-weight-decay]] — do not read the adapter's gate value
  climbing off its init as evidence of learning; measured, that climb is
  decoupled weight decay pulling the logit toward 0, and the objective's own
  pull is the other direction. Applies to `dinov3_multiscale`'s gate exactly
  as it did to the pre-crop reference arm.
- [[grace-reference-arm]] — `dinov3_multiscale.yaml` is the crop-era's single
  reference arm (analogous to `dinov3_clean.yaml` pre-crop); every crop-era
  ablation is that file with exactly one key changed.
- [[stale-adapter-checkpoints]] — a checkpoint saved before `noise_dim` was
  removed from `AdapterConfig` fails to load with a `TypeError`; not a
  concern for a fresh `dinov3_multiscale` run, but relevant if resuming an
  older adapter checkpoint into this pipeline.

## 10. File map

```
eval_pipeline/pipeline/freq/dct.py           patch-DCT arithmetic (pure numpy)
eval_pipeline/pipeline/freq/view.py          FreqExtract — the picklable (image)->tensor wrapper
eval_pipeline/pipeline/detectors/dinov3.py   trunk+head detector, INPUT_MODES, VIEWS
eval_pipeline/pipeline/data/dataset.py       Inputs, AIGCDataset (aux read), collate
eval_pipeline/pipeline/eval/runner.py        the harness loop, result JSON schema
eval_pipeline/pipeline/eval/metrics.py       AUC, retention, thresholding, CIs

grace_adapter/grace/config.py                CropConfig, FreqConfig, EnricherConfig, *TrainConfig
grace_adapter/grace/models/adapter.py        GatedResidualAdapter (stage 1)
grace_adapter/grace/models/frequency.py      FrequencyEnricher, BandExpert (stage 2)
grace_adapter/grace/models/severity.py       SeverityHead
grace_adapter/grace/models/factory.py        build/save/load for adapter and enricher
grace_adapter/grace/splits/dinov3.py         DINOv3Split — the trunk/head seam
grace_adapter/grace/detectors/adapted.py     AdaptedDetector (GRACE, GRACE-D)
grace_adapter/grace/detectors/fused.py       FusedDetector (GRACE-freq) -- §2, §4
grace_adapter/grace/cache/writer.py          MultiViewDataset, build_cache -- §3.2
grace_adapter/grace/cache/spec.py            CacheSpec, fingerprints (crop_sha, freq_sha, ...)
grace_adapter/grace/train/loop.py            train_adapter (stage 1), train_enrich (stage 2)

grace_adapter/configs/probe/dinov3_wildfake_multiscale.yaml    stage 0
grace_adapter/configs/cache/dinov3_multiscale{,_val}.yaml      spatial+freq cache, train/val
grace_adapter/configs/cache/wildfake_freq{,_val}.yaml          standalone freq-only cache (unused by this pipeline's merged render; kept as the freq-alone protocol)
grace_adapter/configs/train/dinov3_multiscale.yaml             stage 1
grace_adapter/configs/train/dinov3_enrich.yaml                 stage 2
grace_adapter/configs/detectors/dinov3-{crop200,r512}+grace{,-freq,-freq-null}.yaml   §4
eval_pipeline/configs/detectors/dinov3-wildfake-{multiscale,crop200,r512}.yaml        §5
eval_pipeline/configs/runs/dinov3_poc_freq.yaml                 the headline run, §7.3 step 7

grace_adapter/scripts/after_fetch.sh    §7.1
grace_adapter/scripts/after_audit.sh    §7.2
grace_adapter/scripts/after_freq.sh     §7.3
```
