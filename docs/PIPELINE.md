# Model Pipeline

**How does the model turn a single input image into a confidence score that the image is AI-generated rather than real?** The pipeline is built around a frozen image classifier (`trunk` + `head`) that was trained on clean images and collapses on degraded ones. Around it sit three corrective components: a **severity head**, a **GRACE adapter**, and a **frequency enricher**. The key design idea is that the image is read **twice**, over the same window: once as a normalized 224px tensor for the spatial branch, and once at native pixel scale for the frequency branch. Generation traces are a local high-frequency phenomenon that resizing destroys, so the information the spatial branch loses must be recovered from a separate read.

```
image
 ├─ spatial branch ── preprocess ──> trunk ──> f ──────> severity head ──> s
 └─ freq branch ──── patch-DCT ──> tokens                        │
                                                                 v
 f ──> GRACE adapter ──> f_corrected ──> frequency enricher ──> fused ──> head ──> logit ──> sigmoid ──> score
```

## What each component does

### 1. Preprocessing
Both branches start from the same image window (a crop of the source image, chosen by the detector's `input_mode`). The spatial branch normalizes it into a fixed 224x224 tensor; the frequency branch keeps the raw pixels at their native scale.

### 2. Trunk (frozen DINOv3 ViT-S/16)
The frozen vision backbone. It embeds the 224px tensor into a single 768-dimension feature vector `f` per image (pooled from the CLS and patch tokens). This feature is the detector's internal representation of the image — the "seam" every later component operates on.

### 3. Severity head
A small MLP that maps the trunk feature `f` to a scalar `s ∈ [0,1]` estimating how corrupted/degraded the image is. It never sees labels; it predicts how far the image has drifted from clean. Both the adapter and the enricher use `s` to scale how much correction to apply.

### 4. GRACE adapter (GatedResidualAdapter)
The stage-1 component. It corrects the degraded trunk feature back toward what a clean image would produce:

```
f_corrected = f + g ⊙ MLP(LN(f)),   g = sigmoid(gate_logit)
```

A learned gate `g` decides how much of the proposed correction to apply per channel, and that gate is modulated by the predicted severity `s` (via FiLM), so a mildly degraded image gets a light touch and a heavily degraded one gets more. At initialization the adapter is exactly the identity — it changes nothing until trained.

### 5. Frequency enricher (FrequencyEnricher)
The stage-2 component, what makes this the GRACE-freq variant. It re-reads the image where the trunk cannot. Its input tokens come from a patch-DCT of the same window at native pixel scale: per-8x8-block DCT-II (8 is the JPEG block size), coefficient magnitudes pooled onto a 14x14 cell grid and log-compressed — giving one token per DINOv3 patch cell. Two band experts (low- and high-frequency) run multi-head cross-attention with the corrected feature as query and the DCT cells as keys/values, each band masked to its own slice of the spectrum. The experts' outputs are gated and added as a residual:

```
fused = f_corrected + Σ_b g_b · BandExpert_b(tokens, f_corrected)
```

This lets the model inject frequency-domain evidence (blur, noise, JPEG artifacts) that the spatial features no longer contain. Like the adapter, it is identity at initialization.

### 6. Head (frozen probe)
A frozen MLP probe, trained on clean features only. It maps the final feature vector to a single logit. Frozen means the upstream components must fix the features to fit the space this head already knows.

### 7. Output
The logit is passed through a sigmoid to yield the confidence score in `[0,1]` — how likely the image is AI-generated.

## Variants
- **Baseline**: no adapter, no enricher — the raw `head(trunk(x))`.
- **GRACE**: no frequency branch — `logit = head(adapter(trunk(x), s))`.