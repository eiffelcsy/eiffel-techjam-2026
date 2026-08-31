# Data

**What was this model trained on in the experiments?** A custom-sampled subset of the **WildFake** dataset (AIGC-generated image detection corpus) from ModelScope, plus a held-out WildFake benchmark subset for testing.

- **Source**: https://modelscope.cn/datasets/hy2628982280/WildFake/summary
- **License**: Apache 2.0 (individual constituent corpora carry their own terms)

## Training data

A weighted subset sampled from the full WildFake corpus (`data/wildfake_train/manifest.parquet`), not the dataset as-is:

| Split | Images | Real | Fake |
|---|---|---|---|
| `train` | 49,999 | 14,191 | 35,808 |
| `validation` | 10,000 | 5,000 | 5,000 |

Reals come from LAION-5B and ImageNet. The fake half is **weighted toward recent diffusion-based models over GANs** (50/35/15 by tier, ~85% diffusion / ~15% GAN in the realized sample): Stable Diffusion and Midjourney dominate, with a minority of GANs (DF-GAN, GALIP, styleGAN, GigaGAN, BigGAN, starGAN). This is a deliberate deviation from the corpus's natural distribution, which is dominated by older SD v1.5/v2.0 material.

## Test / benchmark data

The primary evaluation benchmark (`data/wildfake_test/manifest.parquet`): **13,841 images = COCO val2017 reals (4,998) + DALL-E 3 fakes (8,843)**. Training and test are disjoint by construction (`DALLE` and `coco` are excluded from the training sample).

## Leakage guard

Everything drawn from WildFake for training excludes the benchmark's strata (`DALLE`, `coco`), so no training image shares a generator or real source with the reported test set.

## Preprocessing: the degradation grid

Degradation is the whole point of the project — robustness is measured by how well a detector keeps working after an image has been *handled*, not attacked. The grid (`preprocessing/configs/degradations.yaml`) is therefore eleven real-world image-handling artefacts, each with an explicit parameter list, never an adversarial perturbation:

| Transform | Parameter | Values | Real-world analog |
|---|---|---|---|
| `jpeg` | quality | 90, 70, 50, 30 | social re-encode |
| `gaussian_blur` | sigma | 0.5, 1.0, 2.0 | out-of-focus capture |
| `resize` | scale | 0.5, 0.25 | thumbnailing |
| `gaussian_noise` | sigma | 0.02, 0.05, 0.10 | low-light sensor |
| `brightness_up/down` | factor | 1.2 / 0.8 | filter / auto-enhance |
| `contrast_up/down` | factor | 1.2 / 0.8 | filter / auto-enhance |
| `saturation_up/down` | factor | 1.2 / 0.8 | filter / auto-enhance |
| `center_crop` | keep | 0.8 | profile-pic framing |

The six photometric entries are colour jitter *unbundled* into individual ±20% steps so each is its own attributable condition, rather than one averaged "jitter" number. Every transform is size-preserving (so they compose in any order and one grid works on mixed-resolution images), and only `gaussian_noise` draws randomness; the rest are deterministic given their parameter.

### The four composition levels

Degradations are applied in four severity levels, which is what separates "why does it fail" from "how much will it fail":

- **L0 — clean.** The reference. Sets the detection threshold and the retention denominator.
- **L1 — single.** One transform at one parameter, the full deterministic grid (19 conditions). One-factor-at-a-time: every drop is attributable to exactly one cause.
- **L2 — pair.** Two *distinct* transforms, parameters drawn uniformly, applied in random order.
- **L3 — multi.** Three to five distinct transforms, sampled the same way.

L1 is a controlled sweep; L2/L3 are Monte-Carlo samples of the composition space (too many combinations to enumerate). The pair is the point: L1 says *why* a detector fails, L2/L3 say *how much* it fails in deployment, and the gap between them is the interaction effect (`interaction_gap`).

Sampling rules make every draw reproducible: transforms are drawn without replacement, parameters are drawn from each transform's own grid, and order is shuffled (these transforms do not commute — blur-then-JPEG leaves a different trace than JPEG-then-blur). Each draw is keyed on `(image index, level, replicate)`, so the degraded set is *bit-identical across runs, machines, and detectors* — the feature cache would be meaningless otherwise.

### How training uses the grid

Training draws degradations as a *distribution* rather than the fixed L1 sweep, using the same deterministic machinery with `epoch` standing in for `replicate` — so epoch 7's corruption of image 412 is computable offline, without running epochs 0–6, which is exactly what makes the offline feature cache possible. Composition levels are weighted **15% / 35% / 30% / 20%** (clean / single / pair / multi). Clean stays in the mix at ~15% on purpose: on those steps the correction target equals the input, so the correct behaviour is to do nothing — that implicit identity constraint is what anchors the adapter to a no-op on clean inputs.

The *severity* of each draw is also computable for free: a transform's parameter gets a normalised rank within its own (mild→severe) grid, combined equally with composition depth into a scalar in `[0,1]`. This is the supervision signal for the severity head, which never sees labels.

### The crop

Degradation is applied at native resolution, and only afterwards is a **multi-scale crop** drawn (128–256px, uniform) before the 224px squash. The crop removes the resolution shortcut — WildFake ships its COCO reals pre-resized to exactly 200×200 while the DALL-E 3 fakes are native 1024px, so whole-image dimensions alone score ~AUC 0.9997. Cropping means the model never sees a whole image, so there is no global resolution to read, and the generation traces it must find are scale-local. The trade is real (global composition and colour statistics are gone), but it runs in the direction the project cares about: generation traces are a local, high-frequency phenomenon that a crop preserves and a whole-image resize destroys. The crop is drawn deterministically per image (seeded on the index) and kept fixed across epochs, because stage-1 training pairs a degraded feature against the *same window's* clean feature — a per-epoch window would be mapping one window onto a different one, which is not a restoration task.

## Evaluation pipeline data

Evaluation scores a fixed subset once per condition over a single **condition lattice**, built once and shared by every detector in a run:

- **26 conditions**: clean + the 19 L1 grid points + L2 and L3 at 3 independent replicates each. Replicates re-draw the composed levels over the same images to tighten the confidence interval and widen coverage of the composition space without growing the eval set.
- The lattice is seeded on the image index, so **every detector sees byte-identical degraded images** — a difference between two detectors is never a difference in the draw.
- Clean is scored first and sets the threshold and the retention denominator; every other condition scores the same images in the same (manifest) order, so clean and degraded scores are **paired per image**.
- **Retention** is the degraded AUC normalised by the detector's own clean AUC — the headline robustness metric.
- At L2/L3 the per-image recipe is logged, turning the composed levels from a single number into an analysable sample of the composition space (which pairs hurt most, whether order mattered, whether one step dominates the drop).

Two deterministic evaluation arms are used, fixed rather than drawn so the dimension shortcut is 0.5 by construction:

- **crop200** — a 200×200 window at native pixel scale (the largest window every benchmark image supplies from its own pixels). The informative arm.
- **r512** — the whole image resampled to 512×512. Out of distribution and spectrally confounded; a scale-robustness check.

Per-image metadata (index, label, generator, condition, recipe) travels with every score, so results are sliceable by generator and by transform — the per-generator breakdown in `summary.json` is the project's replacement for a "hard" validation split.
