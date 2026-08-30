#!/usr/bin/env bash
# The third and last leg: from a measured baseline to the frequency-branch result.
#
#   bash scripts/after_freq.sh                 # the whole thing
#   bash scripts/after_freq.sh --list          # print the plan and exit
#   bash scripts/after_freq.sh --from 5        # resume at step 5
#   bash scripts/after_freq.sh --only 3        # just step 3
#   bash scripts/after_freq.sh --smoke         # 2 epochs / 2 cache views -- wiring only
#   bash scripts/after_freq.sh --skip-ablations
#   WANDB=1 bash scripts/after_freq.sh
#
# Run after scripts/after_audit.sh. That one leaves you with an audited crop
# range, a head refit on multi-scale crops, the 32x32 gate, and P2' -- the
# project's first retention measurement. This one spends the ~31 GB.
#
#   1  E0-freq   do the bands beat the rolloff floor?     <- HARD STOP
#   2  render the crop-era SPATIAL caches (train + val)
#   3  S0'       stage 1 under the crop protocol, 5 seeds
#   4  E0-drift  parallel_fraction -- the frozen-head CEILING
#   5  render the FREQUENCY caches (train + val)          ~31 GB
#   6  stage 2   the enricher; frozen head AND fine-tuned (E14)
#   7  score every arm through the harness, both eval arms
#   8  E10       the null enricher must reproduce +grace  <- HARD STOP
#   9  the ablations: E11, E13, E15
#
# TWO HARD STOPS, and both are cheaper than what they guard.
#
# Step 1 is DECISION 0. WildFake's reals are COCO images the packager downsampled
# to 200x200 and its fakes are untouched native pixels, so the spectrum carries a
# resampling signature that survives cropping and that a frequency branch reads
# perfectly. E-rolloff measures exactly that channel; if the bands do not clear
# it, whatever they separate is resampling history and not generation traces.
# That is a finding, and it arrives before a byte of the 31 GB is written.
#
# Step 8 is E10. Every expert's output projection is zero-initialised, so a fresh
# enricher is EXACTLY the identity and the null arm must reproduce +grace to the
# last decimal, through the full harness. If it does not, the module is not
# spliced where it claims to be and step 7's numbers are for a model nobody
# benchmarked. `train_enrich.py` makes the same check at step 0 of training, from
# the other side; this one also exercises the aux pathway -- the DCT in the
# workers, `Inputs` across the collate, the tokens reaching `forward`.
#
# IDEMPOTENT, like run_all.sh: every step checks for its own output and skips.
# Caches resume one view at a time; a training run is skipped if its summary.json
# exists; a harness run is skipped if its result JSON does. Re-run freely.
#
# WHY THIS IS NOT PART OF run_all.sh. That script is the WHOLE-IMAGE pipeline: it
# renders cache/, trains dinov3_clean, and every one of its ablations is that run
# with one key changed. The crop era has its own reference arm
# (dinov3_multiscale), its own caches and its own noise floor, and mixing the two
# in one script would invite reading a retention number from one protocol against
# a ceiling from the other. `crop_sha` refuses that at the file level; this keeps
# it out of the command line too.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export PYTHONIOENCODING=utf-8       # Windows consoles default to cp1252

if [[ -n "${PYTHON:-}" ]]; then :
elif [[ -x ../.venv/Scripts/python.exe ]]; then PYTHON=../.venv/Scripts/python.exe
elif [[ -x ../.venv/bin/python ]];        then PYTHON=../.venv/bin/python
else PYTHON=python
fi
# See after_fetch.sh: Python on Windows emits \r\n, `$(...)` strips only the \n,
# and the surviving \r mangles every message quoting this path.
PY_ABS=$("$PYTHON" -c "import sys; print(sys.executable)" | tr -d '\r')

SMOKE=(); CACHE_EPOCHS=(); FROM=0; ONLY=""; FORCE=0; LIST=0; SKIP_ABL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)          SMOKE=(--epochs 2); CACHE_EPOCHS=(--epochs 2); shift ;;
    --from)           FROM="$2"; shift 2 ;;
    --only)           ONLY="$2"; shift 2 ;;
    --force)          FORCE=1; shift ;;
    --list)           LIST=1; shift ;;
    --skip-ablations) SKIP_ABL=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

# Tracking is a property of the INVOCATION, not of the config files. Several
# ship `wandb.enabled: true`, so without an explicit --no-wandb a plain run would
# reach the network once per stage. summary.json is the record either way.
if [[ "${WANDB:-0}" == "1" ]]; then
  WB=(--wandb --wandb-group dinov3_freq); TRACKING=1
else
  WB=(--no-wandb); TRACKING=0
fi

N=0
step() {
  N=$((N + 1))
  if (( LIST )); then printf '%2d  %s\n' "$N" "$1"; return 1; fi
  if [[ -n "$ONLY" && "$ONLY" != "$N" ]]; then return 1; fi
  if (( N < FROM )); then printf '\033[2m-- %2d  %s (skipped: --from %s)\033[0m\n' "$N" "$1" "$FROM"; return 1; fi
  printf '\n\033[1m== %2d  %s\033[0m\n' "$N" "$1"
  return 0
}
have() { (( FORCE )) && return 1; [[ -e "$1" ]]; }
note() { printf '   \033[2m%s\033[0m\n' "$*"; }
die()  { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

EV=../eval_pipeline
R=$EV/results
CKPT=checkpoints/grace
MS_CACHE=cache_ms/dinov3-wildfake-multiscale
FREQ_CACHE=cache_freq/dinov3-wildfake-multiscale
ARMS=(crop200 r512)

stage1() {                       # stage1 <run_id> <config> [extra...]
  local run="$1" cfg="$2"; shift 2
  if have "$CKPT/$run/summary.json"; then note "$run already trained -- skipping"; return; fi
  "$PY_ABS" scripts/train_adapter.py "$cfg" "${SMOKE[@]}" "${WB[@]}" "$@" \
    || die "stage 1 failed: $run"
}

enrich() {                       # enrich <run_id> <config> [extra...]
  local run="$1" cfg="$2"; shift 2
  if have "$CKPT/$run/summary.json"; then note "$run already trained -- skipping"; return; fi
  # train_enrich.py exits non-zero if the step-0 identity does not hold, which is
  # E10 from inside the training run. Do not soften this into a warning.
  "$PY_ABS" scripts/train_enrich.py "$cfg" "${SMOKE[@]}" "${WB[@]}" --run-id "$run" "$@" \
    || die "stage 2 failed: $run"
}

render() {                       # render <config>
  # --dry-run first, always: it prints the on-disk size, and the frequency view
  # is ~49x the features. Finding that out after an hour is avoidable.
  "$PY_ABS" scripts/build_cache.py "$1" --dry-run || die "cannot plan $1"
  "$PY_ABS" scripts/build_cache.py "$1" "${CACHE_EPOCHS[@]}" || die "render failed: $1"
}

# ------------------------------------------------------ 0  the gate is set ----
if (( ! LIST )); then
  "$PY_ABS" - <<'PY' || die "run scripts/after_fetch.sh first -- the crop range is not written"
import sys
sys.path[:0] = [".", "../eval_pipeline"]
from grace.config import load_cache_config, load_enrich_config, load_probe_config
# Every config that carries the window protocol, loaded together: CropConfig
# raises on `enabled: true` with no `s_max`, so this is the audit gate applied to
# all of them at once rather than discovered one script at a time.
cfgs = [
    load_probe_config("configs/probe/dinov3_wildfake_multiscale.yaml").crop,
    load_cache_config("configs/cache/dinov3_multiscale.yaml").crop,
    load_cache_config("configs/cache/dinov3_multiscale_val.yaml").crop,
    load_cache_config("configs/cache/wildfake_freq.yaml").crop,
    load_cache_config("configs/cache/wildfake_freq_val.yaml").crop,
    load_enrich_config("configs/train/dinov3_enrich.yaml").crop,
]
ranges = {(c.s_min, c.s_max, c.policy, c.seed) for c in cfgs}
if len(ranges) != 1:
    raise SystemExit(f"the crop range differs across configs: {sorted(ranges)}")
(s_min, s_max, policy, seed), = ranges
print(f"   crop {s_min}-{s_max}px {policy}, seed {seed}  (agreed across 6 configs)")
PY
fi

# --------------------------------------------------------- 1  E0-freq --------
if step "E0-freq  do the bands beat the rolloff floor?  (HARD STOP)"; then
  # Runs on wildfake_train_val, NEVER on the reported benchmark: band count and
  # patch size are choices, and choosing them against a test-set number is
  # selection on the test set.
  OUT=results/freq_analysis__wildfake-train-val.json
  if have "$OUT"; then note "$OUT exists -- skipping"
  else
    "$PY_ABS" scripts/analyze_freq.py --out "$OUT" || die "DECISION 0: STOP.
The band energies do not clear the spectral-rolloff floor, so what the spectrum
separates here is WildFake's resampling history -- its reals are COCO images
downsampled to 200x200, its fakes are untouched native pixels -- and not
generation traces. Cropping does not remove that; it is in the pixels.

Read $OUT. \`per_band\` says which bands carry
anything, \`signatures\` says whether blur, noise and JPEG move the spectrum in
the directions physics requires, and \`signature_spread\` says whether the bands
are eight numbers or one wearing a hat.

The re-scope, if you want one: feed band ENERGY DISCREPANCY (clean vs degraded)
to the GRACE-D head instead of building a cross-attention branch over a channel
that turns out to be a packaging artifact. That is a much smaller change and it
is not what this script continues into."
  fi
fi

# ------------------------------------------- 2  the crop-era spatial cache ----
if step "P3' render the crop-era SPATIAL caches -- train + val"; then
  # Re-rendered, not reused: the trunk now sees 128-S_max px windows at native
  # pixel scale, which is a different feature space and not a harder version of
  # the same one. `crop_sha` refuses the old cache outright, which is the point.
  render configs/cache/dinov3_multiscale.yaml
  render configs/cache/dinov3_multiscale_val.yaml
fi

# ----------------------------------------- 3  the crop-era reference arm ------
if step "S0' stage 1 under the crop protocol, five seeds -- and the NOISE FLOOR"; then
  # Five seeds are the unit of measurement, not a robustness gesture. On the
  # round-1 data the hard-split delta AUC moved by ~0.0007 on the seed alone, so
  # an arm that shifts a metric by less than about twice that has shifted
  # nothing -- and every ablation in step 9 is read against this number.
  stage1 dinov3_multiscale configs/train/dinov3_multiscale.yaml
  for s in 1 2 3 4; do
    stage1 "dinov3_multiscale_s$s" configs/train/dinov3_multiscale.yaml \
      --seed "$s" --run-id "dinov3_multiscale_s$s"
  done
  "$PY_ABS" scripts/seed_stats.py dinov3_multiscale 'dinov3_multiscale_s*' || true
fi

# ---------------------------------------------------------- 4  E0-drift ------
if step "E0  drift geometry on the CROP-ERA cache -- the frozen-head CEILING"; then
  # READ THIS BEFORE ANY GAIN IN STEP 7. `parallel_fraction` is the share of the
  # drift lying along the one direction the frozen head can see; everything
  # orthogonal is unrepairable by construction. Round 1 measured 0.0298 on NTIRE
  # -- 97% invisible -- and if that holds here, a FROZEN head reduces the whole
  # cross-attention enricher to a scalar logit shift, which is GRACE-D's function
  # class at a thousand times the parameters. That is what makes E14's
  # fine-tuned-head arm a first-class arm rather than an afterthought.
  OUT=results/dinov3_poc_drift__wildfake-train-multiscale.json
  if have "$OUT"; then note "$OUT exists -- skipping"
  else
    "$PY_ABS" scripts/analyze_drift.py \
      --cache    "$MS_CACHE" \
      --dataset  $EV/configs/datasets/wildfake_train.yaml \
      --detector $EV/configs/detectors/dinov3-wildfake-multiscale.yaml \
      --split    grace.splits.dinov3.DINOv3Split \
      --out      "$OUT" || die "drift analysis failed"
  fi
  "$PY_ABS" -c "
import json,sys
d=json.load(open('$OUT'))
pf=d.get('overall',{}).get('parallel_fraction')
print(f'   parallel_fraction = {pf}')
print('   -> a frozen head can only read this fraction of the damage. Read every')
print('      step-7 gain against it, and expect E14 to matter if it is small.')
" || true
fi

# ------------------------------------------------- 5  the frequency cache ----
if step "render the FREQUENCY caches -- features AND the DCT view, ~31 GB"; then
  # Both views in one root: stage 2 reads f_deg and freq_deg for the same row in
  # the same step, and `index.npy` is shared by every view in the directory, so
  # their alignment is structural rather than asserted.
  #
  # Clean + 5 degraded + 1 held out, not 15. Views are RESUMABLE -- epochs can be
  # added later at zero rework -- and the coefficient set is not, so the saving
  # is taken on the reversible knob and the irreversible one ships at its safest
  # setting.
  render configs/cache/wildfake_freq.yaml
  render configs/cache/wildfake_freq_val.yaml
fi

# ---------------------------------------------------------- 6  stage 2 -------
if step "stage 2 -- the enricher, frozen head AND fine-tuned head (E14)"; then
  # The adapter is frozen and bit-identical to stage 1's, so GRACE and
  # GRACE-freq ship the same adapter weights and "the adapter is trained without
  # labels" stays literally true of the shipped artifact.
  #
  # Both E14 arms, always. Frozen is the default and the one whose provenance is
  # worth something -- it is the head the baseline was measured with, so a gain
  # is attributable to the features. Fine-tuned gives that up and is the only arm
  # that can exploit drift the frozen head cannot see. Which is the honest
  # headline depends on step 4's number, and neither is knowable without both.
  enrich dinov3_enrich    configs/train/dinov3_enrich.yaml
  enrich dinov3_enrich_ft configs/train/dinov3_enrich.yaml --finetune-head
fi

# ----------------------------------------------------------- 7  the run ------
if step "THE HEADLINE -- six arms x two eval arms, one condition lattice"; then
  # One invocation for all six so the 26 conditions are built once: they are
  # seeded on the image index, so every detector sees byte-identical degraded
  # images and a difference between two rows is never a difference in the draw.
  FIRST=$R/dinov3_poc_freq__dinov3-crop200+grace__wildfake-coco-dalle3.json
  if have "$FIRST"; then note "already scored -- skipping"
  else
    ( cd "$EV" && "$PY_ABS" scripts/run_eval.py \
        --config configs/runs/dinov3_poc_freq.yaml ) || die "the headline run failed"
  fi
fi

# ------------------------------------------------------------- 8  E10 --------
if step "E10 THE GATE -- the null enricher must reproduce +grace exactly"; then
  for arm in "${ARMS[@]}"; do
    BASE=$R/dinov3_poc_freq__dinov3-${arm}+grace__wildfake-coco-dalle3.json
    NULL=$R/dinov3_poc_freq__dinov3-${arm}+grace-freq-null__wildfake-coco-dalle3.json
    [[ -e "$BASE" && -e "$NULL" ]] || die "missing result JSONs for arm $arm -- run step 7"
    "$PY_ABS" scripts/compare.py --baseline "$BASE" --adapted "$NULL" --assert-identity \
      || die "E10 FAILED on arm $arm.
A freshly built enricher has zero-initialised output projections, so it is
EXACTLY the identity and this arm must reproduce +grace to the last decimal. It
did not, which means the module is not spliced where it claims to be -- or the
aux pathway (the DCT in the workers, Inputs across the collate, the tokens
reaching forward) is handing it something other than what it was trained on.

Every number in step 7 is therefore against a model nobody benchmarked. Stop."
  done
  note "E10 passed on both arms -- the enrichment is attributable from here on"
fi

# ---------------------------------------------------------- 9  ablations -----
if step "the ablations -- E11 (band split), E13 (coefficients), E15 (anchor)"; then
  if (( SKIP_ABL )); then note "--skip-ablations"; else
    # One key each against the step-6 reference arm, expressed as CLI overrides
    # rather than four near-identical config files -- the same reason the seed
    # replicates in step 3 are. Read every one of them against step 3's seed
    # floor: an arm that moves a metric by less than about twice that spread has
    # moved nothing, whatever its sign.
    ENRICH=configs/train/dinov3_enrich.yaml
    enrich dinov3_enrich_e11_single   "$ENRICH" --n-bands 1     # one branch, not HF/LF
    enrich dinov3_enrich_e13_topk16   "$ENRICH" --top-k 16      # 16 lowest coeffs/channel
    enrich dinov3_enrich_e13_nopos    "$ENRICH" --no-pos-emb    # cells become a set
    enrich dinov3_enrich_e15_noanchor "$ENRICH" --lam-anchor 0  # no anchor term
    note "compare checkpoints/grace/dinov3_enrich*/summary.json -> validation"
    note "held_out_images/*: auc_fused against the reference arm's, per epoch."
  fi
fi

TOTAL=$N
if (( LIST )); then echo; echo "$TOTAL steps. bash scripts/after_freq.sh --from N to resume."; exit 0; fi

cat <<EOF

$(printf '\033[1m== done. What to read, and in what order\033[0m')

1. WAS THE MECHANISM REAL AT ALL?
   results/freq_analysis__wildfake-train-val.json

   \`margin_over_rolloff\` is the whole of it. The bands have to beat spectral
   rolloff, because WildFake's reals are downsamples and its fakes are not, and
   that difference is in the pixels where no crop reaches it.

2. WHAT WAS THE MOST A FROZEN HEAD COULD HAVE DONE?
   results/dinov3_poc_drift__wildfake-train-multiscale.json -> parallel_fraction

   Quote every gain against this, not against 1.0. If it is small, report the
   fine-tuned-head arm as the headline and say plainly what it gave up.

3. DID THE ENRICHMENT DO ANYTHING?
   $R/dinov3_poc_freq__*.json

   Three rows per eval arm. +grace-freq-null must equal +grace exactly (step 8
   checked it). +grace-freq minus +grace is E12, and it is only a result if it
   exceeds twice the seed spread from step 3.

   Read arm (a) -- crop200 -- as the number. Arm (b) is out of distribution for a
   crop-trained head and spectrally confounded besides: it upsamples the reals
   2.56x and downsamples the fakes 2x, so the rolloff floor sits very high there.
   It is a robustness check, not a comparison.

4. CAN IT EXCEED RETENTION 1.0?
   scripts/compare.py normalises by the BASELINE's clean AUC. A restorer cannot
   pass 1.0 -- the clean-image score is its ceiling. This arm can, because the
   DCT reads pixels the trunk's resize threw away, and that is information the
   clean FEATURES do not contain. Passing it is the strongest available evidence
   that the frequency branch is doing something the adapter cannot.

5. WHAT IT DOES NOT SHOW.
   The model never sees a whole image, so global composition, whole-image colour
   statistics and aspect ratio are gone -- deliberately, since that is where the
   dimension shortcut lived. Any comparison against a published whole-image
   detector is not like-for-like: they are scored on a strictly larger input. And
   a degradation whose damage is global is only seen here through its local
   residue, so this is a local-trace robustness curve, which is the claim to
   make.
EOF
