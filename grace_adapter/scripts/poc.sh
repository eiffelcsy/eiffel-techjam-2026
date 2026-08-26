#!/usr/bin/env bash
# The proof-of-concept path, end to end, in one command.
#
#   bash scripts/poc.sh              # the full run
#   bash scripts/poc.sh --smoke      # 2 epochs, 2 cache views -- minutes, proves wiring
#
# Runs from grace_adapter/. Every step is idempotent: the manifest is skipped if
# it exists, the cache is resumable at view granularity, and re-running a
# training stage overwrites its own run directory and nothing else.
#
# Prerequisite, once: the DINOv3 backbone is a licence-gated Hub repo. Accept it
# at https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m and run
# `hf auth login`. Or point `backbone_id` in
# ../eval_pipeline/configs/detectors/dinov3-sid.yaml at a mirror you have.
#
# Set WANDB=1 to track every stage under one group.
set -euo pipefail

cd "$(dirname "$0")/.."
HARNESS=../eval_pipeline

SMOKE=""; CACHE_EPOCHS=""
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE="--epochs 2"
  CACHE_EPOCHS="--epochs 2"
fi

WB=()
if [[ "${WANDB:-0}" == "1" ]]; then
  WB=(--wandb --wandb-group dinov3_poc)
fi

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------------------
step "0/7  dataset -- SID_Set train + validation into one manifest"
# Two splits, one table, disjoint image directories. Skipped if already built:
# rebuilding it changes `manifest_sha` and invalidates every cached feature.
# `data/` is at the REPO ROOT, not inside either package: both read it, and a
# dataset config's `../data/...` resolves to the same directory whether the CWD
# is eval_pipeline/ or grace_adapter/.
( cd "$HARNESS"
  if [[ -f ../data/sid_poc/manifest.parquet ]]; then
    echo "   ../data/sid_poc/manifest.parquet exists -- skipping (never rebuild a"
    echo "   manifest a cache was rendered against)"
  else
    python scripts/build_manifest.py --config configs/datasets/sid_poc.yaml
  fi )

step "1/7  stage 0 -- fit the MLP head on CLEAN features"
# The one step that trains a detector. Writes to the head_checkpoint path the
# detector config names, so nothing else needs telling where it went.
python scripts/train_probe.py configs/probe/dinov3_sid.yaml $SMOKE "${WB[@]}"

step "2/7  baseline -- the retention curve GRACE is measured against"
# Run before training an adapter: if retention does not collapse here, there is
# no gap to close and the PoC has answered its own question early.
( cd "$HARNESS" && python scripts/run_eval.py --config configs/runs/dinov3_poc_baseline.yaml )

step "3/7  render the feature cache -- clean view + one view per epoch"
python scripts/build_cache.py configs/cache/dinov3.yaml --dry-run
python scripts/build_cache.py configs/cache/dinov3.yaml $CACHE_EPOCHS

step "4/7  E0 -- does RA-Det's drift asymmetry hold on this data?"
# AFTER the render, not before it: the analysis is a comparison of the clean view
# against a degraded one, so there is nothing to compute until both exist. It
# still comes before anything is *trained*, which is the sense in which it is
# "first" -- both outcomes are useful and one of them saves a day.
python scripts/analyze_drift.py \
  --cache cache/dinov3-sid \
  --dataset ../eval_pipeline/configs/datasets/sid_poc.yaml \
  --detector ../eval_pipeline/configs/detectors/dinov3-sid.yaml \
  --split grace.splits.dinov3.DINOv3Split \
  --out results/dinov3_poc_drift.json

step "5/7  stage 1 -- the label-free adapter (arm B), and its control (arm A)"
python scripts/train_adapter.py configs/train/dinov3_clean.yaml $SMOKE "${WB[@]}"
python scripts/train_adapter.py configs/train/dinov3_degraded.yaml $SMOKE "${WB[@]}"

step "6/7  stage 2 -- the supervised discrepancy head, adapter frozen"
python scripts/train_discrepancy.py configs/train/dinov3_discrepancy.yaml "${WB[@]}"

step "7/7  score all three arms through the SAME harness as the baseline"
( cd "$HARNESS" && python scripts/run_eval.py --config configs/runs/dinov3_poc_grace.yaml )

cat <<'EOF'

Done. What to read, in order:

  0. E0 -- results/dinov3_poc_drift.json. `significant: true` means the drift
     asymmetry is real here, so stage 2 has something to read and the label-free
     objective is knowingly erasing it. `false` means stage 2 will be weak on
     this data -- a finding about the dataset, not a refutation.

  1. E1 -- dinov3+identity must equal dinov3_poc_baseline exactly. If it does
     not, stop: the split is wrong and nothing below means anything.

       python scripts/compare.py \
         --baseline ../eval_pipeline/results/dinov3_poc_baseline__dinov3-sid__sid_poc_eval.json \
         --adapted  ../eval_pipeline/results/dinov3_poc_grace__dinov3+identity__sid_poc_eval.json

  2. E2 -- arm B against arm A. checkpoints/grace/dinov3_{clean,degraded}/summary.json.
     If the degraded-target control matches the clean-teacher arm, the clean
     teacher was not the mechanism.

  3. The headline -- retention recovered, against the BASELINE's clean AUC:

       python scripts/compare.py \
         --baseline ../eval_pipeline/results/dinov3_poc_baseline__dinov3-sid__sid_poc_eval.json \
         --adapted  ../eval_pipeline/results/dinov3_poc_grace__dinov3+grace__sid_poc_eval.json

  4. gate().mean() in checkpoints/grace/dinov3_clean/summary.json. It should
     climb off 0.018 and plateau around 0.1-0.5. Still at init = the alignment
     term never outweighed the identity term. Saturated at 1.0 = over-correction.

EOF
