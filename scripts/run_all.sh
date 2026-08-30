#!/usr/bin/env bash
# Every experiment in this project, in the order the argument needs them, in one
# command.
#
#   bash scripts/run_all.sh                 # the whole thing
#   bash scripts/run_all.sh --smoke         # 2 epochs, 2 cache views -- proves wiring
#   bash scripts/run_all.sh --from 8        # resume at step 8
#   bash scripts/run_all.sh --only 12       # just step 12
#   bash scripts/run_all.sh --list          # print the plan and exit
#   bash scripts/run_all.sh --skip-slow     # drop step 22 (E6, the live control)
#   WANDB=1 bash scripts/run_all.sh         # track every stage under one group
#
# Run it from the repo root (or anywhere -- it cd's to it).
#
# IDEMPOTENT. Every step checks for its own output and skips if it is already
# there: manifests are never rebuilt (that would invalidate every cache rendered
# against them), the cache resumes one view at a time, a stage-1 or stage-2 run
# is skipped if its summary.json exists, and a harness run is skipped if its
# result JSON exists. So an interrupted run is resumed by re-running the same
# command. `--force` re-does everything anyway.
#
# ONE HARD STOP. Step 5 is E1, the identity check, and it runs `compare.py
# --assert-identity`, which exits non-zero unless the null adapter reproduces the
# baseline exactly. If the trunk/head split is wrong, every number after it is
# measured against a model nobody benchmarked, so the script stops there rather
# than spending the next several hours producing them.
#
# Prerequisite, once: the DINOv3 backbone is a licence-gated Hub repo. Accept it
# at https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m and run
# `hf auth login`. Or point `backbone_id` in
# eval/configs/detectors/dinov3-wildfake.yaml at a mirror you have.
#
# THIS IS THE WHOLE-IMAGE PIPELINE. It renders cache/, trains dinov3_clean, and
# every ablation in it is that run with one key changed. The MULTI-SCALE CROP era
# is a separate chain with its own reference arm, its own caches and its own
# noise floor:
#
#   scripts/after_fetch.sh   wait for the corpus, build the manifest, audit the
#                            crop range                       <- the range GATE
#   scripts/after_audit.sh   E-shortcut, refit stage 0 on crops, the 32x32 round
#                            trip, P2' on both eval arms      <- two HARD STOPS
#   scripts/after_freq.sh    E0-freq, the crop-era caches, stage 1 and 2, the
#                            headline run, E10                <- two HARD STOPS
#
# They are deliberately not steps here. Mixing the two would invite reading a
# retention number from one protocol against a ceiling from the other;
# `CacheSpec.crop_sha` refuses that at the file level, and keeping the drivers
# apart keeps it out of the command line too.
#
# WHAT THE ORDER IS FOR. Briefly: steps 3-7 establish that there is a gap, that
# the instrument is exact, and what the ceiling on any repair is -- all before a
# single adapter is trained. Steps 8-11 produce the headline number and the
# noise floor it has to be read against. Steps 12-16 are attribution: which
# parts of the objective earn their place. Steps 17-22 are the two hard
# questions (what the method destroys, whether localization helps) and the
# slow control.

set -euo pipefail
cd "$(dirname "$0")/.."

# Windows consoles default to cp1252 and several scripts print Greek letters.
export PYTHONIOENCODING=utf-8

R=results
CKPT=checkpoints/grace
BASE=$R/dinov3_poc_baseline__dinov3-wildfake__wildfake-coco-dalle3.json

# ---------------------------------------------------------------- interpreter --
# "One command" has to mean one command, so find the project's own interpreter
# rather than trusting whatever `python` resolves to. On Windows a bare `python`
# is usually the Microsoft Store shim, which is not this project's environment.
if [[ -n "${PYTHON:-}" ]]; then :
elif [[ -x .venv/Scripts/python.exe ]]; then PYTHON=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]];        then PYTHON=.venv/bin/python
else PYTHON=python
fi
# `tr -d '\r'`: Python on Windows writes \r\n, `$(...)` strips only the trailing
# \n, and the surviving carriage return makes every echoed command containing
# this path overwrite its own first line. Harmless under Git Bash, visible the
# moment someone invokes this from PowerShell.
PY_ABS=$("$PYTHON" -c "import sys; print(sys.executable)" | tr -d '\r')

# ---------------------------------------------------------------------- args --
SMOKE=(); CACHE_EPOCHS=(); FROM=0; ONLY=""; FORCE=0; LIST=0; SKIP_SLOW=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)     SMOKE=(--epochs 2); CACHE_EPOCHS=(--epochs 2); shift ;;
    --from)      FROM="$2"; shift 2 ;;
    --only)      ONLY="$2"; shift 2 ;;
    --force)     FORCE=1; shift ;;
    --list)      LIST=1; shift ;;
    --skip-slow) SKIP_SLOW=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

# Tracking is a property of the INVOCATION, not of eighteen config files. Several
# configs ship `wandb.enabled: true`, so without an explicit --no-wandb a plain
# run would try to reach the network eighteen times. WANDB=1 turns it on for all
# of them under one group; anything else turns it off for all of them.
# summary.json next to the checkpoints is the record either way.
if [[ "${WANDB:-0}" == "1" ]]; then
  WB=(--wandb --wandb-group dinov3_poc); TRACKING=1
else
  WB=(--no-wandb); TRACKING=0
fi

# `--wandb-group X` on its own IMPLIES --wandb (naming a group is not a request
# to log nothing), so a per-experiment group may only be passed when tracking is
# actually wanted. Sets GRP, which callers splice in as "${GRP[@]}".
GRP=()
wb_group() { GRP=(); if (( TRACKING )); then GRP=(--wandb-group "$1"); fi; }

# --------------------------------------------------------------- step engine --
N=0
step() {                         # step "<title>" -> 0 if this step should run
  N=$((N + 1))
  if (( LIST )); then printf '%2d  %s\n' "$N" "$1"; return 1; fi
  if [[ -n "$ONLY" && "$ONLY" != "$N" ]]; then return 1; fi
  if (( N < FROM )); then printf '\033[2m-- %2d  %s (skipped: --from %s)\033[0m\n' "$N" "$1" "$FROM"; return 1; fi
  printf '\n\033[1m== %2d  %s\033[0m\n' "$N" "$1"
  return 0
}

have() { (( FORCE )) && return 1; [[ -e "$1" ]]; }
note() { printf '   \033[2m%s\033[0m\n' "$*"; }

# A manifest is built once and then treated as fixed: the cache stores features
# by row position and records a hash of the manifest to prove the rows still line
# up, so rebuilding one invalidates every feature cached against it. Skip, never
# overwrite.
manifest() {
  if [[ -f "$2" ]]; then echo "   $2 exists -- skipping (never rebuild a manifest a cache was rendered against)"
  else "$PY_ABS" scripts/build_manifest.py --config "$1"; fi
}

# Stage 1. `summary.json` is written last, so its presence means the run finished.
stage1() {                       # stage1 <run_id> <config> [extra args...]
  local run="$1" cfg="$2"; shift 2
  if have "$CKPT/$run/summary.json"; then note "$run already trained -- skipping"; return; fi
  "$PY_ABS" scripts/train_adapter.py "$cfg" "${SMOKE[@]}" "${WB[@]}" "$@"
}

stage2() {                       # stage2 <run_id> <config> [extra args...]
  local run="$1" cfg="$2"; shift 2
  if have "$CKPT/$run/summary.json"; then note "$run already trained -- skipping"; return; fi
  "$PY_ABS" scripts/train_discrepancy.py "$cfg" "${WB[@]}" "$@"
}

evalrun() {                      # evalrun <result-json> <run config>
  if have "$1"; then note "$(basename "$1") exists -- skipping"; return; fi
  "$PY_ABS" scripts/run_eval.py --config "$2"
}

cmp_arm() {                      # cmp_arm <adapted-json> [extra compare.py args]
  local adapted="$1"; shift
  "$PY_ABS" scripts/compare.py --baseline "$BASE" --adapted "$adapted" "$@"
}

# =============================================================================
# PHASE A -- the instrument. Nothing is trained here that GRACE touches, and
# every one of these steps can kill the project before it costs anything.
# =============================================================================

if step "P0  manifests -- the four tables this path reads"; then
  # `data/` sits at the REPO ROOT, so every script -- this one included -- reads
  # it as plain `data/...` since everything now runs from there.
  #   wildfake_train  ONE table, two splits. `train` (100,000) is the head fit
  #                   and the adapter fit; `validation` (20,000, balanced) is
  #                   stage-0 epoch selection and stage-1 held-out IMAGE
  #                   validation, read through wildfake_train_val.yaml. Disjoint
  #                   by construction, so one build covers both.
  #   wildfake        the eval set. Held out from the above by `exclude_groups`
  #                   (DALLE + coco), and the only set any reported retention
  #                   number comes from.
  #
  # Needs the images: scripts/fetch_wildfake_train.py.
  manifest load_data/configs/datasets/wildfake_train.yaml       data/wildfake_train/manifest.parquet
  manifest load_data/configs/datasets/wildfake_coco_dalle3.yaml data/wildfake_test/manifest.parquet
fi

if step "P1  stage 0 -- fit the two detector heads on CLEAN features"; then
  # The only step in the project that trains a detector, and it exists because
  # the PoC detector is assembled here: a DINOv3 trunk has no classifier, and
  # GRACE cannot splice into a seam whose head does not exist yet.
  #
  # CLEAN IMAGES ONLY, no degradation augmentation. That is the premise, not a
  # corner cut -- a head trained under augmentation would have already solved
  # part of the problem GRACE exists to solve, and every retention number after
  # it would be measuring the augmentation.
  #
  # Two heads, differing in `input_mode` alone. They are D1's two arms and they
  # are deliberately not interchangeable: the trunk sees the image at a different
  # scale in each, so `_assert_head_matches` refuses to load one into the other.
  if have checkpoints/probe/dinov3_wildfake/head.pt; then note "resize head exists -- skipping"
  else "$PY_ABS" scripts/train_probe.py train/configs/probe/dinov3_wildfake.yaml "${SMOKE[@]}" "${WB[@]}"; fi
  if have checkpoints/probe/dinov3_wildfake_crop/head.pt; then note "crop head exists -- skipping"
  else "$PY_ABS" scripts/train_probe.py train/configs/probe/dinov3_wildfake_crop.yaml "${SMOKE[@]}" "${WB[@]}"; fi
fi

if step "P2  the baseline -- the retention curve everything is measured against"; then
  # THE PREMISE CHECK. If retention does not collapse here, there is no gap for
  # GRACE to close and the project has answered its own question at ~8% of what
  # the stage-1 cache costs. 13,841 images x 26 conditions.
  evalrun "$BASE" eval/configs/runs/dinov3_poc_baseline.yaml
fi

if step "D1  the preprocessing confound -- the crop-fed baseline"; then
  # Is the head reading generation traces, or is it reading CONTENT? A head that
  # separates "looks generated" from "looks photographed" is making a SEMANTIC
  # distinction, and no transform in the grid destroys semantics -- a blurred
  # dragon is still a dragon. Such a head never collapses, and there is no room
  # for a repair to show up. It has happened once already, on this project's
  # previous dataset, and the cause was upstream of the head: resize to 224 with
  # default_to_square shrinks a 1024px source ~4.6x BEFORE the trunk runs.
  #
  #   crop collapses, resize does not -> the resize head took the shortcut.
  #                                      Repoint the cache and train configs at
  #                                      dinov3-wildfake-crop.yaml and re-render.
  #   both collapse                   -> preprocessing was not the confound.
  #   neither collapses               -> the DATASET separates on content. That
  #                                      is the finding, and no adapter fixes it.
  evalrun "$R/dinov3_poc_baseline_crop__dinov3-wildfake-crop__wildfake-coco-dalle3.json" \
          eval/configs/runs/dinov3_poc_baseline_crop.yaml
  "$PY_ABS" scripts/report.py --results results/ || true
fi

if step "E1  THE GATE -- the null adapter must reproduce the baseline exactly"; then
  # If it does not, the trunk/head split is wrong and nothing downstream is a
  # measurement. --assert-identity exits non-zero on any nonzero delta, and
  # `set -e` turns that into a full stop. The identity is exact rather than
  # approximate: every adapter block's final projection is zero-initialised, so
  # an untrained adapter returns its input bit for bit.
  evalrun "$R/dinov3_poc_identity__dinov3+identity__wildfake-coco-dalle3.json" \
          eval/configs/runs/dinov3_poc_identity.yaml
  cmp_arm "$R/dinov3_poc_identity__dinov3+identity__wildfake-coco-dalle3.json" --assert-identity
fi

if step "P3  render the feature caches -- train + both val sets"; then
  # The trunk is frozen and a clean image never changes, so its features are
  # always the same: compute them once and the "teacher" becomes a lookup.
  #
  # The same reasoning extends to the DEGRADED side, which is the part that is
  # not obvious. Every recipe is drawn from stable_seed(index, level, replicate,
  # seed) -- a hash, not a running RNG counter -- so a degraded view is a pure
  # function of (image, condition) and a training epoch is just the `replicate`
  # field under another name. Epoch 7's degradation of image 412 can be computed
  # now, without having run epochs 0-6. That is what makes pre-rendering possible
  # at all, and it is what E6 (step 22) exists to check is not being exploited.
  #
  # The two val caches are NOT optional: every train/configs/train/dinov3_*.yaml
  # names them in `val_cache_dirs` and train_adapter.py opens them in its first
  # few statements, so a missing one fails at second zero rather than after the
  # run. Separate out_dir each, because build_cache.py names its root after the
  # DETECTOR alone and two datasets would otherwise collide.
  "$PY_ABS" scripts/build_cache.py train/configs/cache/dinov3.yaml --dry-run
  "$PY_ABS" scripts/build_cache.py train/configs/cache/dinov3.yaml          "${CACHE_EPOCHS[@]}"
  "$PY_ABS" scripts/build_cache.py train/configs/cache/dinov3_val.yaml      "${CACHE_EPOCHS[@]}"
fi

if step "E0  drift geometry -- the CEILING on everything after this"; then
  # Two numbers come out of this, and the second one matters more.
  #
  # `significant` is RA-Det's premise on this data: do generated images drift
  # further under perturbation than real ones? If yes, the discrepancy branch has
  # something to read -- and stage 1's objective is erasing it every time its
  # loss improves, which is what E4 (step 17) measures.
  #
  # `parallel_fraction` is the ceiling. The frozen head collapses a whole feature
  # vector into one number, so it can only notice movement along the one
  # direction it is sensitive to. Drift orthogonal to that is invisible to it,
  # and no correction of it can move AUC. Read this BEFORE the retention numbers:
  # it tells you what fraction of the damage was ever repairable, and therefore
  # how to read whatever gain comes back.
  #
  # It needs the render (it compares the clean view against a degraded one), but
  # it comes before anything is TRAINED, which is the sense in which it is first.
  # The output carries the dataset in its name, and so does the guard. It used
  # to be a bare `dinov3_poc_drift.json`, which meant the round-1 NTIRE artifact
  # sitting in results/ made this step skip silently -- and `parallel_fraction`
  # is the denominator of the headline claim, so the run would have carried an
  # NTIRE ceiling into a WildFake pipeline without saying a word. The old file
  # is kept as `dinov3_poc_drift__ntire-train.json`, a prior rather than a record.
  if have results/dinov3_poc_drift__wildfake-train.json
  then note "drift analysis exists -- skipping"
  else
    "$PY_ABS" scripts/analyze_drift.py \
      --cache    cache/dinov3-wildfake \
      --dataset  load_data/configs/datasets/wildfake_train.yaml \
      --detector eval/configs/detectors/dinov3-wildfake.yaml \
      --split    eval.splits.dinov3.DINOv3Split \
      --out      results/dinov3_poc_drift__wildfake-train.json
  fi
fi

# =============================================================================
# PHASE B -- the result, and the yardstick it has to be read against.
# =============================================================================

if step "S0  the reference arm, five seeds -- and the NOISE FLOOR"; then
  # Every ablation after this is dinov3_clean.yaml with one key changed, so this
  # is both the headline label-free adapter and the control for all of them.
  #
  # The five seeds are not a robustness gesture -- they are the unit of
  # measurement. On this data the hard-split delta AUC moves by ~0.0007 on the
  # seed alone, so an ablation that shifts a metric by less than about 0.0015 has
  # shifted nothing, and without this number none of steps 12-16 can be read at
  # all. Seed replicates are CLI overrides rather than five near-identical config
  # files, which is what --seed/--run-id exist for.
  stage1 dinov3_clean train/configs/train/dinov3_clean.yaml
  for s in 1 2 3 4; do
    stage1 "dinov3_clean_s$s" train/configs/train/dinov3_clean.yaml --seed "$s" --run-id "dinov3_clean_s$s"
  done
  "$PY_ABS" scripts/seed_stats.py dinov3_clean 'dinov3_clean_s*'
fi

if step "E5a stage 2 -- the supervised discrepancy head, adapter frozen"; then
  # Stage 2 never touches the adapter, so GRACE and GRACE-D ship the same weights
  # bit for bit and "the adapter is trained without labels" stays literally true.
  # That separation is also what makes step 19 (E4) possible.
  stage2 dinov3_disc train/configs/train/dinov3_discrepancy.yaml
fi

if step "E5b the beta sweep -- bound the fusion over ALL weightings"; then
  # A single learned beta cannot tell you whether the fusion is weak or merely
  # mis-weighted. Sweeping beta answers it: if the best achievable retention over
  # every beta is still under 1.0, no weighting rescues the branch and the aux
  # logit is redundant with the main head rather than uninformative. Written to
  # JSON, because a bound that exists only in a comment is not a result.
  if have results/dinov3_beta_sweep.json; then note "beta sweep exists -- skipping"
  else
    "$PY_ABS" scripts/sweep_beta.py train/configs/train/dinov3_discrepancy.yaml \
      --run dinov3_disc --out results/dinov3_beta_sweep.json
  fi
fi

if step "E5c THE HEADLINE -- score +grace and +grace-d through the harness"; then
  # Same conditions, same threshold rule, same JSON schema as the baseline,
  # because it is the same harness. The arms differ by one config file each.
  evalrun "$R/dinov3_poc_grace__dinov3+grace__wildfake-coco-dalle3.json" \
          eval/configs/runs/dinov3_poc_grace.yaml
  echo; echo "--- GRACE (label-free restoration). retention > 1.0 is IMPOSSIBLE here ---"
  cmp_arm "$R/dinov3_poc_grace__dinov3+grace__wildfake-coco-dalle3.json" \
          --out results/compare_grace.json
  echo; echo "--- GRACE-D (fused). the only arm allowed to exceed retention 1.0 ---"
  cmp_arm "$R/dinov3_poc_grace__dinov3+grace-d__wildfake-coco-dalle3.json" \
          --out results/compare_grace_d.json
fi

# =============================================================================
# PHASE C -- attribution. Which parts of the objective earn their place?
# Every arm below is dinov3_clean.yaml with exactly ONE key changed.
# =============================================================================

if step "E2  the clean teacher -- is it the mechanism, or self-distillation?"; then
  # target_view: degraded asks the adapter to reproduce its own INPUT. That adds
  # no information, so it should achieve nothing. If it matches the clean-teacher
  # arm, the clean teacher was not the mechanism and something else explains the
  # result. grace_adapter/tests/test_configs.py pins the pair at one key apart.
  stage1 dinov3_degraded train/configs/train/dinov3_degraded.yaml
fi

if step "E3  the loss ablation -- does the Jacobian weighting earn its place?"; then
  # Plain MSE treats every feature dimension as equally worth fixing; the head
  # does not agree, since only error along the direction it is sensitive to can
  # change AUC. `weighting: none` makes L_err PROVABLY plain F.mse_loss.
  #
  # Read hard-split delta AUC against step 8's seed floor. Do NOT read
  # `cos_decision`: it came back anti-correlated with the outcome last round --
  # the arm with the lowest alignment had the highest gain -- so it is not a
  # usable proxy for whether the weighting is working.
  stage1 dinov3_plain_mse train/configs/train/dinov3_plain_mse.yaml
fi

if step "E8  the gate -- is it learning, or is AdamW decaying it open?"; then
  # The gate is reported everywhere as evidence the adapter is "learning to apply
  # its correction". Decoupled weight decay pulls a LOGIT toward 0, i.e. the
  # sigmoid toward 0.5, so the gate opens whether or not the objective asks it
  # to. `decay_gate: false` exempts gate_logit and touches nothing else -- a
  # global weight_decay: 0.0 would also change the MLP and confound it.
  #
  # Control: the reference arm from step 8 (decay on, gate_init -4 by default).
  stage1 dinov3_gate_nodecay train/configs/train/dinov3_gate_nodecay.yaml
  "$PY_ABS" scripts/seed_stats.py dinov3_clean --vs dinov3_gate_nodecay
fi

if step "E9a hyperparameters -- the loss-weight ratio (the only live axis)"; then
  # The reference is ratio 16, so there is no wratio_16 config. Note the ratio's
  # preference reversed between splits last round -- 16 on ntire_val_hard, 4 on
  # ntire_val -- and both directions were backed by five-seed families. There is
  # only ONE validation split now, so that comparison cannot be repeated; the
  # split-dependence is a round-1 observation, not something round 2 can re-test.
  # That is a
  # measured split-dependence to report, not noise to resolve by picking a split.
  stage1 dinov3_sweep_wratio_0.25 train/configs/train/dinov3_sweep_wratio_0.25.yaml
  stage1 dinov3_sweep_wratio_1    train/configs/train/dinov3_sweep_wratio_1.yaml
  stage1 dinov3_sweep_wratio_4    train/configs/train/dinov3_sweep_wratio_4.yaml
fi

if step "E9b hyperparameters -- capacity and gate init"; then
  # A saturated capacity axis is a result FOR the method: GRACE's claim is that
  # the evidence is displaced rather than destroyed, and a displacement should be
  # undoable by a SMALL correction. If doubling the adapter buys nothing, the
  # parameter budget is not the binding constraint.
  stage1 dinov3_sweep_nblocks_2      train/configs/train/dinov3_sweep_nblocks_2.yaml
  stage1 dinov3_sweep_bottleneck_256 train/configs/train/dinov3_sweep_bottleneck_256.yaml
  stage1 'dinov3_sweep_gate_-3'      'train/configs/train/dinov3_sweep_gate_-3.yaml'
  "$PY_ABS" scripts/seed_stats.py dinov3_clean 'dinov3_clean_s*' \
    --vs dinov3_sweep_wratio_0.25 dinov3_sweep_wratio_1 dinov3_sweep_wratio_4 \
         dinov3_sweep_nblocks_2 dinov3_sweep_bottleneck_256 'dinov3_sweep_gate_-3'
fi

# =============================================================================
# PHASE D -- the sharpest critique, and the architecture question.
# =============================================================================

if step "E4  the erasure trade-off -- does restoring destroy the evidence?"; then
  # THE sharpest question in the project. E0 says fakes drift further under
  # perturbation than reals. Stage 1 trains the adapter to undo drift. So an
  # adapter that gets better at its job is removing more drift from the fakes
  # than from the reals -- i.e. removing the thing that distinguished them --
  # while its reconstruction loss falls the whole time.
  #
  # Testable only because stage 2 never touches the adapter: train stage 2
  # against each stage-1 checkpoint in turn and the adapter is the only thing
  # that varies. `checkpoint_every: 2` over 12 epochs leaves six of them, and
  # stage 2 takes seconds.
  #
  # Read validation.epoch_*.auc_aux against stage-1 progress -- and plot
  # |AUC - 0.5|, not AUC: each stage-2 run is independent and the aux head's
  # POLARITY is not stable across them. Falling = restoration erases evidence.
  # Rising = it concentrates it, and the obvious critique does not land.
  shopt -s nullglob
  cks=("$CKPT"/dinov3_clean/step_*.pt)
  shopt -u nullglob
  if (( ${#cks[@]} == 0 )); then
    echo "   no step_*.pt under $CKPT/dinov3_clean -- run step 8 first" >&2; exit 1
  fi
  wb_group e4_erasure
  for ck in "${cks[@]}"; do
    rid="e4_$(basename "$ck" .pt)"
    stage2 "$rid" train/configs/train/dinov3_discrepancy.yaml \
      --adapter "$ck" --run-id "$rid" "${GRP[@]}"
  done
fi

if step "E7a the ladder -- render the tap caches (a separate ~21 GB)"; then
  # The plain adapter has to infer, from the pooled seam alone, WHICH trunk stage
  # the damage entered at, and it is nearly blind to that (0.376 vs 0.896 nine-way
  # transform ID). If displaced evidence is recoverable at all, does knowing where
  # it was displaced from recover more of it?
  "$PY_ABS" scripts/build_cache.py train/configs/cache/dinov3_taps.yaml --dry-run
  "$PY_ABS" scripts/build_cache.py train/configs/cache/dinov3_taps.yaml          "${CACHE_EPOCHS[@]}"
  "$PY_ABS" scripts/build_cache.py train/configs/cache/dinov3_val_taps.yaml      "${CACHE_EPOCHS[@]}"
fi

if step "E7b the ladder -- the 12-epoch arm, and the tap_dim sweep"; then
  # dinov3_ladder_final is the ONLY legitimate ladder arm: same epochs, same loss
  # weights, same seed as the reference. A 4-epoch ladder read against a 12-epoch
  # plain arm is measuring epochs, not taps -- that is the mistake this step
  # exists not to repeat.
  #
  # Read `tap_gate/block*` in history: how much the correction leans on each
  # block. That per-layer profile is the figure this arm exists to make. Read it
  # knowing E8's result -- weight decay moves a gate on its own, so a tap_gate
  # sitting near its 0.018 init is not evidence of anything either way.
  #
  # tap_dim 64 IS dinov3_ladder_final, so the sweep has three other rungs. The
  # cache is unaffected by tap_dim: it stores raw (K, 768) taps and every rung
  # reads the same bytes.
  stage1 dinov3_ladder_final    train/configs/train/dinov3_ladder_final.yaml
  stage1 dinov3_sweep_tapdim_32  train/configs/train/dinov3_sweep_tapdim_32.yaml
  stage1 dinov3_sweep_tapdim_128 train/configs/train/dinov3_sweep_tapdim_128.yaml
  stage1 dinov3_sweep_tapdim_256 train/configs/train/dinov3_sweep_tapdim_256.yaml
  "$PY_ABS" scripts/seed_stats.py dinov3_clean 'dinov3_clean_s*' \
    --vs dinov3_ladder_final dinov3_sweep_tapdim_32 dinov3_sweep_tapdim_128 dinov3_sweep_tapdim_256
fi

if step "E7c the ladder -- stage 2 with and without per-tap drift"; then
  # On a `vector` seam the auxiliary head sees ONE drift norm however deep the
  # damage entered. `use_taps: true` gives it one log1p'd norm per tapped block --
  # the per-block damage profile, which is the closest this PoC gets to the input
  # a `layers` seam would hand it for free. No such detector is in the tree any
  # more, so the ladder is now the ONLY way to get that signal here. It is the
  # difference between the branch's weakest possible form and something worth a
  # null result.
  stage2 disc_ladder      train/configs/train/dinov3_discrepancy_ladder.yaml
  stage2 disc_ladder_taps train/configs/train/dinov3_discrepancy_ladder_taps.yaml
fi

if step "E7d the ladder -- score it on the same denominator as everything else"; then
  evalrun "$R/dinov3_poc_ladder__dinov3+grace-ladder__wildfake-coco-dalle3.json" \
          eval/configs/runs/dinov3_poc_ladder.yaml
  cmp_arm "$R/dinov3_poc_ladder__dinov3+grace-ladder__wildfake-coco-dalle3.json" \
          --out results/compare_grace_ladder.json
fi

# =============================================================================
# PHASE E -- the slow control. Last, because it is the cheapest thing to defer
# and the only one that costs hours.
# =============================================================================

if step "E6  cached vs live -- is the finite augmentation set being exploited?"; then
  if (( SKIP_SLOW )); then note "--skip-slow: not running the live control"
  else
    # Pre-rendering fixes the augmentation at 12 draws per image. The fair
    # objection is that the adapter memorises those 12 corruptions rather than
    # learning to undo the FAMILY they were drawn from. `source: live` settles
    # it: same schedule, same grid, same level weights, but a fresh recipe every
    # step and the trunk running inside the loop.
    #
    # This is the one arm that is not fast -- it pays a full DINOv3 forward per
    # image per step where every other arm pays two memmap reads. batch_size
    # drops 256 -> 32 because it now counts IMAGES, not cached feature vectors.
    #
    #   cached ~= live -> the finite epoch set is not being exploited, and every
    #                     cached number above stands.
    #   a gap          -> the cached runs are partly memorising specific
    #                     corruptions, and n_epochs in
    #                     train/configs/cache/dinov3.yaml has to go up, which
    #                     means a re-render.
    stage1 dinov3_live train/configs/train/dinov3_live.yaml
    "$PY_ABS" scripts/seed_stats.py dinov3_clean 'dinov3_clean_s*' --vs dinov3_live
  fi
fi

TOTAL=$N
if (( LIST )); then echo; echo "$TOTAL steps. bash scripts/run_all.sh --from N to resume."; exit 0; fi

cat <<'EOF'

===============================================================================
Done. What to read, and in what order -- the argument, not the file list.
===============================================================================

1. IS THERE A PROBLEM AT ALL?
   results/dinov3_poc_baseline__*.json -> levels.*.retention

   If retention does not collapse, GRACE has nothing to repair and the finding
   is about the dataset. Read the crop baseline next to it (D1): if neither
   collapses, the detector is separating on CONTENT, and no adapter fixes that.

2. WHAT WAS THE MOST ANY REPAIR COULD HAVE DONE?
   results/dinov3_poc_drift__wildfake-train.json -> overall.parallel_fraction

   The fraction of the drift that lies along the direction the frozen head can
   see. Everything orthogonal to it is unrepairable BY CONSTRUCTION, because the
   head cannot notice it moving. This is the ceiling, and it is computed before
   any adapter exists. Quote the retention gain against it, not against 1.0.

   `significant` / `asymmetry_ci` is the separate question of whether fakes drift
   further than reals -- i.e. whether the discrepancy branch has anything to read.

3. WHAT DID THE LABEL-FREE ADAPTER RECOVER?
   results/compare_grace.json, and the printed table above.

   Baseline-normalized retention: divided by the BASELINE's clean AUC, not the
   adapted detector's. retention > 1.0 is impossible for GRACE by construction --
   a restorer cannot beat the clean-image score. If you see it, you are reading a
   GRACE-D file.

4. IS THAT NUMBER BIGGER THAN NOTHING?
   scripts/seed_stats.py output from step 8.

   Five seeds of one config. Any two arms closer together than about twice this
   spread are the same arm. Every verdict in steps 13-18 is this comparison.

5. WHY DOES IT WORK?
   checkpoints/grace/dinov3_{clean,degraded}/summary.json  (E2)

   The degraded-target control adds no information and should gain nothing. If it
   matches the clean-teacher arm, the teacher was not the mechanism.

   Then dinov3_plain_mse (E3) for the weighting, and dinov3_gate_nodecay (E8) for
   the gate. Read E8 first if you are about to report a gate number as evidence
   of learning -- decoupled weight decay moves it on its own.

6. WHAT DOES IT COST?
   checkpoints/grace/e4_step_*/summary.json -> validation.*.auc_aux  (E4)

   Plotted against stage-1 progress, as |AUC - 0.5|. Falling = restoration is
   erasing the forensic evidence it was trained to remove. Rising = it is
   concentrating it, and the sharpest critique of the approach does not land.

7. CAN THE CEILING BE BROKEN?
   results/compare_grace_d.json -> exceeds_clean_ceiling
   results/dinov3_beta_sweep.json

   Only the fused score can exceed retention 1.0, because how much damage an
   image took is information the clean image never contained. If it does not, the
   beta sweep says whether that is weak signal or bad weighting -- a ceiling
   under 1.0 across ALL beta means the aux logit is REDUNDANT with the main head,
   which is a different and more interesting claim than "uninformative".

   Report this as the weakest possible form of the branch: on a `vector` split
   the aux head sees ONE drift norm. A `layers` detector would give it 24.

8. WHAT IS STILL OPEN?
   dinov3_live (E6) -- if cached and live disagree, every cached number above is
   partly measuring memorised corruptions.
   dinov3_ladder_final + tap_gate/block* (E7) -- whether knowing where the damage
   entered recovers more of it.

W&B is never the record. summary.json next to the checkpoints is, and it is
written whether or not anything was tracked.
EOF
