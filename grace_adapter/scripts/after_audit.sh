#!/usr/bin/env bash
# The second half: from an audited crop range to the project's first measurement.
#
#   bash scripts/after_audit.sh
#
# Run after scripts/after_fetch.sh, once `crop.s_max` is written into
# configs/probe/dinov3_wildfake_multiscale.yaml. Refuses to start otherwise --
# CropConfig raises on `enabled: true` with no `s_max`, which is the point.
#
#   1  E-shortcut   what the FILE gives away, on the benchmark and on both arms
#   2  P1'          refit stage 0 on multi-scale crops
#   3  -1d          the 32x32 round trip           <- HARD STOP
#   4  P2'          baseline retention, both arms  <- THE FIRST MEASUREMENT
#
# TWO HARD STOPS, and they are the reason this is a script rather than a list.
#
# Step 1 must come out at chance on the two arms. Both arms give every image
# identical dimensions by construction, so if the shortcut still scores, the
# arms are not doing what they were built to do and nothing after this is
# interpretable. It is expected to score ~1.0 on the RAW benchmark -- that is the
# finding the arms exist to answer, not a failure.
#
# Step 3 must FAIL in the old sense: the head must fall toward chance when its
# input is destroyed by a 32x32 round trip. A head that survives is reading
# content, its retention curve will look excellent for the wrong reason, and the
# frequency branch has nothing to enrich. Under whole-image `resize` a probe held
# 0.985 through this. If the multi-scale head does too, stop and report that.
#
# IDEMPOTENT: every step skips if its output JSON is already there.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

if [[ -n "${PYTHON:-}" ]]; then :
elif [[ -x ../.venv/Scripts/python.exe ]]; then PYTHON=../.venv/Scripts/python.exe
elif [[ -x ../.venv/bin/python ]];        then PYTHON=../.venv/bin/python
else PYTHON=python
fi
# An absolute POSIX path, built by the shell rather than by Python.
# `sys.executable` returns backslashes, and a bash launched from PowerShell
# cannot exec that -- 'command not found', for a file that plainly exists. It
# also arrives with a trailing CR, which mangles any message quoting it.
# Absolute because several steps below run from inside ../eval_pipeline.
PY_RESOLVED=$(command -v "$PYTHON" 2>/dev/null) || PY_RESOLVED="$PYTHON"
PY_ABS="$(cd "$(dirname "$PY_RESOLVED")" && pwd)/$(basename "$PY_RESOLVED")"
if ! "$PY_ABS" -c "import sys" >/dev/null 2>&1; then
  printf "\ncannot execute %s\n" "$PY_ABS" >&2
  printf "Set PYTHON to an interpreter this shell can run, e.g.\n" >&2
  printf "    PYTHON=python bash scripts/after_audit.sh\n" >&2
  exit 1
fi

FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   \033[2m%s\033[0m\n' "$*"; }
die()  { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
have() { (( FORCE )) && return 1; [[ -e "$1" ]]; }

PROBE=configs/probe/dinov3_wildfake_multiscale.yaml
HEAD=checkpoints/probe/dinov3_wildfake_multiscale/head.pt
EV=../eval_pipeline

# ------------------------------------------------- 0  the gate is actually set --
step "0  checking the crop range is written"
"$PY_ABS" - "$PROBE" <<'PY'
import sys
sys.path[:0] = [".", "../eval_pipeline"]
from grace.config import load_probe_config
cfg = load_probe_config(sys.argv[1])          # raises if enabled with no s_max
if not cfg.crop.enabled:
    raise SystemExit("crop.enabled is false -- this whole protocol is the crop")
print(f"   crop {cfg.crop.s_min}-{cfg.crop.s_max}px {cfg.crop.policy}, seed {cfg.crop.seed}")
PY
rc=$?
# Distinguish the config being wrong from python failing to run at all.
# These shared one message, so a "command not found" was reported as a
# broken crop block -- sending you to edit a file that was already correct.
if (( rc != 0 )); then
  die "the crop range did not load from $PROBE (python exited $rc; its own
message is above). If that message is about s_max, run
    bash scripts/after_fetch.sh --no-wait --write-range
to write the audited range into all seven crop configs."
fi

# ------------------------------------------------------------ 1  E-shortcut ----
step "1  E-shortcut  what the file gives away, before any model"
for ds in wildfake_coco_dalle3 wildfake_train_val; do
  out="results/shortcut__${ds}.json"
  if have "$out"; then note "$out exists -- skipping"; continue; fi
  ( cd "$EV" && "$PY_ABS" scripts/shortcut_baseline.py \
      --dataset "configs/datasets/${ds}.yaml" \
      --out "../grace_adapter/$out" ) || die "shortcut baseline failed on $ds"
done
note "the benchmark is EXPECTED to score ~1.0 combined -- every real in it is"
note "exactly 200x200. Read the two BLOCKS, not the combined number: the"
note "dimension rows die under both arms by construction, and the"
note "bits-per-pixel row does not. That surviving separation is the floor"
note "every frequency result has to clear."

# --------------------------------------------------------------- 2  P1' --------
step "2  P1'  refit stage 0 on multi-scale crops"
if have "$HEAD"; then
  note "$HEAD exists -- skipping"
else
  "$PY_ABS" scripts/train_probe.py "$PROBE" --no-wandb || die "stage 0 failed"
fi

# --------------------------------------------------------------- 3  -1d --------
step "3  -1d  the 32x32 round trip  (HARD STOP)"
RT=results/roundtrip__dinov3_wildfake_multiscale.json
if have "$RT"; then
  note "$RT exists -- skipping"
else
  ( cd "$EV" && "$PY_ABS" scripts/roundtrip_check.py \
      --detector configs/detectors/dinov3-wildfake-multiscale.yaml \
      --dataset  configs/datasets/wildfake_train_val.yaml \
      --out "../grace_adapter/$RT" )
  if (( $? != 0 )); then
    die "THE HEAD SURVIVED THE ROUND TRIP.
It is reading content, not generation traces: nothing forensic survives 32x32, so
whatever it scores on is still there afterwards. Its retention curve would look
excellent for a reason that has nothing to do with robustness, and there would be
no damage for GRACE to repair or for the frequency branch to enrich.

That is a finding, not a bug to route around. Read $RT,
then decide whether the crop range is too coarse (a 448px window of a 512px image
is nearly the whole frame) or whether this corpus separates on content."
  fi
fi

# --------------------------------------------------------------- 4  P2' --------
step "4  P2'  baseline retention on both arms  (THE FIRST MEASUREMENT)"
BASE="$EV/results/dinov3_poc_baseline_arms__dinov3-wildfake-crop200__wildfake-coco-dalle3.json"
if have "$BASE"; then
  note "already scored -- skipping"
else
  ( cd "$EV" && "$PY_ABS" scripts/run_eval.py \
      --config configs/runs/dinov3_poc_baseline_arms.yaml ) || die "P2' failed"
fi

cat <<EOF

$(printf '\033[1m== done. What you now have, and how to read it\033[0m')

  results/shortcut__*.json         the floor. Must be ~0.5 on the two arms.
  $RT
                                   must show a COLLAPSE, not a survival.
  $EV/results/dinov3_poc_baseline_arms__*.json
                                   retention, arm (a) and arm (b).

Read arm (a) -- crop200 -- as the real number. Arm (b) is out of distribution for
a crop-trained head and spectrally confounded besides, so it is a robustness
check, not a comparison.

THE QUESTION THIS ANSWERS: does retention collapse at all? If it does not, GRACE
has nothing to repair and the frequency branch has nothing to enrich, and that is
the result -- established before a single adapter is trained, which is the whole
reason these run first.

Then, once you have read those numbers:

    bash scripts/after_freq.sh --list      # the plan, 9 steps
    bash scripts/after_freq.sh

which runs the E0-freq falsification gate, renders the crop-era spatial and
frequency caches, trains stage 1 and the enricher, scores every arm on both
evaluation arms, and checks the E10 identity. It stops on its own at the first
gate that fails.
EOF
