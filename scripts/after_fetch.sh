#!/usr/bin/env bash
# Wait for the WildFake fetch, then take the corpus as far as the first gate.
#
#   bash scripts/after_fetch.sh                # wait, then run
#   bash scripts/after_fetch.sh --no-wait      # fetch is already done
#   bash scripts/after_fetch.sh --poll 300     # check every 5 min (default 120s)
#
# Run it from the repo root (or anywhere -- it cd's to it). Leave it in its own
# terminal; the fetch is ~17 hours and this sits on top of it.
#
# WHAT IT DOES
#
#   0  wait for fetch_wildfake_train.py to exit
#   1  verify the corpus is actually complete
#   2  P0   build the manifest          (once, then never again)
#   3  -1b  audit native sizes          -> THE CROP RANGE GATE
#
# and then STOPS, because the next thing needs a number this cannot choose for
# you. See the gate text it prints.
#
# WHY IT STOPS THERE. `crop.s_max` is the largest window every image in the
# corpus can supply from its own pixels. Draw beyond it and one class clamps more
# than the other, the realized crop size becomes a classifier, and the
# augmentation hands back the very shortcut it was introduced to remove -- on
# wildfake_test an unaudited 128-512 range scores E-cropsize 0.9895 by itself.
# The audit computes the safe bound; writing it into the config is one line, and
# it is deliberately yours to write, because it is the number the whole
# multi-scale protocol rests on.
#
# IDEMPOTENT, like run_all.sh: the manifest is never rebuilt (that would
# invalidate every cache rendered against it) and the audit is skipped if its
# JSON is already there. Rerun freely.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

if [[ -n "${PYTHON:-}" ]]; then :
elif [[ -x .venv/Scripts/python.exe ]]; then PYTHON=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]];        then PYTHON=.venv/bin/python
else PYTHON=python
fi
# An absolute POSIX path, built by the shell rather than by Python.
#
# `sys.executable` returns a Windows path with backslashes, and a bash
# launched from PowerShell cannot exec that -- 'command not found', for a file
# that plainly exists. It also arrives with a trailing carriage return, since
# Python writes CRLF and $(...) strips only the LF; that one mangles any
# message quoting the path, because the CR returns the cursor to column 0.
#
# `cd $(dirname) && pwd` sidesteps both: the shell reports the directory in
# whatever form the shell itself can use.
PY_RESOLVED=$(command -v "$PYTHON" 2>/dev/null) || PY_RESOLVED="$PYTHON"
PY_ABS="$(cd "$(dirname "$PY_RESOLVED")" && pwd)/$(basename "$PY_RESOLVED")"
if ! "$PY_ABS" -c "import sys" >/dev/null 2>&1; then
  printf "\ncannot execute %s\n" "$PY_ABS" >&2
  printf "Set PYTHON to an interpreter this shell can run, e.g.\n" >&2
  printf "    PYTHON=python bash scripts/after_fetch.sh\n" >&2
  exit 1
fi

WAIT=1; POLL=120; WRITE=0
MAX_UNREADABLE=${MAX_UNREADABLE:-0}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-wait)     WAIT=0; shift ;;
    --poll)        POLL="$2"; shift 2 ;;
    --write-range) WRITE=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   \033[2m%s\033[0m\n' "$*"; }
die()  { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

IMAGES=data/wildfake_train/images
MANIFEST=data/wildfake_train/manifest.parquet
AUDIT=results/audit_sizes__wildfake-train.json
EXPECT=60000

# ------------------------------------------------------- 0  wait for the fetch --
# `ps` and `tasklist` on Windows show the executable but not its arguments, and
# several unrelated python.exe live in this tree (the editor's formatter, for
# one), so the only reliable test is a command-line match through PowerShell.
# `@(...)` forces an array, so a single match still answers 1 rather than $null.
fetch_count() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | \
      Where-Object { \$_.CommandLine -like '*fetch_wildfake*' }).Count" 2>/dev/null \
    | tr -d '\r\n'
}

fetch_running() {
  local n; n=$(fetch_count)
  [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 ))
}

if (( WAIT )); then
  step "0  waiting for fetch_wildfake_train.py"
  # Prove the probe works before trusting a negative from it. Without this, a
  # PowerShell that is missing, blocked or slow reads as "the fetch finished"
  # and the script charges into a half-downloaded corpus.
  probe=$(fetch_count)
  if [[ ! "$probe" =~ ^[0-9]+$ ]]; then
    die "cannot tell whether the fetch is running: the PowerShell probe returned
${probe:-<nothing>} instead of a count. Refusing to guess -- a wrong 'it
finished' here starts stage 0 on a partial corpus. Either fix the probe, or
rerun with --no-wait once you can see for yourself that the fetch is done."
  fi
  if ! fetch_running; then
    note "not running -- assuming it already finished"
  else
    while fetch_running; do
      printf '\r   \033[2m%s  images on disk: %s / %s\033[0m\033[K' \
        "$(date +%H:%M:%S)" "$(find "$IMAGES" -type f 2>/dev/null | wc -l)" "$EXPECT"
      sleep "$POLL"
    done
    printf '\n'
    note "fetch exited"
  fi
fi

# --------------------------------------------------- 1  verify the corpus --------
step "1  verifying the corpus"
HAVE=$(find "$IMAGES" -type f 2>/dev/null | wc -l)
note "$HAVE images under $IMAGES"
if (( HAVE < EXPECT )); then
  die "expected $EXPECT, found $HAVE. The fetch did not finish cleanly.
Rerun it before this script:
    $PY_ABS scripts/fetch_wildfake_train.py --resume
A short count is never something to work around -- the sample is proportional,
so a missing archive silently re-weights every stratum in the manifest."
fi

# ------------------------------------------------------- 2  P0  the manifest ----
step "2  P0  build the manifest"
if [[ -e "$MANIFEST" ]]; then
  note "$MANIFEST exists -- NOT rebuilding (its hash is what every cache checks)"
else
  "$PY_ABS" scripts/build_manifest.py \
      --config load_data/configs/datasets/wildfake_train.yaml || die "manifest build failed"
fi

# --------------------------------------------- 3  -1b  the crop range gate ------
step "3  -1b  audit native sizes -- THE CROP RANGE GATE"
if [[ -e "$AUDIT" ]]; then
  note "$AUDIT exists -- skipping the audit"
else
  mkdir -p results
  "$PY_ABS" scripts/audit_sizes.py \
      --config load_data/configs/datasets/wildfake_train.yaml \
      --max-unreadable "$MAX_UNREADABLE" \
      --out "$AUDIT"
  rc=$?
  # Branch on WHICH failure. These used to share one message, so a corrupt
  # PNG and a missing interpreter were both reported as "the crop range
  # leaks" -- a specific, confident, wrong diagnosis, which is worse than
  # none at all. Exit 1 is the verdict; exit 2 is a corpus problem.
  case $rc in
    0) ;;
    1) die "NO SAFE CROP RANGE. Every candidate leaks the label through crop
size, which means this corpus cannot supply a class-independent window
distribution at s_min=128. Lower s_min and re-audit, or reconsider the corpus.
Do not pick a range by hand to get past this -- the leak is real and it would
end up in the headline number." ;;
    2) die "UNREADABLE IMAGES -- a corpus problem, not a crop-range one. The
audit named them above. Re-extract them, or rerun as
    MAX_UNREADABLE=<n> bash scripts/after_fetch.sh --no-wait
to audit the rest and handle them separately. Do not just leave them: what
cannot be read here also cannot be rendered into the cache later." ;;
    *) die "the audit exited $rc before reaching a verdict -- read its output
above. Nothing has been written." ;;
  esac
fi

S_MAX=$("$PY_ABS" -c "import json;print(json.load(open('$AUDIT'))['recommended_s_max'])")
PROBE=train/configs/probe/dinov3_wildfake_multiscale.yaml

# Every config that carries the window protocol. The number has to be restated
# wherever the protocol is -- stage 0 fits the head on those windows, four caches
# render them, two training runs fingerprint them -- and seven files edited by
# hand is seven chances to typo the one number the whole protocol rests on.
# `tests/train/test_configs.py::test_every_crop_config_agrees_on_the_range`
# is the check that they still match afterwards.
CROP_CONFIGS=(
  "$PROBE"
  train/configs/cache/dinov3_multiscale.yaml
  train/configs/cache/dinov3_multiscale_val.yaml
  train/configs/cache/wildfake_freq.yaml
  train/configs/cache/wildfake_freq_val.yaml
  train/configs/train/dinov3_multiscale.yaml
  train/configs/train/dinov3_enrich.yaml
)

if (( WRITE )); then
  step "4  writing the audited range into ${#CROP_CONFIGS[@]} configs"
  for cfg in "${CROP_CONFIGS[@]}"; do
    if grep -qE '^\s+s_max:' "$cfg"; then
      note "$cfg: s_max already set -- leaving it alone"
      continue
    fi
    # Replace the placeholder comment, so each file keeps its shape and the
    # provenance of the number stays next to it.
    "$PY_ABS" - "$cfg" "$S_MAX" "$AUDIT" <<'PY'
import sys
path, s_max, audit = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path, encoding="utf-8").read()
marker = "  # s_max: <-- SET ME from audit_sizes.py before running"
if marker not in text:
    raise SystemExit(f"placeholder line not found in {path}; set s_max by hand")
open(path, "w", encoding="utf-8").write(
    text.replace(marker, f"  s_max: {s_max}          # from {audit}")
)
print(f"   {path}: s_max {s_max}")
PY
  done
  note "tests/train/test_configs.py::test_the_crop_range_must_be_audited"
  note "will now fail -- that is the signal to EMPTY AWAITING_AUDIT, not to"
  note "soften the check. test_every_crop_config_agrees_on_the_range keeps them"
  note "in step from here on."
fi

cat <<EOF

$(printf '\033[1m== GATE: the crop range\033[0m')

The audit recommends:

    s_max: $S_MAX

$(if (( WRITE )); then echo "Written into all ${#CROP_CONFIGS[@]} configs that carry the crop block."; else cat <<'INNER'
It goes in the `crop:` block of SEVEN configs -- the probe, four caches and two
training runs -- each of which currently has s_max commented out on purpose:

    train/configs/probe/dinov3_wildfake_multiscale.yaml
    train/configs/cache/dinov3_multiscale.yaml
    train/configs/cache/dinov3_multiscale_val.yaml
    train/configs/cache/wildfake_freq.yaml
    train/configs/cache/wildfake_freq_val.yaml
    train/configs/train/dinov3_multiscale.yaml
    train/configs/train/dinov3_enrich.yaml

Rerun this script with --write-range and it will do all seven, which is the safe
way: a head fit at one range and a cache rendered at another is a head scored on
a feature space it was never fit on.
INNER
fi)

Read $AUDIT before you accept it. The E-cropsize
column is the one that matters: it is the AUC of realized crop size treated as a
classifier, and it must sit at chance. If the recommendation is much smaller than
512, that is the corpus telling you its two classes are not the same size, and
the narrow range is the honest one.

Then:

    bash scripts/after_audit.sh

which refits stage 0 on crops, runs the 32x32 gate, and measures P2' on both
evaluation arms.
EOF
