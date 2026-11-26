#!/bin/bash
# Run Chai-1 on every target FASTA written by `cofoldbench inputs`.
#
# Usage: scripts/run_chai.sh [extra chai-lab args...]
#
# Needs: pip install chai_lab pyarrow  (and a CUDA GPU). --use-msa-server sends
# the sequences to the public ColabFold MSA server; remove it to run
# single-sequence.
#
# Chai-1 needs more GPU memory than Boltz-2 and runs out on larger targets with
# 12 GB. One target failing must not end the run, so each is attempted on its
# own; a target that fails is retried with fewer trunk recycles and diffusion
# steps (RETRY_ARGS), and the targets that needed that fallback are listed in
# runs/chai1/_reduced_settings.txt so the write-up can say so. Targets that
# still produce nothing are reported and count as failures in the report.
#
# Env: RETRY (1), RETRY_ARGS (memory-saving args for the retry pass).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
need chai-lab "Activate the chai env (see docs/RUNBOOK.md)"

IN="inputs/chai"
OUT="runs/chai1"
[ -d "$IN" ] || { echo "no $IN - run: cofoldbench inputs" >&2; exit 1; }
# chai_lab saves its MSAs as parquet but does not install a parquet engine.
PQ_ERR="$(mktemp)"; PQ_OUT="$(mktemp -u).parquet"
if ! python - "$PQ_OUT" 2>"$PQ_ERR" <<'EOF'
import sys
import pandas as pd
pd.DataFrame({"a": [1]}).to_parquet(sys.argv[1])
EOF
then
    echo "error: this env cannot write parquet, which Chai needs for its MSAs:" >&2
    tail -5 "$PQ_ERR" >&2
    echo "try: pip install -U pyarrow" >&2
    rm -f "$PQ_ERR" "$PQ_OUT"; exit 1
fi
rm -f "$PQ_ERR" "$PQ_OUT"
mkdir -p "$OUT"
RETRY="${RETRY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Only pass memory-saving flags this chai-lab version actually has.
default_retry_args() {
    local help args=""
    help=$(chai-lab fold --help 2>&1 || true)
    case "$help" in *--num-trunk-recycles*) args="$args --num-trunk-recycles 1" ;; esac
    case "$help" in *--num-diffn-timesteps*) args="$args --num-diffn-timesteps 100" ;; esac
    case "$help" in *--low-memory*) args="$args --low-memory" ;; esac
    echo "$args"
}

fold_one() {  # fasta target [extra args...]
    local fasta="$1" target="$2"; shift 2
    # Chai refuses to write into a non-empty directory: clear any earlier attempt.
    rm -rf "$OUT/$target"
    chai-lab fold "$fasta" "$OUT/$target" --use-msa-server "$@"
}

FAILED=(); REDUCED=()
for fasta in "$IN"/*.fasta; do
    target=$(basename "$fasta" .fasta)
    if [ -f "$OUT/$target/pred.model_idx_0.cif" ]; then
        echo "skip $target (done)"; continue
    fi
    echo "=== $target"
    if fold_one "$fasta" "$target" "$@" > "$OUT/$target.log" 2>&1; then
        continue
    fi
    echo "    failed: $(grep -oiE 'CUDA out of memory|[A-Za-z]*Error[^\n]{0,60}' "$OUT/$target.log" | tail -1)"
    if [ "$RETRY" != "1" ]; then FAILED+=("$target"); continue; fi
    RETRY_ARGS="${RETRY_ARGS:-$(default_retry_args)}"
    if [ -z "$RETRY_ARGS" ]; then
        echo "    no memory-saving flags available in this chai-lab"; FAILED+=("$target"); continue
    fi
    echo "    retrying with:$RETRY_ARGS"
    if fold_one "$fasta" "$target" $RETRY_ARGS "$@" > "$OUT/$target.retry.log" 2>&1; then
        REDUCED+=("$target")
    else
        echo "    failed again: $(grep -oiE 'CUDA out of memory|[A-Za-z]*Error[^\n]{0,60}' "$OUT/$target.retry.log" | tail -1)"
        FAILED+=("$target")
    fi
done

echo
if [ "${#REDUCED[@]}" -gt 0 ]; then
    printf '%s\n' "${REDUCED[@]}" >> "$OUT/_reduced_settings.txt"
    echo "NOTE: ${#REDUCED[@]} target(s) needed reduced sampling to fit the GPU: ${REDUCED[*]}"
    echo "      Recorded in $OUT/_reduced_settings.txt - say so when reporting Chai-1 results."
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "WARNING: no Chai output for ${#FAILED[@]} target(s): ${FAILED[*]}" >&2
    echo "  These count as failures in the report (cofoldbench report lists them)." >&2
    echo "  Logs: $OUT/<target>.log and $OUT/<target>.retry.log" >&2
fi

cfb collect chai --results "$OUT"
