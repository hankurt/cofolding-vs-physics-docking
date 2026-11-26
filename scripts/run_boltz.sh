#!/bin/bash
# Run Boltz-2 on every target YAML written by `cofoldbench inputs`.
#
# Usage: scripts/run_boltz.sh [blind|pocket] [extra boltz args...]
#   blind  (default)  sequence + SMILES only
#   pocket            with the binding-site pocket constraint (fair vs docking)
#
# Needs: pip install boltz  (and a CUDA GPU). --use_msa_server sends the
# sequences to the public ColabFold MSA server; drop it and add msa: paths
# to the YAMLs if the sequences must stay private.
#
# Boltz prints "WARNING: ran out of memory, skipping batch" and carries on, so
# a target too large for the GPU silently produces nothing. This script finds
# those targets afterwards and retries each one alone with a shallower MSA
# (RETRY_ARGS), then reports any that still have no output, so they are never
# mistaken for targets the model simply got wrong.
#
# Env: SAMPLES (5), PARALLEL (1), KERNELS (0), RETRY (1 = retry OOM targets),
#      RETRY_ARGS (memory-saving args for the retry pass).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
need boltz "Activate the boltz env (see docs/RUNBOOK.md)"

VARIANT="${1:-blind}"; shift || true
IN="inputs/boltz/${VARIANT}"
OUT="runs/boltz2_${VARIANT}"
SAMPLES="${SAMPLES:-5}"
# One sample at a time keeps memory low enough for 12 GB GPUs; raise on bigger cards.
PARALLEL="${PARALLEL:-1}"
# Fast cuEquivariance kernels need `pip install cuequivariance-torch
# cuequivariance-ops-torch-cu12`; without them run with KERNELS=0 (default).
KERNELS="${KERNELS:-0}"
KERNEL_FLAG=""; [ "$KERNELS" = "0" ] && KERNEL_FLAG="--no_kernels"
RETRY="${RETRY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[ -d "$IN" ] || { echo "no $IN - run: cofoldbench inputs" >&2; exit 1; }
mkdir -p "$OUT"

# Targets in $IN with no predicted model under $OUT.
missing_targets() {
    local yaml target
    for yaml in "$IN"/*.yaml; do
        target=$(basename "$yaml" .yaml)
        if ! compgen -G "$OUT/**/predictions/$target/${target}_model_*" >/dev/null 2>&1 \
           && ! compgen -G "$OUT/*/predictions/$target/${target}_model_*" >/dev/null 2>&1; then
            echo "$target"
        fi
    done
}

# Only pass memory-saving flags this boltz version actually has.
default_retry_args() {
    local help args=""
    help=$(boltz predict --help 2>&1 || true)
    case "$help" in *--subsample_msa*) args="$args --subsample_msa" ;; esac
    case "$help" in *--num_subsampled_msa*) args="$args --num_subsampled_msa 512" ;; esac
    case "$help" in *--sampling_steps*) args="$args --sampling_steps 100" ;; esac
    echo "$args"
}

shopt -s globstar nullglob

boltz predict "$IN" \
    --out_dir "$OUT" \
    --use_msa_server \
    --diffusion_samples "$SAMPLES" \
    --max_parallel_samples "$PARALLEL" \
    --output_format pdb \
    $KERNEL_FLAG \
    "$@" || echo "boltz exited non-zero; checking which targets have output" >&2

if [ "$RETRY" = "1" ]; then
    mapfile -t GONE < <(missing_targets)
    if [ "${#GONE[@]}" -gt 0 ]; then
        RETRY_ARGS="${RETRY_ARGS:-$(default_retry_args)}"
        echo
        echo "=== ${#GONE[@]} target(s) produced nothing (most likely out of memory):"
        echo "    ${GONE[*]}"
        echo "=== retrying one at a time with:$RETRY_ARGS"
        for target in "${GONE[@]}"; do
            echo "--- retry $target"
            boltz predict "$IN/$target.yaml" \
                --out_dir "$OUT" \
                --use_msa_server \
                --diffusion_samples "$SAMPLES" \
                --max_parallel_samples 1 \
                --output_format pdb \
                $KERNEL_FLAG $RETRY_ARGS "$@" \
                || echo "    $target failed again" >&2
        done
    fi
fi

mapfile -t STILL_GONE < <(missing_targets)
if [ "${#STILL_GONE[@]}" -gt 0 ]; then
    echo
    echo "WARNING: no Boltz output for ${#STILL_GONE[@]} target(s): ${STILL_GONE[*]}" >&2
    echo "These count as failures in the report (cofoldbench report lists them)." >&2
    echo "A target too large for this GPU should be dropped from data/targets.csv" >&2
    echo "for every method, not left failing for Boltz alone." >&2
fi

METHOD="boltz2"; [ "$VARIANT" = "pocket" ] && METHOD="boltz2_pocket"
cfb collect boltz --results "$OUT" --method "$METHOD"
