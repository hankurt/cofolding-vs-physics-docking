#!/bin/bash
# Dock every target in inputs/dock/ with GNINA and AutoDock Vina, then collect.
#
# Usage: scripts/run_docking.sh [gnina|vina|both]     (default: both)
#   Run `cofoldbench dockprep` first (writes receptor, ligand, box, PDBQT).
#   Binaries: scripts/install_docking_tools.sh (no sudo needed).
#
# NOTE: GNINA scores poses with a CNN on the GPU, so do NOT run it while a
# co-folding model is using the card - it will abort with out-of-memory. Either
# wait for Boltz/Chai to finish, or run GNINA on the CPU with
# GNINA_ARGS=--no_gpu (slower, same poses). Vina is CPU-only and can run at
# any time.
#
# Each engine's failure is independent: a GNINA crash no longer skips Vina for
# the same target. Re-running skips targets that already have output.
#
# Env: EXHAUSTIVENESS (16), NUM_MODES (9), SEED (1), CPU (all cores),
#      GNINA_ARGS (extra GNINA flags, e.g. --no_gpu), VINA_ARGS.
#
# SEED=2 METHOD_SUFFIX=_s2 re-runs everything under a second seed, collected as
# separate methods (gnina_s2, vina_s2). Comparing the two says how much of a
# difference between methods is just docking run-to-run noise.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

WHICH="${1:-both}"
IN="inputs/dock"
EXH="${EXHAUSTIVENESS:-16}"
MODES="${NUM_MODES:-9}"
SEED="${SEED:-1}"
# A suffix keeps a second seed's runs and manifest rows separate from the first.
SUFFIX="${METHOD_SUFFIX:-}"
CPU="${CPU:-$(nproc)}"
[ -d "$IN" ] || { echo "no $IN - run: cofoldbench dockprep" >&2; exit 1; }
mkdir -p "runs/gnina$SUFFIX" "runs/vina$SUFFIX"

FAILED_GNINA=(); FAILED_VINA=()

run_gnina() {  # target dir
    local t="$1" d="$2" out="runs/gnina$SUFFIX/$1.sdf" tmp="runs/gnina$SUFFIX/$1.tmp.sdf"
    if [ -s "$out" ]; then echo "gnina $t: done"; return 0; fi
    echo "gnina $t"
    rm -f "$tmp"
    if ! gnina -r "$d/receptor_h.pdb" -l "$d/ligand.sdf" --config "$d/box.txt" \
               --exhaustiveness "$EXH" --num_modes "$MODES" --seed "$SEED" --cpu "$CPU" \
               -o "$tmp" ${GNINA_ARGS:-} > "runs/gnina$SUFFIX/$t.log" 2>&1; then
        rm -f "$tmp"
        echo "  gnina failed on $t: $(tail -2 "runs/gnina$SUFFIX/$t.log" | tr '\n' ' ')"
        return 1
    fi
    mv "$tmp" "$out"
}

run_vina() {  # target dir
    local t="$1" d="$2" out="runs/vina$SUFFIX/$1.pdbqt" tmp="runs/vina$SUFFIX/$1.tmp.pdbqt"
    if [ -s "$out" ]; then echo "vina $t: done"; return 0; fi
    if [ ! -f "$d/receptor.pdbqt" ]; then
        echo "vina $t: no PDBQT - pip install -e '.[dock]' and rerun dockprep"; return 1
    fi
    echo "vina $t"
    rm -f "$tmp"
    if ! vina --receptor "$d/receptor.pdbqt" --ligand "$d/ligand.pdbqt" --config "$d/box.txt" \
              --exhaustiveness "$EXH" --num_modes "$MODES" --seed "$SEED" --cpu "$CPU" \
              --out "$tmp" ${VINA_ARGS:-} > "runs/vina$SUFFIX/$t.log" 2>&1; then
        rm -f "$tmp"
        echo "  vina failed on $t: $(tail -2 "runs/vina$SUFFIX/$t.log" | tr '\n' ' ')"
        return 1
    fi
    mv "$tmp" "$out"
}

[ "$WHICH" = "vina" ] || need gnina "Run scripts/install_docking_tools.sh"
[ "$WHICH" = "gnina" ] || need vina "Run scripts/install_docking_tools.sh"

for d in "$IN"/*/; do
    d="${d%/}"
    t=$(basename "$d")
    # Each engine is attempted independently: one crashing must not cost the other.
    if [ "$WHICH" != "vina" ]; then
        if run_gnina "$t" "$d" && \
           cfb collect gnina --target "$t" --receptor "$d/receptor.pdb" \
               --poses "runs/gnina$SUFFIX/$t.sdf" --method "gnina$SUFFIX"; then
            :
        else
            FAILED_GNINA+=("$t")
        fi
    fi
    if [ "$WHICH" != "gnina" ]; then
        if run_vina "$t" "$d" && \
           cfb collect vina --target "$t" --receptor "$d/receptor.pdb" \
               --poses "runs/vina$SUFFIX/$t.pdbqt" --method "vina$SUFFIX"; then
            :
        else
            FAILED_VINA+=("$t")
        fi
    fi
done

echo
if [ "${#FAILED_GNINA[@]}" -gt 0 ]; then
    echo "WARNING: gnina produced nothing for ${#FAILED_GNINA[@]} target(s): ${FAILED_GNINA[*]}" >&2
    echo "  These count as failures in the report. If the logs say out of memory," >&2
    echo "  the GPU was busy: rerun once it is free, or use GNINA_ARGS=--no_gpu." >&2
fi
if [ "${#FAILED_VINA[@]}" -gt 0 ]; then
    echo "WARNING: vina produced nothing for ${#FAILED_VINA[@]} target(s): ${FAILED_VINA[*]}" >&2
fi
if [ "${#FAILED_GNINA[@]}" -eq 0 ] && [ "${#FAILED_VINA[@]}" -eq 0 ]; then
    echo "all targets docked"
fi
