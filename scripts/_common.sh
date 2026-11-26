# Shared helpers for the run scripts (sourced, not run).

# cfb: run the `cofoldbench` command even when another environment (boltz,
# chai) is active. Looks on PATH first, then in the cofoldbench conda env
# (name from $CFB_ENV, default "cofoldbench") under the usual env locations.
cfb() {
    if command -v cofoldbench >/dev/null 2>&1; then
        cofoldbench "$@"; return
    fi
    local env="${CFB_ENV:-cofoldbench}" base="" exe
    if [ -n "${CONDA_EXE:-}" ]; then base="$(dirname "$(dirname "$CONDA_EXE")")"; fi
    for exe in "$base/envs/$env/bin/cofoldbench" "$HOME/.conda/envs/$env/bin/cofoldbench" \
               "$HOME/miniforge3/envs/$env/bin/cofoldbench" "$HOME/miniconda3/envs/$env/bin/cofoldbench"; do
        if [ -x "$exe" ]; then "$exe" "$@"; return; fi
    done
    echo "error: cofoldbench not found. Activate the cofoldbench env and run:" >&2
    echo "  cofoldbench $*" >&2
    return 1
}

need() {
    command -v "$1" >/dev/null 2>&1 || { echo "error: '$1' not found. $2" >&2; exit 1; }
}
