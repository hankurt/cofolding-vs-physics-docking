#!/bin/bash
# Download the GNINA and AutoDock Vina binaries into ~/.local/bin (no sudo).
#
# Usage: scripts/install_docking_tools.sh
#   GNINA_URL / VINA_URL override the download links.
#
# GNINA 1.1 is built for CUDA 11 and runs on older drivers (e.g. 535). If it
# does not start, pick another binary from https://github.com/gnina/gnina/releases
# and set GNINA_URL, or run docking on the CPU (GNINA_ARGS=--no_gpu).
set -euo pipefail

BIN="${BIN:-$HOME/.local/bin}"
GNINA_URL="${GNINA_URL:-https://github.com/gnina/gnina/releases/download/v1.1/gnina}"
VINA_URL="${VINA_URL:-https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64}"
mkdir -p "$BIN"

get() {  # url dest
    if [ -x "$2" ]; then echo "have $2"; return; fi
    echo "downloading $1"
    if command -v wget >/dev/null 2>&1; then wget -q --show-progress -O "$2.part" "$1"
    else curl -fL --progress-bar -o "$2.part" "$1"; fi
    mv "$2.part" "$2"
    chmod +x "$2"
}

get "$VINA_URL" "$BIN/vina"
get "$GNINA_URL" "$BIN/gnina"

case ":$PATH:" in
    *":$BIN:"*) ;;
    *) echo; echo "add to ~/.bashrc:  export PATH=\"$BIN:\$PATH\"" ;;
esac
echo
"$BIN/vina" --version || true
"$BIN/gnina" --version 2>&1 | head -3 || true
