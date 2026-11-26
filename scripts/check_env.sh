#!/bin/bash
# Print what this machine has for the benchmark: GPU, driver, conda envs, tools.
# Paste the output when asking for help.
#
# Usage: scripts/check_env.sh        (env names: CFB_ENV, BOLTZ_ENV, CHAI_ENV)
source "$(dirname "$0")/_common.sh"

CFB_ENV="${CFB_ENV:-cofoldbench}"; BOLTZ_ENV="${BOLTZ_ENV:-boltz}"; CHAI_ENV="${CHAI_ENV:-chai}"

echo "== machine"
uname -srm; echo "cores: $(nproc)"; free -g | awk '/Mem:/{print "RAM: " $2 " GB"}'
df -h . | awk 'NR==2{print "disk free here: " $4}'

echo; echo "== GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
    nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true
else
    echo "nvidia-smi not found"
fi

echo; echo "== conda"
if command -v conda >/dev/null 2>&1; then
    conda --version; conda env list | grep -v "^#"
else
    echo "conda not found (install Miniforge in your home directory: see docs/RUNBOOK.md)"
fi

torch_check() {  # env
    conda run -n "$1" python -c "
import torch
ok = torch.cuda.is_available()
print(f'  torch {torch.__version__} (CUDA {torch.version.cuda}) cuda_available={ok}',
      torch.cuda.get_device_name(0) if ok else '')" 2>&1 | grep -v '^\s*$' | tail -2
}

echo; echo "== environments"
for env in "$CFB_ENV" "$BOLTZ_ENV" "$CHAI_ENV"; do
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$env"; then
        echo "$env:"
        case "$env" in
            "$CFB_ENV")
                conda run -n "$env" python -c "
import cofoldbench, gemmi, rdkit
print('  cofoldbench', cofoldbench.__version__, '| gemmi', gemmi.__version__, '| rdkit', rdkit.__version__)
try:
    import meeko, openbabel; print('  docking prep: meeko', meeko.__version__, '+ openbabel ok')
except ImportError as e: print('  docking prep missing:', e)" 2>&1 | grep -v '^\s*$' | tail -3 ;;
            "$BOLTZ_ENV")
                conda run -n "$env" python -c "import importlib.metadata as m; print('  boltz', m.version('boltz'))" 2>&1 | grep -v '^\s*$' | tail -1
                torch_check "$env" ;;
            "$CHAI_ENV")
                conda run -n "$env" python -c "import importlib.metadata as m; print('  chai_lab', m.version('chai_lab'))" 2>&1 | grep -v '^\s*$' | tail -1
                torch_check "$env" ;;
        esac
    else
        echo "$env: not created yet"
    fi
done

echo; echo "== docking binaries"
if command -v vina >/dev/null 2>&1; then echo "vina: $(vina --version 2>&1 | head -1)"; else echo "vina: not on PATH"; fi
if command -v gnina >/dev/null 2>&1; then
    echo "gnina: $(gnina --version 2>&1 | grep -v '^\s*$' | head -1)"
else
    echo "gnina: not on PATH"
fi

echo; echo "== benchmark state"
[ -f data/targets.csv ] && echo "targets: $(grep -vc '^#' data/targets.csv | awk '{print $1-1}')"
for d in inputs/boltz/blind inputs/dock runs/boltz2_blind runs/boltz2_pocket runs/chai1 runs/gnina runs/vina; do
    [ -d "$d" ] && echo "$d: $(ls "$d" | wc -l) entries"
done
[ -f results/manifest.csv ] && echo "manifest rows: $(($(wc -l < results/manifest.csv) - 1))"
[ -f results/results.csv ] && echo "results rows: $(($(wc -l < results/results.csv) - 1))"
exit 0
