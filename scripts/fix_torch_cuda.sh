#!/bin/bash
# Reinstall the active env's PyTorch from a CUDA 12.x build the GPU driver supports.
#
# Run inside the model env (boltz or chai) when torch says
# "The NVIDIA driver on your system is too old" or cuda_available=False.
# First tries the installed torch version from CUDA 12.6 / 12.4 / 12.1 builds
# (all run on driver 535+); if that version has no CUDA 12 build, installs the
# newest torch that has one. No sudo needed.
#
# Usage: conda activate boltz && scripts/fix_torch_cuda.sh
set -euo pipefail

python -c "import torch" 2>/dev/null || { echo "no torch in this env (activate boltz or chai first)" >&2; exit 1; }
TV=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "torch $TV, built for CUDA $(python -c 'import torch; print(torch.version.cuda)')"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed 's/^/driver /' || true

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA already works - nothing to do."; exit 0
fi

for cu in cu126 cu124 cu121; do
    echo "trying torch==$TV from $cu ..."
    if pip install --force-reinstall "torch==$TV" --index-url "https://download.pytorch.org/whl/$cu"; then
        if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
            python -c "import torch; print('OK:', torch.__version__, torch.cuda.get_device_name(0))"
            pip check || echo "(pip check warnings above: usually harmless version pins)"
            exit 0
        fi
        echo "installed but CUDA still unavailable with $cu"
    fi
done
# Newer torch releases have no CUDA 12.x build at all (e.g. 2.15 is CUDA 13 only):
# take the newest torch that does.
for cu in cu126 cu124; do
    echo "trying the newest torch built for $cu ..."
    if pip install --force-reinstall torch --index-url "https://download.pytorch.org/whl/$cu"; then
        if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
            python -c "import torch; print('OK:', torch.__version__, torch.cuda.get_device_name(0))"
            pip check || echo "(check the pip warnings above: a package may need a newer torch)"
            exit 0
        fi
        echo "installed but CUDA still unavailable with $cu"
    fi
done
echo "No build worked. Paste the output of scripts/check_env.sh when asking for help." >&2
exit 1
