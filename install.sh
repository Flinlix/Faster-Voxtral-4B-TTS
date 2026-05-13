#!/usr/bin/env bash
# install.sh — one-shot installer for Faster Voxtral 4B TTS
#
# Usage:
#   ./install.sh                  # NF4 quantization (recommended, ~5 GB VRAM)
#   ./install.sh --int8           # INT8 quantization (~6 GB VRAM)
#   ./install.sh --bf16           # Full BF16 precision (~9 GB VRAM)
#   ./install.sh --cuda cu124     # Override CUDA wheel index (default: cu126)
#   ./install.sh --max-jobs 4     # Cap flash-attn build parallelism (lower if you OOM)

set -euo pipefail

# Always run from the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_VERSION="3.12"       # 3.11–3.13 work; 3.12 has the best wheel coverage
CUDA_VERSION="cu126"        # match your driver: cu118/cu121/cu124/cu126/cu128...
FLASH_ATTN_VERSION="2.8.3"  # keep in sync with torch; 2.8.x supports torch 2.4+
MAX_JOBS_OVERRIDE=""        # set e.g. "4" or lower if flash-attn build OOMs on your machine

# ── Parse flags ────────────────────────────────────────────────────────
EXTRA="nf4"
while [ $# -gt 0 ]; do
    case "$1" in
        --int8) EXTRA="int8"; shift ;;
        --bf16) EXTRA=""; shift ;;
        --cuda) CUDA_VERSION="$2"; shift 2 ;;
        --max-jobs) MAX_JOBS_OVERRIDE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--int8 | --bf16] [--cuda cuXXX] [--max-jobs N]"
            echo "  (default)   NF4  ~5 GB VRAM"
            echo "  --int8      INT8 ~6 GB VRAM"
            echo "  --bf16      BF16 ~9 GB VRAM"
            echo "  --cuda      PyTorch CUDA wheel suffix (default: cu126)"
            echo "  --max-jobs  Cap flash-attn build parallelism"
            exit 0
            ;;
        *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VERSION}"

# ── Sanity checks ──────────────────────────────────────────────────────
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in ${SCRIPT_DIR}" >&2
    exit 1
fi

require() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' is required but not installed. ${2:-}" >&2
        exit 1
    fi
}

require curl "Install with: apt install curl  /  brew install curl"
require git  "Install with: apt install git   /  brew install git"
require cc   "A C/C++ toolchain is required (apt install build-essential)."

if ! command -v nvcc &>/dev/null; then
    echo "WARNING: 'nvcc' (CUDA toolkit) not found on PATH." >&2
    echo "         flash-attn must compile CUDA kernels and will likely fail." >&2
    echo "         Install the CUDA toolkit matching your driver:" >&2
    echo "           https://developer.nvidia.com/cuda-downloads" >&2
fi

# ── Ensure uv is available ─────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "uv not found — installing via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv installation failed. Install it manually: https://docs.astral.sh/uv/" >&2
        exit 1
    fi
    echo "uv installed: $(uv --version)"
fi

# ── Create / reuse venv (validate Python version too) ─────────────────
need_new_venv=1
if [ -d ".venv" ] && .venv/bin/python --version &>/dev/null; then
    actual_ver="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "${actual_ver}" = "${PYTHON_VERSION}" ]; then
        echo "Reusing existing .venv (Python ${actual_ver})"
        need_new_venv=0
    else
        echo "Existing .venv has Python ${actual_ver}, expected ${PYTHON_VERSION} — recreating..."
        rm -rf .venv
    fi
elif [ -d ".venv" ]; then
    echo "Existing .venv is broken — recreating..."
    rm -rf .venv
fi
if [ "${need_new_venv}" = "1" ]; then
    echo "Creating virtual environment (Python ${PYTHON_VERSION})..."
    uv venv --python "${PYTHON_VERSION}"
fi

# Large NVIDIA wheels (200-800 MB) need a generous HTTP timeout
export UV_HTTP_TIMEOUT=600

# ── Install torch from CUDA index first ───────────────────────────────
echo "Installing PyTorch (CUDA ${CUDA_VERSION})..."
uv pip install --python .venv/bin/python \
    --extra-index-url "${TORCH_INDEX}" \
    --extra-index-url "https://pypi.nvidia.com" \
    "torch>=2.4"

# ── Install flash-attn ────────────────────────────────────────────────
echo "Installing flash-attn ${FLASH_ATTN_VERSION} (compiling CUDA kernels — may take 10–30 minutes)..."
# flash-attn 2.8.3's pyproject.toml uses the old `license = "string"` format
# which setuptools >= 67 rejects. Pin setuptools to a version that still
# accepts it AND drive the build via the venv's `python -m pip` (not uv) so
# get_build_requires runs inside our venv with our pinned setuptools.
uv pip install --python .venv/bin/python \
    "setuptools==66.1.0" "pip>=24" ninja packaging wheel psutil

flash_env=()
if [ -n "${MAX_JOBS_OVERRIDE}" ]; then
    flash_env+=("MAX_JOBS=${MAX_JOBS_OVERRIDE}")
fi

env "${flash_env[@]}" .venv/bin/python -m pip install \
    --no-build-isolation \
    "flash-attn==${FLASH_ATTN_VERSION}"

# ── Install the package itself ─────────────────────────────────────────
# Pass the same indices so any torch-pinning in [project] is resolved from
# the CUDA wheel index rather than vanilla PyPI.
if [ -n "${EXTRA}" ]; then
    echo "Installing package with [${EXTRA}] extras..."
    uv pip install --python .venv/bin/python \
        --extra-index-url "${TORCH_INDEX}" \
        --extra-index-url "https://pypi.nvidia.com" \
        -e ".[${EXTRA}]"
else
    echo "Installing package (BF16, no extras)..."
    uv pip install --python .venv/bin/python \
        --extra-index-url "${TORCH_INDEX}" \
        --extra-index-url "https://pypi.nvidia.com" \
        -e .
fi

# ── Verify ─────────────────────────────────────────────────────────────
echo ""
echo "Verifying installation..."
.venv/bin/python - <<'PY'
import importlib, sys
ok = True
for mod in ("torch", "flash_attn", "voxtral"):
    try:
        m = importlib.import_module(mod)
        print(f"  ok    {mod} {getattr(m, '__version__', '(no __version__)')}")
    except Exception as e:
        print(f"  FAIL  {mod}: {e}")
        ok = False
sys.exit(0 if ok else 1)
PY

# ── Done ───────────────────────────────────────────────────────────────
echo ""
echo "Installation complete."
echo ""
echo "Start the server:"
echo "  source .venv/bin/activate && voxtral-server"
echo ""
echo "Or without activating:"
echo "  .venv/bin/voxtral-server"
