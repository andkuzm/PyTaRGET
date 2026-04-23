#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
DEEPSEEK=0

usage() {
    echo "Usage: $0 [--deepseek]"
    echo ""
    echo "  --deepseek   Also clone and install HuggingFace/transformers from source"
    echo "               (required for DeepSeek model support)"
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --deepseek) DEEPSEEK=1 ;;
        -h|--help)  usage ;;
        *) echo "Unknown argument: $arg"; usage ;;
    esac
done

# ── find Python 3.12+ ────────────────────────────────────────────────────────
find_python() {
    for cmd in python3.13 python3.12 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            if "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON=$(find_python) || {
    echo "ERROR: Python 3.12+ not found."
    echo "Install it and make sure it is on your PATH, then re-run this script."
    exit 1
}
echo "Python : $($PYTHON --version) ($PYTHON)"

# ── create venv ──────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "Venv   : already exists at $VENV_DIR (skipping creation)"
    echo "         Delete it first if you want a clean rebuild: rm -rf $VENV_DIR"
else
    echo "Venv   : creating at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON_VENV="$VENV_DIR/bin/python"

# ── install dependencies ─────────────────────────────────────────────────────
echo ""
echo "Installing requirements.txt ..."
"$PIP" install --upgrade pip --quiet
"$PIP" install -r "$SCRIPT_DIR/requirements.txt"

# tree-sitter must be exactly 0.24.0; pip warns about conflicts — that is expected
echo ""
echo "Pinning tree-sitter==0.24.0 (conflict warnings below are expected) ..."
"$PIP" install tree-sitter==0.24.0

# ── optional: DeepSeek transformers ──────────────────────────────────────────
if [ "$DEEPSEEK" -eq 1 ]; then
    echo ""
    echo "DeepSeek: cloning HuggingFace/transformers from source ..."
    TRANSFORMERS_DIR="$SCRIPT_DIR/transformers"
    if [ -d "$TRANSFORMERS_DIR" ]; then
        echo "  Directory $TRANSFORMERS_DIR already exists, pulling latest ..."
        git -C "$TRANSFORMERS_DIR" pull --ff-only
    else
        git clone --depth 1 https://github.com/huggingface/transformers.git "$TRANSFORMERS_DIR"
    fi
    echo "  Installing in editable mode ..."
    "$PIP" install -e "$TRANSFORMERS_DIR"
fi

# ── done ─────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setup complete."
echo ""
echo "Activate:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Run code from the PARENT directory of PyTaRGET:"
echo "  cd $(dirname "$SCRIPT_DIR")"
echo "  python -c 'from PyTaRGET.data_processing.encode_tune_test import Eftt'"
echo ""
if [ "$DEEPSEEK" -eq 0 ]; then
    echo "For DeepSeek support, re-run with:  $0 --deepseek"
    echo ""
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
