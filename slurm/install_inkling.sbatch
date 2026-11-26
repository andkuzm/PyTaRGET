#!/usr/bin/bash
# One-off: separate venv (venv_vllm stays untouched) with the vLLM NIGHTLY (needed for Inkling-2-Mini, vLLM >= 0.35, CUDA 13 wheels) and the official weights.
#   sbatch slurm/install_inkling.sbatch      (from the PyTaRGET directory)
#SBATCH -J install-inkling
#SBATCH -p amd
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -euo pipefail
export UV_HTTP_TIMEOUT=900
VENV="$HOME/venv_inkling"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip uv
# pinned nightly wheel: otherwise the PyPI 0.34.0 release outranks it
"$VENV/bin/uv" pip install --python "$VENV/bin/python" -U "vllm==0.34.0rc1.dev156+g4e3463c87" --prerelease=allow    \
    \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130 --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
"$VENV/bin/python" -I -c "import importlib.metadata as m;print({p:m.version(p) for p in ('vllm','torch','transformers')})"
"$VENV/bin/python" -I -c "import vllm,os,subprocess;d=os.path.dirname(vllm.__file__);print('inkling files:',subprocess.run(['grep','-ril','inkling',d],capture_output=True,text=True).stdout[:1500])"
echo "=== download weights $(date)"
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 "$VENV/bin/python" -I -c "
from huggingface_hub import snapshot_download
p = snapshot_download('thinkingmachines/Inkling-2-Mini'); print(p)"
du -shL "$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
echo "=== done $(date)"
