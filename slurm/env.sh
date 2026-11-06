# Sourced by the sbatch scripts in this directory.
# Everything can be overridden when submitting: sbatch --export=ALL,CSV=...,OUT=...,PY=... <script>
export ROOT="${PYTAROGET_DIR:-$HOME/PyTaRGET}"
export PY="${PY:-$HOME/venv3.12/bin/python}"
export CSV="${CSV:-$HOME/annotated_cases_v2_pr59_filtered_cap_clean.csv}"   # the final filtered cases
export OUT="${OUT:-$HOME/eval_results}"                                      # <OUT>/<model>/<train fraction>/...
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"            # the base models are in ~/.cache/huggingface; no network needed
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Run one stage in the background and pass Slurm's SIGTERM on to it, so the stage can save its state before the
# job ends (the scripts ask Slurm for the signal 5 minutes before the time limit: #SBATCH --signal=B:TERM@300).
run_stage() {
    "$PY" -I "$ROOT/slurm/run_stage.py" "$@" &
    local pid=$! rc=0
    trap 'kill -TERM "$pid" 2>/dev/null' TERM
    wait "$pid" || rc=$?
    if kill -0 "$pid" 2>/dev/null; then wait "$pid" || rc=$?; fi    # the first wait returns early when the trap fired
    trap - TERM
    return "$rc"
}
