#!/usr/bin/bash
# Submit encode -> train -> test for each given model; the models are independent chains and run in parallel.
#   slurm/submit_pipeline.sh [--gpu a100-80g] [--from encode|train|test] [--dry-run] MODEL [MODEL...]
#   slurm/submit_pipeline.sh --gpu h200-141g codet5p plbart codegen
# Each stage is started when the previous one finished successfully (afterok). A stage that was stopped by the time limit
# exits with status 75, which cancels the stages after it: submit the same stage again (--from train) and the chain
# continues from the saved state. Run from anywhere; logs go to <PyTaRGET>/slurm/logs/.
set -euo pipefail
ROOT="${PYTAROGET_DIR:-$HOME/PyTaRGET}"
GPU=a100-80g; FROM=encode; DRY=0; MODELS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --from) FROM="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        codet5p|plbart|codegen) MODELS+=("$1"); shift ;;
        *) echo "unknown argument: $1 (models: codet5p plbart codegen)" >&2; exit 2 ;;
    esac
done
[ ${#MODELS[@]} -gt 0 ] || { echo "give at least one model: codet5p plbart codegen" >&2; exit 2; }
case "$FROM" in encode|train|test) ;; *) echo "--from must be encode, train or test" >&2; exit 2 ;; esac
cd "$ROOT"; mkdir -p slurm/logs

submit() {   # submit <dependency job id or ""> <sbatch args...>; prints the job id
    local dependency="$1"; shift
    local args=(--parsable)
    [ -n "$dependency" ] && args+=(--dependency="afterok:$dependency")
    if [ "$DRY" = 1 ]; then echo "sbatch ${args[*]} $*" >&2; echo "DRY"; else sbatch "${args[@]}" "$@"; fi
}

for model in "${MODELS[@]}"; do
    previous=""
    if [ "$FROM" = encode ]; then
        previous=$(submit "" -J "encode-$model" --export=ALL,MODEL="$model" slurm/encode.sbatch)
        echo "$model: encode job $previous"
    fi
    if [ "$FROM" = encode ] || [ "$FROM" = train ]; then
        dep="$previous"; [ "$previous" = DRY ] && dep=""
        previous=$(submit "$dep" -J "train-$model" --gres="gpu:$GPU:1" --export=ALL,MODEL="$model" slurm/train.sbatch)
        echo "$model: train job $previous"
    fi
    dep="$previous"; [ "$previous" = DRY ] && dep=""
    previous=$(submit "$dep" -J "test-$model" --gres="gpu:$GPU:1" --export=ALL,MODEL="$model" slurm/test.sbatch)
    echo "$model: test job $previous"
done
