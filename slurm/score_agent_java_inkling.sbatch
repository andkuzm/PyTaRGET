#!/usr/bin/bash
# CPU: score the Java Inkling agent run, strict (final method) and relaxed (whole final test file), as for the Qwen runs. Resumable.     WARNING: runs third-party builds with model-written code.
#SBATCH -J score-agent-java
#SBATCH -p amd
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
cd "$HOME/PyTaRGET"
N="${NAME:-java_inkling_low_80}"; R="${ROOT:-$HOME/agent_java_ink}"
for mode in strict relaxed; do
  out="$HOME/eval_agent_plaus/$N.json"; flag=""; [ $mode = relaxed ] && { out="$HOME/eval_agent_plaus/${N}_relaxed.json"; flag="--relaxed"; }
  S="/tmp/javaagent_${USER}_${SLURM_JOB_ID:-manual}_$mode"; mkdir -p "$S"
  echo "=== $mode $(date)"
  env -u GH_TOKEN -u HF_TOKEN PYTHONPATH="$HOME/PyTaRGET" PYTHONUNBUFFERED=1 "$HOME/venv3.12/bin/python" scripts/agent/java_agent_score.py --root "$R" --agent-out "$HOME/eval_agent/$N" --out "$out" --workdir "$S" --workers 3 $flag
  echo "=== $mode exit $? $(date)"; rm -rf "$S"
done
