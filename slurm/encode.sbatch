#!/usr/bin/bash
# Tokenize the cases for one model and split them by commit (oldest commits train, then validation, then test).
# CPU only; the whole CSV is read into memory (est. 64-96 GB for the 6.6 GB file, not measured: see "peak RAM" in the log).
# Partition amd, not main: on 2026-11-06 the scheduler estimated the start in main for 12 Dec (also for 2 CPUs) and at once in amd.
#   sbatch --export=ALL,MODEL=codet5p slurm/encode.sbatch      (run from the PyTaRGET directory; logs: slurm/logs/)
#SBATCH -J encode
#SBATCH -p amd
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o slurm/logs/%x-%j.out
#SBATCH --requeue
set -euo pipefail
source "${PYTAROGET_DIR:-$HOME/PyTaRGET}/slurm/env.sh"
: "${MODEL:?set MODEL=codet5p|plbart|codegen, e.g. sbatch --export=ALL,MODEL=codet5p slurm/encode.sbatch}"
echo "encode $MODEL on $(hostname), $CSV -> $OUT"
run_stage encode --model "$MODEL"
