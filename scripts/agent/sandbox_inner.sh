#!/usr/bin/bash
# Runs inside `unshare -m` (own mount namespace, inside the job's private network namespace). Hides the home directory and the shared file system, then
# makes visible ONLY: the case workspace (read/write), the agent's output directory (read/write) and the agent virtualenv (read-only is not enforced).
# usage: sandbox_inner.sh <workspace> <outdir> <extra visible path>... -- <command...>
set -euo pipefail
ws="$1"; out="$2"; shift 2
extra=(); while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
real=$(readlink -f "$HOME/venv_agent/bin/python"); case "$real" in /gpfs/space/software/*) pybase=/gpfs/space/software;; *) pybase=$(dirname "$(dirname "$real")");; esac      # the interpreter of the venv (cluster software, not user data)
keep=("$ws" "$out" "$HOME/venv_agent" "$pybase" "${extra[@]}")
tmp=$(mktemp -d)                                   # staging area outside the hidden trees
for p in "${keep[@]}"; do mkdir -p "$tmp$p"; mount --rbind "$p" "$tmp$p"; done
mount -t tmpfs tmpfs /gpfs                        # hide the whole shared file system (home, other users, datasets)
for p in "${keep[@]}"; do mkdir -p "$p"; mount --rbind "$tmp$p" "$p"; done
umount -l "$tmp" 2>/dev/null || true
export HOME="$out/home"; mkdir -p "$HOME"          # the agent's home is in its output directory, not in the workspace
cd "$ws"
exec "$@"
