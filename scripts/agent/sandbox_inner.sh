#!/usr/bin/bash
# Runs inside `unshare -m` (own mount namespace, inside the job's private network namespace). Hides the home directory and the shared file system, then
# makes visible ONLY: the case workspace (read/write), the agent's output directory (read/write) and the agent virtualenv (read-only is not enforced).
# usage: sandbox_inner.sh <workspace> <outdir> <extra visible path>... -- <command...>
set -euo pipefail
ws="$1"; out="$2"; shift 2
extra=(); while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
real=$(readlink -f "$HOME/venv_agent/bin/python")                                    # the interpreter of the venv (cluster software, not user data)
case "$real" in /gpfs/space/software/*) pybase=/gpfs/space/software;; *) pybase=$(dirname "$(dirname "$real")");; esac
[ -n "${SANDBOX_NARROW:-}" ] && pybase=$(dirname "$(dirname "$real")")               # narrow mode (fix D1): only the install folder; the caller passes any further folder needed (runtime_paths.py)
keep=("$ws" "$out" "$HOME/venv_agent" "$pybase" "${extra[@]}")
if [ -n "${SANDBOX_NARROW:-}" ]; then mount -t tmpfs tmpfs /mnt; tmp=/mnt          # staging area: a private mount, so nothing is left behind on the node
else tmp=$(mktemp -d); fi                          # staging area outside the hidden trees
for p in "${keep[@]}"; do mkdir -p "$tmp$p"; mount --rbind "$p" "$tmp$p"; done
mount -t tmpfs tmpfs /gpfs                        # hide the whole shared file system (home, other users, datasets)
if [ -n "${SANDBOX_NARROW:-}" ]; then              # the node's scratch folders can hold other jobs' and other users' files: give the sandbox empty private ones
  for d in /tmp /var/tmp /dev/shm; do if [ -d "$d" ]; then mount -t tmpfs -o mode=1777 tmpfs "$d"; fi; done
fi
for p in "${keep[@]}"; do mkdir -p "$p"; mount --rbind "$tmp$p" "$p"; done
umount -l "$tmp" 2>/dev/null || true
export HOME="$out/home"; mkdir -p "$HOME"          # the agent's home is in its output directory, not in the workspace
cd "$ws"
exec "$@"
