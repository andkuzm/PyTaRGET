#!/usr/bin/bash
# Verified backup of the datasets and subsets (read-only for the originals): every file is gzip-compressed on its own, hashed (sha256 of the ORIGINAL), and
# verified by decompressing the archive and comparing the hash; small directories are tar.gz'ed. One MANIFEST.tsv lists name, original bytes, sha256, archive bytes,
# status. A backup on the same file system protects against accidental deletion, not against the loss of the file system.
# usage: backup_datasets.sh <backup dir> ; resumable (an archive that is already verified in MANIFEST.tsv is skipped)
set -uo pipefail
B="${1:?backup dir}"; mkdir -p "$B"; M="$B/MANIFEST.tsv"; touch "$M"; cd "$HOME"
nice -n 10 true
backup_file() {      # path relative to $HOME
  f="$1"; [ -f "$f" ] || { echo "missing $f"; return; }
  out="$B/$(echo "$f" | tr '/' '__').gz"
  grep -q "^$f	.*	OK$" "$M" && { echo "skip $f"; return; }
  echo "=== $f $(date +%T)"
  size=$(stat -c %s "$f"); sha=$(nice -n 10 sha256sum "$f" | cut -d' ' -f1)
  nice -n 10 gzip -6 -c "$f" > "$out.tmp" && back=$(nice -n 10 gzip -dc "$out.tmp" | sha256sum | cut -d' ' -f1)
  if [ "$sha" = "$back" ]; then mv "$out.tmp" "$out"; printf "%s\t%s\t%s\t%s\tOK\n" "$f" "$size" "$sha" "$(stat -c %s "$out")" >> "$M"; else rm -f "$out.tmp"; printf "%s\t%s\t%s\t0\tVERIFY_FAILED\n" "$f" "$size" "$sha" >> "$M"; echo "VERIFY FAILED $f"; fi
}
backup_dir() {       # directory relative to $HOME, tar.gz, verified by a full read and a listing compared with find
  d="$1"; [ -d "$d" ] || { echo "missing $d"; return; }
  out="$B/$(echo "$d" | tr '/' '__').tar.gz"
  grep -q "^$d/	.*	OK$" "$M" && { echo "skip $d"; return; }
  echo "=== $d $(date +%T)"
  nice -n 10 tar -czf "$out.tmp" "$d" 2>/dev/null; n_tar=$(tar -tzf "$out.tmp" | grep -vc '/$'); n_find=$(find "$d" -type f | wc -l)
  if [ "$n_tar" = "$n_find" ]; then mv "$out.tmp" "$out"; printf "%s/\t%s\t%s\t%s\tOK\n" "$d" "$n_find files" "-" "$(stat -c %s "$out")" >> "$M"; else rm -f "$out.tmp"; printf "%s/\t%s\t-\t0\tVERIFY_FAILED(tar %s vs find %s)\n" "$d" "$n_find files" "$n_tar" "$n_find" >> "$M"; echo "VERIFY FAILED $d"; fi
}
# 1) mined datasets, funnels, resume lists (irreplaceable: days of mining)
for f in annotated_cases_v2_pr59.csv annotated_cases_v2_pr59_filtered_cap_clean.csv annotated_cases_v2_pr59_filtered.csv annotated_cases_v1_pr57.csv \
         annotated_cases_trial4_pr58.csv annotated_cases_trial3_pr58.csv annotated_cases_trial2_pr58.csv annotated_cases_trial_pr58.csv annotated_cases_v2_pr59_smoke.csv \
         annotated_cases_v2_pr59_funnel.csv processed_repositories_v2_pr59.txt processed_repositories_v1_pr57.txt processed_repositories_trial4_pr58.txt \
         processed_repositories_trial3_pr58.txt processed_repositories_trial2_pr58.txt processed_repositories_trial_pr58.txt Untitled1.ipynb; do backup_file "$f"; done
for f in $(ls annotated_cases_*_funnel.csv 2>/dev/null); do backup_file "$f"; done
# 2) subsets and small result sets
for d in eval_results_java_llm eval_ablation eval_java_plaus eval_agent eval_ablation_plaus eval_agent_plaus eval_openrouter eval_openrouter_plaus eval_java_check references; do backup_dir "$d"; done
for f in eval_results_perproject_llm/contrast300/splits/test.json eval_results_perproject_llm/contrast300/subset_meta.json eval_results_perproject_llm/matched/splits/test.json; do backup_file "$f"; done
for f in $(cd "$HOME"; find eval_results_perproject_llm -maxdepth 3 -name "*.json" -size -3M 2>/dev/null); do backup_file "$f"; done
# 3) encoded splits (derived from the CSV, expensive to rebuild): per-project split first (the one the results use), then the global split and the full-context one
for base in eval_results_perproject eval_results eval_results_perproject_fullctx; do for f in $(find $base -path "*/splits/*.json" -size +0 2>/dev/null | sort); do backup_file "$f"; done; done
echo "=== all done $(date +%T)"; grep -c "OK$" "$M"; grep -c "FAILED" "$M" || true
