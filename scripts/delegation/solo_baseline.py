"""Solo strong-arm baseline per task: tokens per run (batch_results.json for the repo-set/real runs, plus the earlier pilot runs whose numbers are in the docs) and size features of the real repair
(reference tests vs old tests: changed files, changed test methods, changed lines, source diff size), so that future solo costs can be estimated instead of re-measured. usage: solo_baseline.py"""
import difflib
import json
import statistics as st
import sys
from pathlib import Path

H = Path.home(); E = H / "delegation_exp"
sys.path.insert(0, str(H / "PyTaRGET/scripts/delegation"))
from dtools import common  # noqa: E402

# earlier solo pilot runs (their transcripts are gone; numbers copied from docs/results/delegation_strong_arm_work.md and delegation_ledger_work.md): task, calls, added ctx, work, output tokens
PILOT = {"shop": [(4, 15053, 36975, 3204), (4, 14919, 36631, 3244)], "gson_a80": [(4, 9564, 22979, 804)], "gson_cbad": [(6, 21972, 84716, 1990)],
         "ledger_dev": [(4, 10961, 22797, 3864), (7, 12695, 56732, 4523)]}
runs = {t: [(v[0], v[1], v[2], v[3]) for v in vs] for t, vs in PILOT.items()}
res = json.loads((E / "batch_results.json").read_text())
for run, r in res.items():
    if "__solo__" in run: runs.setdefault(run.split("__")[0], []).append((r["calls"], r["added_ctx"], r["work"], r["out"]))


def methods(text):
    lines = text.split("\n"); return {n: "\n".join(lines[a - 1:b]) for n, a, b in common.find_methods(text)}


def features(task):
    d = E / "tasks" / task
    if not (d / "ref").exists(): return None
    old = {p.relative_to(d / "old_tests"): p.read_text(errors="ignore") for p in (d / "old_tests").rglob("*.java")}
    ref = {p.relative_to(d / "ref"): p.read_text(errors="ignore") for p in (d / "ref").rglob("*.java")}
    files = meths = lines = tests = 0
    for k, new in ref.items():
        o = old.get(k, ""); tests += len(methods(new))
        if o == new: continue
        files += 1; om, nm = methods(o), methods(new); meths += sum(1 for n, b in nm.items() if om.get(n) != b)
        lines += sum(1 for l in difflib.unified_diff(o.split("\n"), new.split("\n"), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return {"files": files, "methods": meths, "lines": lines, "diff_chars": meta.get("diff_chars"), "tests_in_changed_files": tests}


rows = []
for task, rs in sorted(runs.items()):
    f = features(task) or {}
    m = lambda i: round(st.mean(r[i] for r in rs))
    rows.append({"task": task, "n": len(rs), "calls": round(st.mean(r[0] for r in rs), 1), "added": m(1), "work": m(2), "out": m(3), **f})
print("| task | solo runs | calls | output tokens | added context | work | changed files | changed test methods | changed lines | source diff chars | out tokens per changed file | per changed method | per 100 changed lines |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    pf = f"{r['out'] / r['files']:.0f}" if r.get("files") else "-"; pm = f"{r['out'] / r['methods']:.1f}" if r.get("methods") else "-"; pl = f"{100 * r['out'] / r['lines']:.0f}" if r.get("lines") else "-"
    print(f"| {r['task']} | {r['n']} | {r['calls']} | {r['out']} | {r['added']} | {r['work']} | {r.get('files', '-')} | {r.get('methods', '-')} | {r.get('lines', '-')} | {r.get('diff_chars', '-')} | {pf} | {pm} | {pl} |")
xs = [r for r in rows if r.get("lines")]
if len(xs) > 3:
    def corr(a, b):
        ma, mb = st.mean(a), st.mean(b); num = sum((x - ma) * (y - mb) for x, y in zip(a, b)); den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den if den else float("nan")
    for k in ("lines", "methods", "files", "diff_chars"):
        ys = [r for r in xs if r.get(k)]; print(f"\ncorrelation of solo output tokens with changed {k}: r = {corr([r[k] for r in ys], [r['out'] for r in ys]):.2f} (n={len(ys)})", end="")
    print()
print("\nall solo runs pooled: n=%d, output tokens mean %.0f (median %.0f), calls mean %.1f" % (sum(r["n"] for r in rows), st.mean(x[3] for rs in runs.values() for x in rs), st.median(x[3] for rs in runs.values() for x in rs), st.mean(x[0] for rs in runs.values() for x in rs)))
