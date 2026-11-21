"""Aggregate ~/delegation_exp/batch_results.json by repo and arm (valid runs only: delegate was available; a run where the delegate answered part of the units counts)."""
import json
import statistics as st
from pathlib import Path

res = json.loads((Path.home() / "delegation_exp/batch_results.json").read_text())
groups = {}
oracle = {}
for run, r in res.items():
    if r.get("delegate_model") == "oracle": oracle[run] = r; continue      # ideal-scenario runs (the delegate inserts the reference repair) are reported separately
    task, arm = run.split("__")[0], ("solo" if "__solo__" in run else "S4" if "S4" in run else "S1")
    valid = not (r.get("delegate_unavailable") and r["delegate_applied"] == 0 and arm != "solo")
    groups.setdefault((task, arm), []).append((run, r, valid))
print("| repo | arm | valid runs | success | output tokens (mean) | model calls (mean) | added context (mean) | work (mean) | delegate $ (mean) |\n|---|---|---|---|---|---|---|---|---|")
tot = {}
for (task, arm), rows in sorted(groups.items()):
    v = [r for _, r, ok in rows if ok]
    if not v: print(f"| {task} | {arm} | 0 of {len(rows)} | - | - | - | - | - | - |"); continue
    m = lambda k: round(st.mean(x[k] for x in v))
    ok = sum(1 for x in v if x["success"]); print(f"| {task} | {arm} | {len(v)} (of {len(rows)}) | {ok}/{len(v)} | {m('out')} | {st.mean(x['calls'] for x in v):.1f} | {m('added_ctx')} | {m('work')} | {st.mean(x['delegate_cost'] for x in v):.4f} |")
    for k in ("out", "calls", "added_ctx", "work"): tot.setdefault(arm, {}).setdefault(k, []).extend(x[k] for x in v)
print("\nAll repos pooled (valid runs):")
for arm, d in tot.items(): print(f"  {arm}: n={len(d['out'])} out mean {st.mean(d['out']):.0f}, calls {st.mean(d['calls']):.1f}, added ctx {st.mean(d['added_ctx']):.0f}, work {st.mean(d['work']):.0f}")

if oracle:
    print("\nORACLE delegate runs (ideal scenario: the delegate inserts the reference repair; solo mean for comparison):")
    solo = {}
    for run, r in res.items():
        if "__solo__" in run: solo.setdefault(run.split("__")[0], []).append(r)
    for run, r in sorted(oracle.items()):
        t = run.split("__")[0]; sm = lambda k: st.mean(x[k] for x in solo[t]) if t in solo else float("nan")
        print(f"  {run}: ok={r['success']} calls {r['calls']} (solo {sm('calls'):.1f}) out {r['out']} (solo {sm('out'):.0f}) work {r['work']} (solo {sm('work'):.0f}) added {r['added_ctx']} (solo {sm('added_ctx'):.0f}) delegate units {r['delegate_applied']}/{r['delegate_units']}")
