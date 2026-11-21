"""Build the two sets of delegate units for particle-1 (free and paid): units it FIXED and units it did NOT, each with the prompt the delegate received (rebuilt from the task's initial state with the current
prompt builder: only the delegate-all arm, round 1, where the workspace was still the task's initial state), its raw answer, the reference method (the real repaired test) and the final method of the run.
Outcome per unit: fixed = the answer has a [REPAIR] block, passed the structural validation, and either equals the reference method (whitespace-insensitive) or survived unchanged in the final, passing workspace;
otherwise not fixed, with the reason (empty answer / no repair block / rejected by validation / applied but rewritten by the strong agent / differs from the reference).
Output: ~/delegation_exp/delegate_units/{fixed,not_fixed}.jsonl and summary.json (third-party code: kept out of git). usage: delegate_units.py"""
import json
import re
import shutil
import sys
from pathlib import Path

H = Path.home(); E = H / "delegation_exp"; OUT = E / "delegate_units"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(H / "PyTaRGET/scripts/delegation"))
from dtools import common, context, delegate  # noqa: E402

norm = lambda s: re.sub(r"\s+", "", s or "")


def method_text(path, name):
    try: text = Path(path).read_text(errors="ignore")
    except OSError: return None
    lines = text.split("\n")
    for n, a, b in common.find_methods(text):
        if n == name: return "\n".join(lines[a - 1:b])
    return None


# 1. runs: delegate-all (S1) runs whose round-1 rows were answered by particle-1
runs = []
for d in sorted((E / "runs").iterdir()):
    f = d / "ws/.delegation/delegate_calls.jsonl"
    if "deleg-S1" not in d.name or not f.exists(): continue
    rows = [json.loads(l) for l in open(f)]; r1 = [r for r in rows if r["round"] == 1 and str(r.get("model", "")).startswith("particle")]
    if r1: runs.append((d, r1))
print(len(runs), "S1 runs with particle-1 round-1 units")
res = json.loads((E / "batch_results.json").read_text()) if (E / "batch_results.json").exists() else {}

# 2. rebuild the initial-state prompts per task
cache = {}
def task_state(task):
    if task in cache: return cache[task]
    ws = E / "scratch" / "units_ws" / task; shutil.rmtree(ws, ignore_errors=True); shutil.copytree(E / "tasks" / task / "ws", ws, symlinks=True)
    cfg = common.load_config(ws); rc, log = common.mvn(ws, timeout=3000); kind, data = common.result_of(ws, rc, log)
    units = {u["name"]: u for u in context.units_from_state(ws, kind, data, cfg)}
    hunks = context.parse_hunks((ws / "SOURCE_CHANGES.diff").read_text()) if (ws / "SOURCE_CHANGES.diff").exists() else []
    cache[task] = (ws, log, units, hunks); return cache[task]

fixed, notfixed = [], []
for d, rows in runs:
    task = d.name.split("__")[0]
    if not (E / "tasks" / task / "ws").exists(): continue
    ws0, log, units, hunks = task_state(task)
    rr = res.get(d.name); final_ok = (rr["failing"] == 0 and bool(rr["pass"])) if rr else True      # pilot runs (not in batch_results): their agents reported a passing suite
    for r in rows:
        u = units.get(r["unit"])
        if not u: continue
        prompt = context.build_prompt(ws0, u, log, hunks, "auto", None)
        ans_f = d / "ws/.delegation/backup/round1" / (u["name"].replace("#", "_").replace(" ", "_") + ".answer.txt")
        ans = ans_f.read_text() if ans_f.exists() else None
        repair = delegate.extract(ans, "REPAIR") if ans else None
        old_text = context.unit_text(ws0, u)
        if u["kind"] == "method":
            ref_m = method_text(E / "tasks" / task / "ref" / u["file"], u.get("method")); fin_m = method_text(d / "ws" / u["file"], u.get("method"))
        else:                                                           # whole-file (or header) unit: compare whole files
            rf, ff = E / "tasks" / task / "ref" / u["file"], d / "ws" / u["file"]
            ref_m = rf.read_text(errors="ignore") if rf.exists() else None; fin_m = ff.read_text(errors="ignore") if ff.exists() else None
        reason = None
        if ans is None or not ans.strip(): reason = "empty answer"
        elif repair is None: reason = "no [REPAIR] block"
        else:
            err = delegate.validate(u, repair, old_text)
            if err: reason = err
            elif not r["applied"]: reason = "rejected by the apply step"
        matches_ref = bool(repair and ref_m and norm(repair) == norm(ref_m))
        survived = bool(repair and fin_m and norm(repair) == norm(fin_m))
        if reason is None and not (matches_ref or (survived and final_ok)): reason = "applied but rewritten by the strong agent" if not survived else "applied, survived, but the run did not end clean"
        if reason is None and not matches_ref and ref_m: pass
        rec = {"id": f"{d.name}::{u['name']}", "task": task, "run": d.name, "model": r["model"], "unit": u["name"], "kind": u["kind"], "file": u["file"], "prompt": prompt, "answer": ans, "repair": repair,
               "reference_method": ref_m, "final_method": fin_m, "matches_reference": matches_ref, "survived_in_final": survived, "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
               "features": {"prompt_chars": len(prompt), "error_lines": len(u["errors"]), "method_lines": len(old_text.split("\n")), "asserts_old": common.count_asserts(old_text), "kind_of_failure": "compile" if u["errors"] and u["errors"][0][0] else "runtime"},
               "outcome": "fixed" if reason is None else "not_fixed", "reason": reason}
        (fixed if reason is None else notfixed).append(rec)

for name, recs in (("fixed", fixed), ("not_fixed", notfixed)):
    with open(OUT / f"{name}.jsonl", "w") as fh:
        for rec in recs: fh.write(json.dumps(rec) + "\n")
summary = {"units": len(fixed) + len(notfixed), "fixed": len(fixed), "not_fixed": len(notfixed)}
print(summary); (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
