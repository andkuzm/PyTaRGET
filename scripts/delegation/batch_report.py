"""Score finished batch runs and compute the strong arm's work. usage: batch_report.py [<run id> <agent id> ...] (registers agent ids first), then prints and stores the table.
Registry: ~/delegation_exp/agent_map.json {run id: agent id}; results: ~/delegation_exp/batch_results.json {run id: {...}}; transcripts in the session's tasks directory."""
import json
import subprocess
import sys
from pathlib import Path

H = Path.home(); EXP = H / "delegation_exp"; T = Path("/tmp/claude-253910/-gpfs-helios-home-kandrei/55172aa0-a22c-4000-9663-4dc4dbc73303/tasks")
PY = str(H / "venv3.12/bin/python"); D = H / "PyTaRGET/scripts/delegation"
amap_f, res_f = EXP / "agent_map.json", EXP / "batch_results.json"
amap = json.loads(amap_f.read_text()) if amap_f.exists() else {}; res = json.loads(res_f.read_text()) if res_f.exists() else {}
args = sys.argv[1:]
for i in range(0, len(args) - 1, 2): amap[args[i]] = args[i + 1]
amap_f.write_text(json.dumps(amap, indent=1))


def handed_back(agent):
    for l in open(T / f"{agent}.output"):
        if '"SubagentHandback"' not in l: continue
        try: d = json.loads(l)
        except ValueError: continue
        c = (d.get("message") or {}).get("content")
        if d.get("type") == "assistant" and any(isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "SubagentHandback" for b in (c if isinstance(c, list) else [])): return True
    return False


def tokens(agent):
    seen = {}
    for l in open(T / f"{agent}.output"):
        try: d = json.loads(l)
        except ValueError: continue
        m = d.get("message") or {}
        if d.get("type") == "assistant" and m.get("usage"): seen[m.get("id") or d.get("uuid")] = m["usage"]
    u = list(seen.values()); ctx = [(x.get("input_tokens", 0) or 0) + (x.get("cache_creation_input_tokens", 0) or 0) + (x.get("cache_read_input_tokens", 0) or 0) for x in u]
    return {"calls": len(u), "added_ctx": ctx[-1] - ctx[0], "work": sum(c - ctx[0] for c in ctx), "out": sum(x.get("output_tokens", 0) or 0 for x in u)}


for run, agent in amap.items():
    if run in res or not (T / f"{agent}.output").exists(): continue
    task = run.split("__")[0]; ws = EXP / "runs" / run / "ws"
    if not ws.exists() or not handed_back(agent): continue      # only runs whose agent has made its final SubagentHandback call (finished)
    s = json.loads(subprocess.run([PY, str(D / "score_task.py"), str(ws), str(EXP / "tasks" / task)], capture_output=True, text=True).stdout or "{}")
    a = json.loads(subprocess.run([PY, str(D / "audit_transcript.py"), str(T / f"{agent}.output"), str(ws)], capture_output=True, text=True).stdout or "{}")
    dc = ws / ".delegation/delegate_calls.jsonl"; rows = [json.loads(x) for x in open(dc)] if dc.exists() else []
    res[run] = {"success": s.get("success"), "pass": s.get("tests_pass"), "total": s.get("tests_final"), "weakened": s.get("weakened_classes"), "shrunk": s.get("shrunk_classes"), "failing": len(s.get("failing", [])),
                "src_mod": s.get("source_modified"), "audit_flags": [f["what"][:60] for f in a.get("flags", []) if "tasks/" not in f["what"]], **tokens(agent),
                "delegate_model": json.loads((ws / ".delegation/config.json").read_text()).get("default_delegate"), "delegate_unavailable": "delegate: UNAVAILABLE" in (T / f"{agent}.output").read_text(), "delegate_units": len(rows), "delegate_applied": sum(r["applied"] for r in rows), "delegate_cost": round(sum(r["cost"] for r in rows), 4)}
    res_f.write_text(json.dumps(res, indent=1))
print(f"{'run':<44} ok   pass/total  calls  added   out   work  deleg(applied/units, $)")
for run, r in sorted(res.items()):
    print(f"{run:<44} {'yes' if r['success'] else 'NO ':<4} {r['pass']}/{r['total']:<6} {r['calls']:>5} {r['added_ctx']:>6} {r['out']:>5} {r['work']:>6}  {r['delegate_applied']}/{r['delegate_units']} ${r['delegate_cost']}" + ("  [DELEGATE UNAVAILABLE - not a valid delegation run]" if r.get("delegate_unavailable") else "") + (f"  weak={r['weakened']}" if r['weakened'] else "") + (f"  FLAGS={r['audit_flags']}" if r['audit_flags'] else ""))
