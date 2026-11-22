"""Replays literal sync + routing (dtools/route.py) on the stored particle-1 units (~/delegation_exp/delegate_units, outside git): how many units the rules would have kept from the delegate, whether those were the
failing ones, and how many would have been removed by literal sync. usage: replay_routing.py"""
import collections
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dtools import route  # noqa: E402

D = Path.home() / "delegation_exp/delegate_units"
rows = [json.loads(l) for f in ("fixed.jsonl", "not_fixed.jsonl") for l in open(D / f)]
sect = lambda p, a, b: (re.search(r"(?s)### " + a + r"[^\n]*\n(.*?)\n### " + b, p) or [0, ""])[1]
for r in rows:
    r["old"] = sect(r["prompt"], "Test Code", "Errors"); r["msg"] = sect(r["prompt"], "Errors", "Test Failure Output").strip().replace("\n", " | ")
    r["ok"] = r["outcome"] == "fixed"
M = [r for r in rows if r["kind"] == "method"]
# literal sync per run: substitutions shared by >= MIN_TESTS failing units
g = collections.defaultdict(set)
for r in M:
    ea = route.expected_actual(r["msg"]); s = ea and route.substitution(*ea)
    if s: g[(r["run"], s)].add(r["unit"])
synced = {(run, u) for (run, s), us in g.items() if len(us) >= route.MIN_TESTS for u in us}
print(f"method units {len(M)}; literal sync would resolve {sum(1 for r in M if (r['run'], r['unit']) in synced)} of them without a model; substitutions: " + "; ".join(sorted({f"{s[0]!r}->{s[1]!r}" for (run, s), us in g.items() if len(us) >= route.MIN_TESTS})[:4]))
tab = collections.Counter()
for r in M:
    sy = (r["run"], r["unit"]) in synced
    why = None if sy else route.route({"kind": "method"}, r["old"], r["msg"])
    cls = "literal sync" if sy else ("routed to strong" if why else "to delegate")
    tab[(cls, "fixed" if r["ok"] else "not fixed")] += 1
print("\n| decision | units | fixed by particle-1 anyway | not fixed |\n|---|---|---|---|")
for c in ("literal sync", "routed to strong", "to delegate"):
    a, b = tab[(c, "fixed")], tab[(c, "not fixed")]; print(f"| {c} | {a+b} | {a} | {b} |")
d = [r for r in M if (r["run"], r["unit"]) not in synced and not route.route({"kind": "method"}, r["old"], r["msg"])]
print(f"\nsent to the delegate: {len(d)} units, fixed {sum(r['ok'] for r in d)} ({100*sum(r['ok'] for r in d)/len(d):.0f}%) vs {sum(r['ok'] for r in M)}/{len(M)} ({100*sum(r['ok'] for r in M)/len(M):.0f}%) without routing")
print("remaining misses among delegated:", collections.Counter((r["task"], r["reason"][:28]) for r in d if not r["ok"]).most_common(6))
print("routed units that particle-1 did fix:", [(r["unit"][-45:], r["msg"][:70]) for r in M if route.route({"kind": "method"}, r["old"], r["msg"]) and r["ok"] and (r["run"], r["unit"]) not in synced][:5])
