"""Where the strong arm's tokens go, by phase, for delegation runs (reads sub-agent transcripts via agent_map.json).
Per model call: output tokens (that call), and the input added by the tool result that call triggered (ctx[i+1]-ctx[i]; the last call has no result). Phases are inferred from the first tool command of the call.
usage: routing_breakdown.py [run-name-substring ...]   (default: all S4 / S1 / solo runs of tasks with a score)"""
import glob, json, os, re, sys, collections
T = glob.glob("/tmp/claude-*/-gpfs-helios-home-kandrei/*/tasks/")
M = json.load(open(os.path.expanduser("~/delegation_exp/agent_map.json")))
R = json.load(open(os.path.expanduser("~/delegation_exp/batch_results.json")))

def phase(name, cmd):
    c = re.sub(r"^cd \S+ && ", "", cmd or "")
    if name == "SubagentHandback": return "final report"
    if name == "ToolSearch" or re.search(r"^(until|while|for i in|sleep)\b|/tmp/claude-\S*tasks/", c) or (name in ("Read", "Bash") and "/tasks/" in c): return "wait/poll for delegate"
    if name == "Read" and "prompt.md" in c: return "read prompt"
    if name in ("Edit", "Write"): return "hand edit"
    if re.match(r"^(tools/)?delegate|^tools/delegate", c): return "delegate call"
    if re.search(r"tools/review", c): return "review (+tests)"
    if re.search(r"\bsed -i|\bperl -pi|xargs sed|python3? - |python3? -c", c): return "sed / scripted edit"
    if re.search(r"SOURCE_CHANGES", c) or re.match(r"^(ls|git )", c): return "orient (ls/diff/tests)"
    if re.match(r"^tools/run_tests", c): return "run tests"
    if name in ("Read", "Grep", "Glob") or re.search(r"\b(cat|sed -n|head|tail|grep|awk)\b", c): return "read code"
    return "other"

def load(path):
    rows, res, order = {}, {}, []
    for l in open(path):
        try: d = json.loads(l)
        except ValueError: continue
        m = d.get("message") or {}; c = m.get("content")
        if d.get("type") == "assistant" and m.get("usage"):
            k = m.get("id") or d.get("uuid")
            if k not in rows: order.append(k); rows[k] = {"u": m["usage"], "acts": []}
            rows[k]["u"] = m["usage"]
            for b in (c if isinstance(c, list) else []):
                if b.get("type") == "tool_use":
                    i = b.get("input", {}); rows[k]["acts"].append((b.get("name"), str(i.get("command") or i.get("file_path") or i.get("pattern") or i), len(json.dumps(i))))
    out = []
    for k in order:
        u = rows[k]["u"]; ctx = sum((u.get(x, 0) or 0) for x in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        a = rows[k]["acts"]; out.append({"ctx": ctx, "out": u.get("output_tokens", 0) or 0, "act": a})
    return out

def breakdown(run):
    f = None
    for t in T:
        p = t + M[run] + ".output"
        if os.path.exists(p): f = p; break
    if not f: return None
    calls = load(f); base = calls[0]["ctx"]; ph = collections.defaultdict(lambda: [0, 0, 0, 0])   # calls, out, added_input, work(ctx re-read share)
    for i, c in enumerate(calls):
        a = c["act"]; name, cmd = (a[0][0], a[0][1]) if a else ("text", "")
        p = phase(name, cmd) if a else "text / thinking only"
        nxt = calls[i + 1]["ctx"] - c["ctx"] if i + 1 < len(calls) else 0
        ph[p][0] += 1; ph[p][1] += c["out"]; ph[p][2] += max(nxt, 0); ph[p][3] += c["ctx"] - base
    return ph, calls

if __name__ == "__main__":
    subs = sys.argv[1:]
    runs = [r for r in M if r in R or True]
    runs = [r for r in runs if (not subs or any(s in r for s in subs))]
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0])); n = collections.Counter()
    for r in runs:
        arm = "solo" if "__solo__" in r else ("S4" if "S4" in r else "S1" if "S1" in r else None)
        if not arm or "oracle" in r: continue
        b = breakdown(r)
        if not b: continue
        n[arm] += 1
        for p, v in b[0].items():
            for j in range(4): agg[arm][p][j] += v[j]
    for arm in ("solo", "S4", "S1"):
        if not n[arm]: continue
        tot = [sum(v[j] for v in agg[arm].values()) for j in range(4)]
        print(f"\n### {arm} ({n[arm]} runs with transcripts); per-run means; fixed start-up context excluded")
        print("| phase | calls | output tok | % out | input added by result | % in | work share (extra ctx re-read) | % work |\n|---|---|---|---|---|---|---|---|")
        for p, v in sorted(agg[arm].items(), key=lambda x: -x[1][1]):
            print(f"| {p} | {v[0]/n[arm]:.1f} | {v[1]/n[arm]:.0f} | {100*v[1]/tot[1]:.0f}% | {v[2]/n[arm]:.0f} | {100*v[2]/max(tot[2],1):.0f}% | {v[3]/n[arm]:.0f} | {100*v[3]/max(tot[3],1):.0f}% |")
        print(f"| **total** | {tot[0]/n[arm]:.1f} | {tot[1]/n[arm]:.0f} | | {tot[2]/n[arm]:.0f} | | {tot[3]/n[arm]:.0f} | |")
