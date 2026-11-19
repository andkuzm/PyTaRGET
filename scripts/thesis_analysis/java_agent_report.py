"""Java: agent (mini-SWE-agent + Qwen3-Coder, offline Maven) against the prompt-only models of the contrast subset on the SAME cases. Checked plausible@1 per group,
weighted with the population sizes; split by clean cases (the repair changed no other test lines: nothing to copy) and cases with other repaired tests in the module
(the agent may copy their pattern). Agent failures by rule (no usable method, edit outside the method, other tracked files changed) count as failures; cases whose reference
does not pass in our environment are excluded everywhere.
Run (python -I is not possible, imports): PYTHONPATH=. python scripts/thesis_analysis/java_agent_report.py"""
import json, re
from pathlib import Path
from data_processing import prediction_files
H = Path.home(); R = H / "eval_results_java_llm"
union = json.load(open(H / "eval_java_plaus/union.json")); span = json.load(open(H / "eval_java_plaus/span.json"))
m300 = json.load(open(R / "contrast300/subset_meta.json")); m273 = json.load(open(R / "contrast273/subset_meta.json"))["cases"]
group = {int(i): v["group"] for i, v in m300["cases"].items()}; pop = m300["population"]
rows = json.load(open(R / "contrast300/splits/test.json"))
manifest = json.load(open(H / "agent_java/manifest.json")); warm = json.load(open(H / "agent_java/warm.json"))
scored = json.load(open(H / "eval_agent_plaus/java_qwen3coder.json")); skipped = json.load(open(H / "eval_agent_plaus/java_qwen3coder.json.skipped.json"))
ASSERT = re.compile(r"\bassert\w*|\bverify\w*\s*\(|\bfail\s*\(|\bexpect\w*|expected\s*=|\.should|\bthrown\b|\bisEqualTo\b")
ids = [int(i) for i, e in manifest.items() if e["status"] == "ok" and warm.get(i) == "success"]
# agent verdict per case: (valid, plain, checked)
agent = {}
for i in ids:
    k = str(i)
    if k in skipped: agent[i] = (True, False, False); continue
    r = scored.get(k)
    if not r or r.get("status") != "evaluated" or not r.get("verdicts"): agent[i] = (False, False, False); continue
    plain = r["verdicts"][0]["status"] == "success"
    cand = json.load(open(H / f"eval_agent/java_qwen3coder/{k}/result.json"))["method"]
    needs = span.get(k, {}).get("span_removed") != "success"
    agent[i] = (True, plain, plain and (needs or len(ASSERT.findall(cand)) >= len(ASSERT.findall(rows[i]["aSource"]["code"]))))
# relaxed agent: the whole final test file is the candidate (imports/helpers may change); other tracked files changed = failure
import subprocess
relaxed_res = json.load(open(H / "eval_agent_plaus/java_qwen3coder_relaxed.json"))
agent_relaxed = {}
for i in ids:
    k = str(i)
    r = relaxed_res.get(k)
    if r is None:
        agent_relaxed[i] = (True, False, False) if skipped.get(k) == "changed other tracked files" else (False, False, False); continue
    if r.get("status") != "evaluated" or not r.get("verdicts"): agent_relaxed[i] = (False, False, False); continue
    plain = r["verdicts"][0]["status"] == "success"
    ws, row = Path(manifest[k]["ws"]), rows[i]
    snap = subprocess.run(["git", "show", f"HEAD:{row['bPath']}"], cwd=ws, capture_output=True, text=True).stdout
    final = (ws / row["bPath"]).read_text(encoding="utf-8", errors="ignore")
    ref_asserts = len(ASSERT.findall(snap)) - len(ASSERT.findall(row["bSource"]["code"])) + len(ASSERT.findall(row["aSource"]["code"]))      # assertions of the after file
    needs = span.get(k, {}).get("span_removed") != "success"
    agent_relaxed[i] = (True, plain, plain and (needs or len(ASSERT.findall(final)) >= ref_asserts))
def column(path, space):
    d = prediction_files.load(path / "test_predictions.json"); out = {}
    for p in d:
        pos = int(p["ID"]); i = pos if space == 300 else int(m273[str(pos)]["index300"])
        c = p["preds"][0].strip(); rec = union.get(str(i), {})
        if rec.get("status") != "evaluated": continue
        plain = c != "" and rec["cache"].get(c, [None])[0] == "success"
        out[i] = (True, plain, plain and (span.get(str(i), {}).get("span_removed") != "success" or bool(ASSERT.search(c))))
    return out
cols = {"qwen3coder": column(R / "local/qwen3coder", 300), "gpt-oss-20b": column(R / "local/gptoss20b", 300),
        "ernie-5.5-flash": column(H / "eval_openrouter/java273_baidu_ernie-5.5-flash-preview_free", 273),
        "particle-1": column(H / "eval_openrouter/java273_particle-ai_particle-1_free", 273),
        "nemotron-3-ultra": column(H / "eval_openrouter/java273_nvidia_nemotron-3-ultra-550b-a55b_free", 273)}
cols["AGENT (qwen)"] = agent
cols["AGENT relaxed"] = agent_relaxed
common = [i for i in ids if all(i in c and c[i][0] for c in cols.values())]
clean = [i for i in common if manifest[str(i)]["other_test_lines_changed"] == 0]
print(f"agent cases {len(ids)}; common to all columns {len(common)}; clean (no other test lines changed) {len(clean)}; agent failures by rule {sum(1 for i in ids if str(i) in skipped)}")
def rates(sel, f):
    out = {}
    for g in ("solved_at_1", "not_solved_at_1"):
        s = [i for i in sel if group[i] == g]; out[g] = (sum(f(i) for i in s), len(s))
    a, b = (out[g][0] / out[g][1] if out[g][1] else float("nan") for g in out)
    return out, 100 * a, 100 * b, 100 * (pop["solved_at_1"] * a + pop["not_solved_at_1"] * b) / (pop["solved_at_1"] + pop["not_solved_at_1"])
print(f"\n{'column':18s} {'solved':>7s} {'unsolved':>9s} {'weighted':>9s} {'total':>8s}   (checked)")
for name, c in cols.items():
    o, s, u, w = rates(common, lambda i, c=c: c[i][2]); print(f"{name:18s} {s:7.1f} {u:9.1f} {w:9.1f} {sum(c[i][2] for i in common):4d}/{len(common)}")
o, s, u, w = rates(common, lambda i: agent[i][1]); print(f"{'agent plain':18s} {s:7.1f} {u:9.1f} {w:9.1f} {sum(agent[i][1] for i in common):4d}/{len(common)}")
print("\nSplit by whether the repair changed other test lines (agent could copy their pattern): checked passes")
for name, sel in (("clean (0 other lines)", clean), ("other tests repaired", [i for i in common if i not in clean])):
    print(f"  {name:22s} n={len(sel):3d} | " + " | ".join(f"{k} {sum(cols[k][i][2] for i in sel)}/{len(sel)}" for k in ("AGENT (qwen)", "AGENT relaxed", "qwen3coder", "particle-1", "nemotron-3-ultra")))
b = "particle-1"
for label in ("AGENT (qwen)", "AGENT relaxed"):
    plus = sum(1 for i in common if cols[label][i][2] and not cols[b][i][2]); minus = sum(1 for i in common if cols[b][i][2] and not cols[label][i][2])
    print(f"paired {label} vs {b}: +{plus} -{minus}")
