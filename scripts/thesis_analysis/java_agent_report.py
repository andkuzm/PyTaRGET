"""Java: agents (mini-SWE-agent + Qwen3-Coder, offline Maven; 40-call and 80-call limits) against the prompt-only models of the contrast subset on the SAME cases.
Checked plausible@1 per group, weighted with the population sizes; split by clean cases (the repair changed no other test lines) and cases with other repaired tests.
Agent 'strict' = the final METHOD replaces the whole after-method (edits outside the method / in other tracked files = failure); 'relaxed' = the whole final test file is
the candidate (only changes of other tracked files are failures). Cases whose reference does not pass in our environment are excluded everywhere.
Run: PYTHONPATH=. python scripts/thesis_analysis/java_agent_report.py"""
import json, re, subprocess
from pathlib import Path
from data_processing import prediction_files
H = Path.home(); R = H / "eval_results_java_llm"
union = json.load(open(H / "eval_java_plaus/union.json")); span = json.load(open(H / "eval_java_plaus/span.json"))
m300 = json.load(open(R / "contrast300/subset_meta.json")); m273 = json.load(open(R / "contrast273/subset_meta.json"))["cases"]
group = {int(i): v["group"] for i, v in m300["cases"].items()}; pop = m300["population"]
rows = json.load(open(R / "contrast300/splits/test.json"))
manifest = json.load(open(H / "agent_java/manifest.json")); warm = json.load(open(H / "agent_java/warm.json"))
ASSERT = re.compile(r"\bassert\w*|\bverify\w*\s*\(|\bfail\s*\(|\bexpect\w*|expected\s*=|\.should|\bthrown\b|\bisEqualTo\b")
ids = [int(i) for i, e in manifest.items() if e["status"] == "ok" and warm.get(i) == "success"]

def agent_columns(name, run_dir, ws_is_current=True, manifest=None):
    """(strict, relaxed): {case: (valid, plain, checked)} for one agent run."""
    manifest = manifest or globals()["manifest"]
    scored = json.load(open(H / f"eval_agent_plaus/{name}.json")); skipped = json.load(open(H / f"eval_agent_plaus/{name}.json.skipped.json"))
    relaxed_res = json.load(open(H / f"eval_agent_plaus/{name}_relaxed.json"))
    strict, relaxed = {}, {}
    for i in ids:
        k = str(i)
        needs = span.get(k, {}).get("span_removed") != "success"
        # strict
        if k in skipped: strict[i] = (True, False, False)
        else:
            r = scored.get(k)
            if not r or r.get("status") != "evaluated" or not r.get("verdicts"): strict[i] = (False, False, False)
            else:
                plain = r["verdicts"][0]["status"] == "success"
                cand = json.load(open(H / f"eval_agent/{run_dir}/{k}/result.json"))["method"]
                strict[i] = (True, plain, plain and (needs or len(ASSERT.findall(cand)) >= len(ASSERT.findall(rows[i]["aSource"]["code"]))))
        # relaxed
        r = relaxed_res.get(k)
        if r is None: relaxed[i] = (True, False, False) if skipped.get(k) == "changed other tracked files" else (False, False, False); continue
        if r.get("status") != "evaluated" or not r.get("verdicts"): relaxed[i] = (False, False, False); continue
        plain = r["verdicts"][0]["status"] == "success"
        ws, row = Path(manifest[k]["ws"]), rows[i]
        if not ws.exists():                          # the workspaces were deleted after scoring: plain pass (it equalled the checked count when they existed: 42/59 and 35/59)
            relaxed[i] = (True, plain, plain); continue
        snap = subprocess.run(["git", "show", f"HEAD:{row['bPath']}"], cwd=ws, capture_output=True, text=True).stdout
        final = (ws / row["bPath"]).read_text(encoding="utf-8", errors="ignore")
        ref_asserts = len(ASSERT.findall(snap)) - len(ASSERT.findall(row["bSource"]["code"])) + len(ASSERT.findall(row["aSource"]["code"]))
        # the assertion check reads the final file from the workspace, which only holds the LATEST run: for an earlier run use the plain pass (it equalled the checked count)
        relaxed[i] = (True, plain, plain and (needs or len(ASSERT.findall(final)) >= ref_asserts)) if ws_is_current else (True, plain, plain)
    return strict, relaxed

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
EA = H / "eval_ablation_java"
for label, path in (("qwen3coder BC (context)", EA / "qwen3coder_BC"), ("particle-1 BC (context)", EA / "particle1_BC")):
    if (path / "test_predictions.json").exists():
        cols[label] = column(path, 273)
for tag, name, run_dir in (("40", "java_qwen3coder", "java_qwen3coder"), ("80", "java_qwen3coder_80", "java_qwen3coder_80")):
    if (H / f"eval_agent_plaus/{name}_relaxed.json").exists():
        s, r = agent_columns(name, run_dir, ws_is_current=(tag == "80"))
        cols[f"AGENT {tag} strict"], cols[f"AGENT {tag} relaxed"] = s, r
# Inkling-2-Mini: the agent (80 calls, workspaces rebuilt in ~/agent_java_ink) and the single prompt (effort low)
ink_manifest = json.load(open(H / "agent_java_ink/manifest.json")); ink_warm = json.load(open(H / "agent_java_ink/warm.json"))
ink_ids = [int(i) for i, e in ink_manifest.items() if e["status"] == "ok" and ink_warm.get(i) == "success"]
print(f"inkling agent cases {len(ink_ids)}; same ids as the Qwen agent preparation: {len(set(ink_ids) & set(ids))}")
if (H / "eval_agent_plaus/java_inkling_low_80_relaxed.json").exists():
    qids = ids; ids = [i for i in qids if i in set(ink_ids)]
    s, r = agent_columns("java_inkling_low_80", "java_inkling_low_80", ws_is_current=True, manifest=ink_manifest)
    cols["INK AGENT strict"], cols["INK AGENT relaxed"] = s, r
cols["inkling low (1 prompt)"] = {}
_u = json.load(open(H / "eval_inkling/java273_low/java_plausibility.json")); _p = prediction_files.load(H / "eval_inkling/java273_low/test_predictions.json")
for q in _p:
    pos = int(q["ID"]); i = int(m273[str(pos)]["index300"]); res = _u.get(str(pos), {}); c = q["preds"][0].strip()
    if res.get("status") != "evaluated": continue
    plain = res["verdicts"][0]["status"] == "success"
    cols["inkling low (1 prompt)"][i] = (True, plain, plain and (span.get(str(i), {}).get("span_removed") != "success" or bool(ASSERT.search(c))))
common = [i for i in ids if all(i in c and c[i][0] for c in cols.values())]
clean = [i for i in common if manifest[str(i)]["other_test_lines_changed"] == 0]
print(f"cases {len(ids)}; common to all columns {len(common)}; clean (no other test lines changed) {len(clean)}")
def rates(sel, f):
    out = {}
    for g in ("solved_at_1", "not_solved_at_1"):
        s = [i for i in sel if group[i] == g]; out[g] = (sum(f(i) for i in s), len(s))
    a, b = (out[g][0] / out[g][1] if out[g][1] else float("nan") for g in out)
    return out, 100 * a, 100 * b, 100 * (pop["solved_at_1"] * a + pop["not_solved_at_1"] * b) / (pop["solved_at_1"] + pop["not_solved_at_1"])
print(f"\n{'column':20s} {'solved':>7s} {'unsolved':>9s} {'weighted':>9s} {'total':>8s}   (checked)")
for name, c in cols.items():
    o, s, u, w = rates(common, lambda i, c=c: c[i][2]); print(f"{name:20s} {s:7.1f} {u:9.1f} {w:9.1f} {sum(c[i][2] for i in common):4d}/{len(common)}")
print("\nSplit by whether the repair changed other test lines (checked passes):")
keys = [k for k in cols if k.startswith(("AGENT", "INK"))] + ["inkling low (1 prompt)", "qwen3coder", "particle-1", "nemotron-3-ultra"]
for name, sel in (("clean (0 other lines)", clean), ("other tests repaired", [i for i in common if i not in clean])):
    print(f"  {name:22s} n={len(sel):3d} | " + " | ".join(f"{k} {sum(cols[k][i][2] for i in sel)}/{len(sel)}" for k in keys))
def paired(a, b):
    return sum(1 for i in common if cols[a][i][2] and not cols[b][i][2]), sum(1 for i in common if cols[b][i][2] and not cols[a][i][2])
print()
for a, b in (("INK AGENT strict", "inkling low (1 prompt)"), ("INK AGENT relaxed", "inkling low (1 prompt)"), ("INK AGENT strict", "AGENT 80 strict"), ("INK AGENT relaxed", "AGENT 80 relaxed"), ("INK AGENT strict", "particle-1"),
             ("AGENT 80 relaxed", "particle-1 BC (context)"), ("AGENT 80 strict", "particle-1 BC (context)"), ("AGENT 80 relaxed", "qwen3coder BC (context)"), ("AGENT 80 strict", "qwen3coder BC (context)"),
             ("AGENT 40 strict", "particle-1"), ("AGENT 40 relaxed", "particle-1"), ("AGENT 80 strict", "particle-1"), ("AGENT 80 relaxed", "particle-1"),
             ("AGENT 80 strict", "AGENT 40 strict"), ("AGENT 80 relaxed", "AGENT 40 relaxed"), ("AGENT 80 relaxed", "qwen3coder")):
    if a in cols and b in cols:
        p, m = paired(a, b); print(f"paired {a} vs {b}: +{p} -{m}")
