"""Events for the OpenRouter contrast300 runs: one line per 50 new cases of a model, a stall warning (no new case for 15 min
while its process is alive), and a line when a model's process has ended. Exits when all three runs have ended."""
import json, subprocess, sys, time
from pathlib import Path
B = Path.home() / "eval_openrouter"
RUNS = {"ernie-flash": ("contrast300_baidu_ernie-5.5-flash-preview_free", "baidu/ernie"),
        "particle-1": ("contrast300_particle-ai_particle-1_free", "particle-ai/particle-1"),
        "nemotron-ultra": ("contrast300_nvidia_nemotron-3-ultra-550b-a55b_free", "nvidia/nemotron-3-ultra")}
STATE = Path(sys.argv[1]); TOTAL = 300
state = json.loads(STATE.read_text()) if STATE.exists() else {}

def alive(pattern):
    return subprocess.run(["pgrep", "-f", f"openrouter_run --model {pattern}"], capture_output=True).returncode == 0

def read(d):
    try:
        p = json.load(open(B / d / "test_predictions.json"))
        ids = p["ID"] if isinstance(p, dict) else [r["ID"] for r in p]
        empty = sum(1 for k in ids if not (p["preds"][k][0] if isinstance(p, dict) else "").strip()) if isinstance(p, dict) else 0
        return len(ids), empty
    except Exception:
        return 0, 0

def errors(d):
    try:
        return sum(1 for _ in open(B / d / "requests.jsonl"))
    except Exception:
        return 0

started = {k: state.get(k, {}).get("started", False) for k in RUNS}
while True:
    for name, (d, pat) in RUNS.items():
        s = state.setdefault(name, {"last_n": 0, "last_milestone": 0, "last_change": time.time(), "ended": False, "started": False, "stalled": False})
        a = alive(pat)
        n, empty = read(d)
        if a: s["started"] = True
        if n != s["last_n"]:
            s["last_n"], s["last_change"], s["stalled"] = n, time.time(), False
        milestone = n // 50
        if milestone > s["last_milestone"] and n < TOTAL:
            s["last_milestone"] = milestone
            print(f"PROGRESS {name}: {n}/{TOTAL} cases, {empty} empty answers, {errors(d)} requests logged", flush=True)
        if a and not s["stalled"] and time.time() - s["last_change"] > 900:
            s["stalled"] = True
            print(f"STALL {name}: no new case for 15 min while its process is alive (n={n})", flush=True)
        if s["started"] and not a and not s["ended"]:
            s["ended"] = True
            print(f"ENDED {name}: process finished with {n}/{TOTAL} cases, {empty} empty answers", flush=True)
    STATE.write_text(json.dumps(state))
    if all(state.get(k, {}).get("ended") for k in RUNS):
        print("ALL RUNS ENDED", flush=True)
        break
    time.sleep(30)
