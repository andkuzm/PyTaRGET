"""Delegation experiment launcher (state: ~/delegation_exp/state.json, runs: ~/delegation_exp/runs/<run id>/).
  launcher.py new-run <task> solo|delegate <rep> [--strategy S1|S2|S3|S4|S5] [--handoff <file>] [--context auto|strong|both] [--verify exec|diff] [--delegate-model particle-free]
        -> copies the task workspace, removes the delegate tool for the solo arm, writes prompt.md and prints its path (you give it to a fresh Claude Code sub-agent)
  launcher.py finish-run <run id> --tokens N --tool-uses N --duration-s S [--percent-before a --percent-after b]
        -> scores the workspace (score_task.py), summarises the delegate calls, updates the state
  launcher.py status"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

H = Path.home(); EXP = H / "delegation_exp"; HERE = Path(__file__).resolve().parent
PROMPTS = H / "PyTaRGET" / "docs" / "delegation_prompts"


def state():
    p = EXP / "state.json"; s = json.loads(p.read_text()) if p.exists() else {}
    s.setdefault("runs", {}); return s


def save(s): (EXP / "state.json").write_text(json.dumps(s, indent=1))


def prompt_for(ws, arm, strategy, context, verify, handoff=None):
    t = lambda n: (PROMPTS / n).read_text().strip()
    parts = [t("run_common.md").replace("{WS}", str(ws))]
    if handoff: parts.append(t("handoff.md").replace("{HANDOFF}", Path(handoff).read_text().strip()))
    if arm == "solo": parts.append(t("arm_solo.md"))
    else: parts += [t("arm_delegate_intro.md"), t(f"strategy_{strategy}.md"), t(f"context_{context}.md"), t(f"verify_{verify}.md")]
    return "\n\n".join(parts) + "\n"


def new_run(a):
    task = EXP / "tasks" / a.task
    rid = f"{a.task}__{'solo' if a.arm == 'solo' else f'deleg-{a.strategy}-{a.context}-{a.verify}'}__r{a.rep}"
    run = EXP / "runs" / rid
    if run.exists(): sys.exit(f"{rid} exists")
    ws = run / "ws"; shutil.copytree(task / "ws", ws, symlinks=True)
    cfg = json.loads((ws / ".delegation/config.json").read_text())
    for name, script in (("start", "start.py"), ("delegate", "delegate_guard.py")):      # automation tools (see docs/results/delegation_routing_breakdown.md); delegate_guard never runs longer than 160 s per call
        f = ws / "tools" / name
        f.write_text('#!/usr/bin/bash\nexec %s %s "$(cd "$(dirname "$0")/.." && pwd)" "$@"\n' % (H / "venv3.12/bin/python", HERE / "dtools" / script)); f.chmod(0o755)
    if a.arm == "solo":
        (ws / "tools" / "delegate").unlink(); cfg.pop("delegates", None)
    else:
        cfg["default_delegate"] = a.delegate_model
        cfg["delegates"]["oracle"] = {"oracle": True, "ref_dir": str(EXP / "tasks" / a.task / "ref")}     # ideal-scenario experiment only (--delegate-model oracle)
        cfg["fallback"] = {"particle-free": "particle"}; cfg["paid_cap_usd"] = 0.30; cfg["paid_ledger"] = str(EXP / "delegate_paid_ledger.json")
    (ws / ".delegation/config.json").write_text(json.dumps(cfg, indent=1))
    (run / "prompt.md").write_text(prompt_for(ws, a.arm, a.strategy, a.context, a.verify, a.handoff))
    s = state(); s["runs"][rid] = {"task": a.task, "arm": a.arm, "strategy": a.strategy if a.arm != "solo" else None, "context": a.context if a.arm != "solo" else None,
                                   "verify": a.verify if a.arm != "solo" else None, "delegate_model": a.delegate_model if a.arm != "solo" else None, "rep": a.rep, "cell": a.cell, "status": "prepared", "created": time.strftime("%F %H:%M")}
    save(s); print(rid); print("prompt:", run / "prompt.md"); print("workspace:", ws)


def finish_run(a):
    run = EXP / "runs" / a.run_id; ws = run / "ws"; task = EXP / "tasks" / json.loads((EXP / "state.json").read_text())["runs"][a.run_id]["task"]
    out = subprocess.run([str(H / "venv3.12/bin/python"), str(HERE / "score_task.py"), str(ws), str(task)], capture_output=True, text=True).stdout
    score = json.loads(out); (run / "score.json").write_text(json.dumps(score, indent=1))
    calls = [json.loads(l) for l in open(ws / ".delegation/delegate_calls.jsonl")] if (ws / ".delegation/delegate_calls.jsonl").exists() else []
    s = state(); r = s["runs"][a.run_id]
    r.update({"status": "scored", "strong_tokens": a.tokens, "strong_tool_uses": a.tool_uses, "strong_duration_s": a.duration_s, "success": score["success"], "tests_pass": f"{score['tests_pass']}/{score['tests_expected']}",
              "delegate_calls": len(calls), "delegate_tokens_in": sum(c["tokens_in"] for c in calls), "delegate_tokens_out": sum(c["tokens_out"] for c in calls), "delegate_cost": round(sum(c["cost"] for c in calls), 5),
              "usage_pct": [a.percent_before, a.percent_after]})
    save(s); print(json.dumps(r, indent=1))


def status(_):
    for rid, r in state()["runs"].items():
        print(f"{rid:48s} {r['status']:9s} success={r.get('success')} strong_tokens={r.get('strong_tokens')} delegate_cost={r.get('delegate_cost')}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new-run"); n.add_argument("task"); n.add_argument("arm", choices=["solo", "delegate"]); n.add_argument("rep", type=int)
    n.add_argument("--strategy", default="S1"); n.add_argument("--context", default="auto"); n.add_argument("--verify", default="exec"); n.add_argument("--delegate-model", default="particle-free"); n.add_argument("--cell", default=None); n.add_argument("--handoff", default=None)
    f = sub.add_parser("finish-run"); f.add_argument("run_id"); f.add_argument("--tokens", type=int, required=True); f.add_argument("--tool-uses", type=int, default=0); f.add_argument("--duration-s", type=int, default=0)
    f.add_argument("--percent-before", type=int, default=None); f.add_argument("--percent-after", type=int, default=None)
    sub.add_parser("status")
    a = ap.parse_args(); {"new-run": new_run, "finish-run": finish_run, "status": status}[a.cmd](a)


if __name__ == "__main__":
    main()
