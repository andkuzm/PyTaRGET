"""Audit a sub-agent transcript (JSONL of the Agent tool): list every tool call with its path or command and flag what the rules forbid: paths outside the run workspace (except the prompt),
anything in the reference material or datasets, the OpenRouter key, network commands, history lookups. usage: audit_transcript.py <transcript.jsonl> <workspace> -> JSON on stdout."""
import json
import re
import sys
from pathlib import Path

transcript, ws = sys.argv[1], str(Path(sys.argv[2]).resolve())
FORBIDDEN_PATH = re.compile(r"/delegation_exp/(tasks|synth|scratch|runs/(?!%s))|/references/|annotated_cases|\.config/openrouter|/backups/|eval_|source_hashes|old_tests|/ref/" % re.escape(Path(ws).parent.name))
NET = re.compile(r"\b(curl|wget|git\s+(clone|fetch|pull|remote|ls-remote|log|show|reflog)|pip\s+(install|download)|ssh|nc)\b")
calls, flags = [], []
for line in open(transcript):
    try: d = json.loads(line)
    except ValueError: continue
    content = (d.get("message") or {}).get("content")
    if not isinstance(content, list): continue
    for c in content:
        if isinstance(c, dict) and c.get("type") == "tool_use":
            inp = c.get("input", {}); name = c.get("name")
            what = inp.get("file_path") or inp.get("command") or inp.get("path") or inp.get("pattern") or str(inp)
            calls.append((name, str(what)[:200]))
            w = str(what)
            if name == "SubagentHandback": continue
            if FORBIDDEN_PATH.search(w): flags.append({"call": name, "what": w[:200], "why": "reference material, dataset or key"})
            if name == "Bash" and NET.search(w): flags.append({"call": name, "what": w[:200], "why": "network or history command"})
            if name in ("Read", "Write", "Edit", "Glob", "Grep"):
                p = inp.get("file_path") or inp.get("path") or ""
                if p and not (str(p).startswith(ws) or p.endswith("prompt.md")): flags.append({"call": name, "what": str(p)[:200], "why": "outside the workspace"})
            if name == "Bash" and re.search(r"\bcd\s+(/|~|\$HOME)(?!\S*%s)" % re.escape(Path(ws).name), w) and ws not in w: flags.append({"call": name, "what": w[:200], "why": "leaves the workspace"})
print(json.dumps({"tool_calls": len(calls), "flags": flags, "clean": not flags}, indent=1))
