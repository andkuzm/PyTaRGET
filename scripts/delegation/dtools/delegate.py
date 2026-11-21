"""tools/delegate [--units all-failing|<file or Class#method>[,...]] [--context auto|strong|both] [--model alias] [--rounds N] [--dry-run]
Sends every failing test unit to a cheap delegate model with an automatically extracted (and optionally strong-agent supplied) context, applies the returned repairs (with safeguards), re-runs the
tests and prints ONE compact result (the delegate's reasoning is never shown). Every call is logged in .delegation/delegate_calls.jsonl with tokens and cost."""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dtools import common, context


class DelegateUnavailable(Exception):
    pass


def call_model(cfg, alias, prompt):
    d = cfg["delegates"][alias]
    key = "none" if d.get("key") == "none" else Path(cfg["key_file"]).expanduser().read_text().strip()
    body = {"model": d["slug"], "messages": [{"role": "user", "content": prompt}], "temperature": d.get("temperature", 0.2), "max_tokens": d.get("max_tokens", 8192)}
    if d.get("reasoning"): body["reasoning"] = {"effort": d["reasoning"]}
    last = ""
    for attempt in range(6):
        req = urllib.request.Request(d["base_url"].rstrip("/") + "/chat/completions", json.dumps(body).encode(), {"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "dtools"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
            u = r.get("usage") or {}
            tin, tout = u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0
            cost = max(float(u.get("cost", 0) or 0), (tin * d.get("price_in", 0) + tout * d.get("price_out", 0)) / 1e6)
            return (r["choices"][0]["message"].get("content") or ""), tin, tout, cost
        except urllib.error.HTTPError as e:
            last = e.read().decode("utf-8", "ignore")[:300]
            if "per-day" in last or "free-models-per-day" in last: raise DelegateUnavailable("daily limit of the free tier reached")
            if e.code in (401, 402, 403): raise DelegateUnavailable(f"HTTP {e.code}: {last[:120]}")
        except Exception as e:
            last = str(e)[:200]
        time.sleep(min(60, 4 * 2 ** attempt))
    raise DelegateUnavailable("model did not answer: " + last)


_ACTIVE = {}
_LOCK = None


def paid_spent(cfg):
    f = Path(cfg.get("paid_ledger", "~/delegation_exp/delegate_paid_ledger.json")).expanduser()
    return json.loads(f.read_text()).get("spent", 0.0) if f.exists() else 0.0


def paid_add(cfg, cost):
    f = Path(cfg.get("paid_ledger", "~/delegation_exp/delegate_paid_ledger.json")).expanduser()
    f.write_text(json.dumps({"spent": round(paid_spent(cfg) + cost, 6)}))


def call_with_fallback(cfg, alias, prompt):
    """Returns (alias used, (text, tokens in, tokens out, cost)). If the model is unavailable and a paid fallback is configured, switches to it for the rest of the run (global spending cap in the ledger)."""
    use = _ACTIVE.get(alias, alias)
    try:
        res = call_model(cfg, use, prompt)
    except DelegateUnavailable:
        fb = cfg.get("fallback", {}).get(use)
        if not fb or paid_spent(cfg) >= cfg.get("paid_cap_usd", 0.30):
            raise
        _ACTIVE[alias] = use = fb
        res = call_model(cfg, use, prompt)
    if cfg["delegates"][use].get("price_out", 0) > 0:
        paid_add(cfg, res[3])
    return use, res


def extract(answer, tag):
    m = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", answer, re.S)
    if not m: return None
    text = m.group(1).strip("\n")
    fence = re.match(r"^```\w*\n(.*?)\n?```\s*$", text.strip(), re.S)
    return fence.group(1) if fence else text


def add_imports(lines, imports):
    existing = set(l.strip() for l in lines)
    new = [i.strip() for i in imports.split("\n") if i.strip().startswith("import ") and i.strip() not in existing]
    if not new: return lines
    last = max([k for k, l in enumerate(lines) if l.startswith("import ")] or [next((k for k, l in enumerate(lines) if l.startswith("package ")), -1)])
    return lines[:last + 1] + new + lines[last + 1:]


def apply(ws, unit, repair, imports):
    """Apply a candidate; returns (ok, note). Refuses candidates that would delete most of the code."""
    path = Path(ws) / unit["file"]; lines = path.read_text(errors="ignore").split("\n"); new = repair.split("\n")
    if unit["kind"] == "file":
        if len(new) < 0.5 * len(lines) or common.count_asserts(repair) < 0.7 * common.count_asserts("\n".join(lines)) - 1:
            return False, "rejected (removes too much code or too many assertions)"
        lines = new
    elif unit["kind"] == "header":
        lines = new + lines[unit["last_line"]:]
    else:
        old = "\n".join(lines[unit["first"] - 1:unit["last"]])
        if common.count_asserts(repair) < common.count_asserts(old) - 1:
            return False, "rejected (fewer assertions than the old test)"
        lines = lines[:unit["first"] - 1] + new + lines[unit["last"]:]
    if imports: lines = add_imports(lines, imports)
    path.write_text("\n".join(lines))
    return True, "applied"


def run_round(ws, cfg, args, rnd, strong):
    ws = Path(ws)
    rc, log = common.mvn(ws)
    kind, data = common.result_of(ws, rc, log)
    if kind == "tests" and all(r[2] in ("pass", "skipped") for r in data):
        return None, "all tests pass"
    units = context.units_from_state(ws, kind, data, cfg)
    if args.units != "all-failing":
        want = set(args.units.split(","))
        units = [u for u in units if u["file"] in want or u["name"] in want or Path(u["file"]).name in want]
    hunks = context.parse_hunks((ws / "SOURCE_CHANGES.diff").read_text()) if (ws / "SOURCE_CHANGES.diff").exists() else []
    units = units[:args.max_units]
    prompts = [context.build_prompt(ws, u, log, hunks, args.context, strong) for u in units]
    if args.dry_run:
        return None, f"dry run: {len(units)} unit(s); first prompt ({len(prompts[0]) if prompts else 0} chars):\n" + (prompts[0][:3000] if prompts else "")
    def one(i):
        t = time.time(); used, (ans, tin, tout, cost) = call_with_fallback(cfg, args.model, prompts[i])
        return ans, tin, tout, cost, time.time() - t, used
    out = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        out = list(pool.map(one, range(len(units))))
    backup = ws / ".delegation" / "backup" / f"round{rnd}"; backup.mkdir(parents=True, exist_ok=True)
    notes, tin_s, tout_s, cost_s = [], 0, 0, 0.0
    for u, (ans, tin, tout, cost, secs, used) in zip(units, out):
        repair, imports = extract(ans, "REPAIR"), extract(ans, "IMPORTS")
        tin_s += tin; tout_s += tout; cost_s += cost
        (backup / (u["name"].replace("#", "_").replace(" ", "_") + ".answer.txt")).write_text(ans)
        if repair is None: ok, note = False, "no [REPAIR] block in the answer"
        else:
            (backup / Path(u["file"]).name).write_text((ws / u["file"]).read_text())
            ok, note = apply(ws, u, repair, imports)
        notes.append(f"{u['name']}: {note}")
        with open(ws / ".delegation" / "delegate_calls.jsonl", "a") as f:
            f.write(json.dumps({"round": rnd, "unit": u["name"], "model": used, "tokens_in": tin, "tokens_out": tout, "cost": cost, "seconds": round(secs, 1), "applied": ok, "context": args.context}) + "\n")
    return {"units": len(units), "tin": tin_s, "tout": tout_s, "cost": cost_s, "notes": notes}, ""


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace"); ap.add_argument("--units", default="all-failing"); ap.add_argument("--context", default="auto", choices=["auto", "strong", "both"])
    ap.add_argument("--model", default=None); ap.add_argument("--rounds", type=int, default=1); ap.add_argument("--max-units", type=int, default=40); ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    ws = Path(args.workspace); cfg = common.load_config(ws); args.model = args.model or cfg["default_delegate"]
    sf = ws / ".delegation" / "context.json"
    strong = json.loads(sf.read_text()) if sf.exists() else {}
    if args.context in ("strong", "both") and not strong:
        print("delegate: --context %s needs .delegation/context.json (written by you); none found" % args.context); return 2
    total = {"units": 0, "tin": 0, "tout": 0, "cost": 0.0}; notes = []
    try:
        for rnd in range(1, args.rounds + 1):
            res, msg = run_round(ws, cfg, args, rnd, strong)
            if res is None:
                print(f"delegate: {msg}"); break
            for k in total: total[k] += res[k]
            notes += res["notes"]
    except DelegateUnavailable as e:
        print(f"delegate: UNAVAILABLE ({e}). Repair the remaining tests yourself."); return 3
    if args.dry_run: return 0
    print(f"delegate: model {args.model} | {total['units']} unit(s) in {args.rounds} round(s) | tokens in {total['tin']} out {total['tout']} | cost ${total['cost']:.4f}")
    for n in notes[:25]: print("  " + n)
    rc, log = common.mvn(ws); kind, data = common.result_of(ws, rc, log)
    print(common.compact(ws, kind, data, rc))
    return 0 if kind == "tests" and all(r[2] in ("pass", "skipped") for r in data) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
