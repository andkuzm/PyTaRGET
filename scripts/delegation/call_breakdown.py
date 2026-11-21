"""Per-call breakdown of one sub-agent transcript: for every model call the output tokens, the context size it started with, what it did (tool + short command, or text) and how many characters
the tool result that followed added. usage: call_breakdown.py <transcript.jsonl>"""
import json
import sys

rows, results, order = {}, {}, []
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    m = d.get("message") or {}; c = m.get("content")
    if d.get("type") == "assistant" and m.get("usage"):
        k = m.get("id") or d.get("uuid")
        if k not in rows: order.append(k); rows[k] = {"u": m["usage"], "acts": []}
        rows[k]["u"] = m["usage"]
        for b in (c if isinstance(c, list) else []):
            if b.get("type") == "tool_use":
                i = b.get("input", {}); w = i.get("command") or i.get("file_path") or i.get("pattern") or str(i)
                rows[k]["acts"].append((b.get("name"), " ".join(str(w).split())[:90], b.get("id")))
            elif b.get("type") == "text": rows[k]["acts"].append(("text", " ".join(b.get("text", "").split())[:60], None))
            elif b.get("type") == "thinking": rows[k]["acts"].append(("thinking", "", None))
    elif d.get("type") == "user" and isinstance(c, list):
        for b in c:
            if b.get("type") == "tool_result":
                t = b.get("content"); t = t if isinstance(t, str) else "".join(x.get("text", "") for x in t if isinstance(x, dict))
                results[b.get("tool_use_id")] = len(t)
print(f"{'#':>2} {'ctx':>7} {'out':>6}  action (result chars)")
tot_out = 0
for n, k in enumerate(order, 1):
    u = rows[k]["u"]; ctx = (u.get("input_tokens", 0) or 0) + (u.get("cache_creation_input_tokens", 0) or 0) + (u.get("cache_read_input_tokens", 0) or 0); out = u.get("output_tokens", 0) or 0; tot_out += out
    acts = "; ".join(f"{a[0]}: {a[1]}" + (f" ({results.get(a[2], 0)} chars)" if a[2] else "") for a in rows[k]["acts"] if a[0] != "thinking") or "(thinking only)"
    print(f"{n:>2} {ctx:>7} {out:>6}  {acts[:230]}")
print(f"total output {tot_out}")
