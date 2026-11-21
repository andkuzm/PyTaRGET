"""Strong-arm work per pilot cell, from the sub-agent transcripts: the fixed start-up context (system prompt + tool definitions, the context of the FIRST model call) is subtracted, because it is the same for every run and unrelated to the task.
calls = model calls; added_ctx = context growth (last call - first call); work = sum over calls of (context - first-call context) = extra input tokens processed because of the task; out = output tokens."""
import json
import sys

T = "/tmp/claude-253910/-gpfs-helios-home-kandrei/55172aa0-a22c-4000-9663-4dc4dbc73303/tasks/"
CELLS = [("P1-1 shop solo r1", "a2e2340b0d9c4a0fd"), ("P3-1 shop solo r2", "afe28becf0aa388ac"), ("P1-2 shop S1", "a8593c64f814dc7eb"), ("P2-0b shop S1 r2", "a79b188e2eab1d762"),
         ("P2-1 shop S2", "a8f9fb26afbfe8203"), ("P2-2 shop S3", "ac4200e28aaf82013"), ("P2-3 shop S1 strong ctx", "a5f34496e13ef6244"), ("P2-4 shop S1 both ctx", "aa12e8e0943a281fa"),
         ("P2-5 shop S1 diff check", "a9bf53b14ff2885d9"), ("P1-3 gson_a80 solo", "a25c1925eef006220"), ("P1-4 gson_a80 S1 (tool bug)", "a95d0dc882c428bf4"), ("P2-0 gson_a80 S1", "a8564fb90bbc2d823"),
         ("P1-5 gson_cbad solo", "ab2132f60ae430498"), ("P1-6 gson_cbad S1", "ad0e02dbdbe1f202b"), ("P3-2 gson_cbad S2", "acd218cb01d150917")]
rows = []
for name, f in CELLS:
    seen = {}
    for l in open(T + f + ".output"):
        try: d = json.loads(l)
        except ValueError: continue
        m = d.get("message") or {}
        if d.get("type") == "assistant" and m.get("usage"): seen[m.get("id") or d.get("uuid")] = m["usage"]
    u = list(seen.values())
    ctx = [(x.get("input_tokens", 0) or 0) + (x.get("cache_creation_input_tokens", 0) or 0) + (x.get("cache_read_input_tokens", 0) or 0) for x in u]
    out = sum(x.get("output_tokens", 0) or 0 for x in u)
    rows.append((name, len(u), ctx[-1] - ctx[0], sum(c - ctx[0] for c in ctx), out, ctx[0]))
print("| Cell | model calls | added context | work (extra input tokens processed) | output tokens |\n|---|---|---|---|---|")
for n, c, a, w, o, c0 in rows: print(f"| {n} | {c} | {a:,} | {w:,} | {o:,} |")
print(f"\nfixed start-up context (first call): {min(r[5] for r in rows):,} - {max(r[5] for r in rows):,} tokens")
