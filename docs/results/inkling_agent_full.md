# Inkling-2-Mini (effort low) as an agent: full runs on the sets of the 80-call Qwen3-Coder agent - 2026-11-26
mini-SWE-agent 2.4.6, vLLM nightly, offline network namespace + per-case mount sandbox, 80 calls max, 128k context, 12288 output tokens per call (Qwen agent: 64k, 2048).
Jobs: Python 4083956/4083957/4083958, Java 4083959/4083960/4083961. The agent took 12.6 min (Python, 74 cases) and 10.5 min after server start (Java, 59 cases) on one H200.

## Python (sibling-safe workspaces), 73 common cases
| | passes | strict (no edit outside the target function) |
|---|---|---|
| Inkling agent | 69 (94.5%) | 68 (93.2%) |
| Qwen3-Coder agent | 39 (53.4%) | 23 (31.5%) |
Paired (strict): only Inkling 45, only Qwen 0. Against Inkling's SINGLE prompt (first answer, same cases, taken from the full-test-set run; 296 of 300 contrast cases matched): agent strict 69 vs 43 of 73 (94.5% vs 58.9%), paired 27 : 1.
25 of 74 final functions equal the real repair.

## Java (same 59 cases as the Qwen agent; 58 common to all columns)
```
inkling agent cases 59; same ids as the Qwen agent preparation: 59
cases 59; common to all columns 58; clean (no other test lines changed) 4

column                solved  unsolved  weighted    total   (checked)
qwen3coder              65.6      19.2      42.7   26/58
gpt-oss-20b             59.4      15.4      37.6   23/58
ernie-5.5-flash         84.4      19.2      52.2   32/58
particle-1              93.8      42.3      68.3   41/58
nemotron-3-ultra        81.2      30.8      56.3   34/58
qwen3coder BC (context)    62.5      26.9      44.9   27/58
particle-1 BC (context)    93.8      57.7      75.9   45/58
AGENT 40 strict         65.6      30.8      48.4   29/58
AGENT 40 relaxed        78.1      38.5      58.5   35/58
AGENT 80 strict         71.9      53.8      63.0   37/58
AGENT 80 relaxed        78.1      61.5      69.9   41/58
INK AGENT strict       100.0     100.0     100.0   58/58
INK AGENT relaxed      100.0     100.0     100.0   58/58
inkling low (1 prompt)    87.5      26.9      57.6   35/58

Split by whether the repair changed other test lines (checked passes):
  clean (0 other lines)  n=  4 | AGENT 40 strict 3/4 | AGENT 40 relaxed 3/4 | AGENT 80 strict 3/4 | AGENT 80 relaxed 3/4 | INK AGENT strict 4/4 | INK AGENT relaxed 4/4 | inkling low (1 prompt) 3/4 | qwen3coder 2/4 | particle-1 4/4 | nemotron-3-ultra 3/4
  other tests repaired   n= 54 | AGENT 40 strict 26/54 | AGENT 40 relaxed 32/54 | AGENT 80 strict 34/54 | AGENT 80 relaxed 38/54 | INK AGENT strict 54/54 | INK AGENT relaxed 54/54 | inkling low (1 prompt) 32/54 | qwen3coder 24/54 | particle-1 37/54 | nemotron-3-ultra 31/54

paired INK AGENT strict vs inkling low (1 prompt): +23 -0
paired INK AGENT relaxed vs inkling low (1 prompt): +23 -0
paired INK AGENT strict vs AGENT 80 strict: +21 -0
paired INK AGENT relaxed vs AGENT 80 relaxed: +17 -0
paired INK AGENT strict vs particle-1: +17 -0
paired AGENT 80 relaxed vs particle-1 BC (context): +7 -11
paired AGENT 80 strict vs particle-1 BC (context): +5 -13
paired AGENT 80 relaxed vs qwen3coder BC (context): +19 -5
paired AGENT 80 strict vs qwen3coder BC (context): +16 -6
paired AGENT 40 strict vs particle-1: +4 -16
paired AGENT 40 relaxed vs particle-1: +7 -13
paired AGENT 80 strict vs particle-1: +8 -12
paired AGENT 80 relaxed vs particle-1: +10 -10
paired AGENT 80 strict vs AGENT 40 strict: +13 -5
paired AGENT 80 relaxed vs AGENT 40 relaxed: +10 -4
paired AGENT 80 relaxed vs qwen3coder: +20 -5
```

## Checks on the very high pass rates (the agent stops when its test passes, and the test is the scorer)
- Network probe 000. Compiled test classes were never inspected; surefire reports were read only after the agent's own Maven run. Outside its workspace an agent only saw its own staging path under /tmp.
- Java keeps sibling tests repaired (confound shared with the Qwen runs: 54 of 58 cases); Python does not, and the agent stays at 93-95% there.
- Heuristic weakening check (scripts/thesis_analysis/agent_weakening_check.py), passing final functions that differ from the real repair:
  Inkling Python 7 of 45 flagged (fewer assertions / exact checks); Qwen Python 7 of 24; Inkling Java 2 of 14 (all final methods); Qwen Java 13 of 27 (all final methods).
- Read by hand (5 Python, 2 Java flagged): real weakening in several (e.g. case 158 changed the 'bad' input and dropped the fail-closed assertion; 189 dropped the checks of the written job file; 83 dropped assert_not_called; Java 60 swallows an exception; Java 2 adds an environment workaround).
  Even counting all 7 flagged Python cases as failures: 62 of 73 (about 85%) against 59% for the single prompt. The plausibility score does NOT detect these.

## Decision (user rule: agent must lead by >= 10 points and >= 2x paired in BOTH languages, strict): the AGENT is the clear winner.
