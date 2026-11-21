"""Read the Claude Code usage with `claude -p "/usage"` and apply the caps of the delegation experiment. Prints JSON:
{session_pct, session_reset, week_pct, week_reset, window_cap, week_cap, window_room, week_room}.
Caps (user, 2026-11-21): the limit window that resets at 8:19/8:20 on Nov 21 -> 80%; every later window (unattended) -> 50%; week -> 93%."""
import json
import re
import subprocess

out = subprocess.run(["claude", "-p", "/usage"], capture_output=True, text=True, timeout=120).stdout
s = re.search(r"Current session: (\d+)% used · resets ([^\n(]+)", out); w = re.search(r"Current week \(all models\): (\d+)% used · resets ([^\n(]+)", out)
sp, sr = (int(s.group(1)), s.group(2).strip()) if s else (None, None)
wp, wr = (int(w.group(1)), w.group(2).strip()) if w else (None, None)
cap = 80 if (sr and re.search(r"Nov 21, 8:(1|2)\d", sr)) else 50
print(json.dumps({"session_pct": sp, "session_reset": sr, "week_pct": wp, "week_reset": wr, "window_cap": cap, "week_cap": 93,
                  "window_room": None if sp is None else cap - sp, "week_room": None if wp is None else 93 - wp}))
