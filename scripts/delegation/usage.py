"""Read the Claude Code usage with `claude -p "/usage"` and apply the caps of the delegation experiment. Prints JSON:
{session_pct, session_reset, week_pct, week_reset, window_cap, week_cap, window_room, week_room}.
Caps (user, 2026-11-21): the limit window that resets at 8:19/8:20 on Nov 21 -> 80%; every later window (unattended) -> 50%; week -> see week_cap_for (user 2026-11-26)."""
import json
import re
import subprocess

out = subprocess.run(["claude", "-p", "/usage"], capture_output=True, text=True, timeout=120).stdout
s = re.search(r"Current session: (\d+)% used · resets ([^\n(]+)", out); w = re.search(r"Current week \(all models\): (\d+)% used · resets ([^\n(]+)", out)
sp, sr = (int(s.group(1)), s.group(2).strip()) if s else (None, None)
wp, wr = (int(w.group(1)), w.group(2).strip()) if w else (None, None)
import datetime
def week_cap_for(reset_text, now=None):
    """user 2026-11-26: cumulative per-day caps. Day k (1..7) of the week that ends at the reset -> k/7 of 100 %, never above 95 % (more only with the user's explicit approval)."""
    m = re.search(r"(\w{3}) (\d+), (\d+)(?::(\d+))?(am|pm)", reset_text or "")
    if not m: return 14, 1                                          # unreadable: the strictest value
    now = now or datetime.datetime.now()
    mon, day, h, mi, ap = m.groups()
    reset = datetime.datetime.strptime(f"{now.year} {mon} {day}", "%Y %b %d").replace(hour=int(h) % 12 + (12 if ap == "pm" else 0), minute=int(mi or 0))
    if reset < now - datetime.timedelta(days=1): reset = reset.replace(year=now.year + 1)
    left = (reset - now).total_seconds() / 86400
    k = min(7, max(1, int(7 - left) + 1))
    return min(95, int(100 * k / 7)), k
wk_cap, wk_day = week_cap_for(wr)
cap = 70   # user, 2026-11-21 15:10: daily window cap 70% "this time", week cap 20% (user 2026-11-21 19:00: continue till 20% at most for now)
print(json.dumps({"session_pct": sp, "session_reset": sr, "week_pct": wp, "week_reset": wr, "window_cap": cap, "week_cap": wk_cap, "week_day": wk_day,
                  "window_room": None if sp is None else cap - sp, "week_room": None if wp is None else wk_cap - wp}))
