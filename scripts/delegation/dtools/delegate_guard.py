"""Runs tools/delegate detached and waits at most ~160 s per call, because the agent harness moves any command that runs longer than 180 s to the background (and agents then burn calls polling).
If delegate is not finished, prints a fixed line; call `tools/delegate --wait` to keep waiting (one cheap call). usage: tools/delegate [args]  |  tools/delegate --wait"""
import os
import subprocess
import sys
import time
from pathlib import Path

ws = Path(sys.argv[1]); args = sys.argv[2:]
d = ws / ".delegation"; d.mkdir(exist_ok=True); out = d / "delegate_stdout.txt"; pidf = d / "delegate.pid"
LIMIT = 160


def alive():
    try: os.kill(int(pidf.read_text()), 0); return True
    except (OSError, ValueError): return False


def reap():
    try: os.waitpid(int(pidf.read_text()), os.WNOHANG)
    except (OSError, ValueError, ChildProcessError): pass


if args[:1] != ["--wait"]:
    if pidf.exists() and alive(): print("a delegate run is already in progress: call `tools/delegate --wait`"); sys.exit(2)
    f = open(out, "w")
    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve().parent / "delegate.py"), str(ws)] + args, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    pidf.write_text(str(p.pid))
t0 = time.time()
while time.time() - t0 < LIMIT:
    reap()
    if not alive(): break
    time.sleep(2)
if alive():
    print("DELEGATE STILL RUNNING (not an error, nothing to fix). Call `tools/delegate --wait` and do nothing else until it prints a result."); sys.exit(3)
lines = out.read_text(errors="replace").splitlines()
print("\n".join(lines[-30:]))
