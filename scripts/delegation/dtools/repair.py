"""tools/repair: the whole routine in ONE call, no model in the loop: autorename + tests, then up to 3 delegate rounds while tests fail, then final tests + review.
Prints ACCEPT (all tests pass, review has no flag: nothing left to do) or ESCALATE with only what needs the strong agent (failing tests / flagged methods).
Runs detached and returns within 160 s (the harness backgrounds longer commands); if unfinished it prints a fixed line: call `tools/repair --wait`. usage: tools/repair | tools/repair --wait"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
FLAGS = ("ASSERTIONS DROPPED", "FAILURE CHECK LOST", "REMOVED")


def sh(args, ws):
    return subprocess.run([sys.executable, str(HERE / args[0])] + [str(ws)] + args[1:], capture_output=True, text=True)


def passed(out): return bool(re.search(r"TESTS: \d+ run, \d+ passed, 0 failed", out))


def benign(ws, rv):
    """review blocks flagged ONLY 'ASSERTIONS DROPPED' where every dropped assertion is assertTrue/False(<call>) and that call's method is void in src/main now = the expected consequence of a boolean->void change.
    returns (rv without those blocks, number auto-accepted)"""
    src = "\n".join(f.read_text(errors="ignore") for f in (ws / "src" / "main").rglob("*.java")) if (ws / "src" / "main").exists() else ""
    if not src: src = "\n".join(f.read_text(errors="ignore") for f in ws.rglob("src/main/**/*.java"))
    blocks = re.split(r"\n(?=\S+\.java#)", rv); keep, n = [blocks[0]], 0
    for b in blocks[1:]:
        head, *body = b.split("\n"); minus = [l[3:].strip() for l in body if l.startswith("  -")]; plus = [l[3:].strip() for l in body if l.startswith("  +")]
        ok = "[ASSERTIONS DROPPED]" in head and "FAILURE CHECK LOST" not in head and "REMOVED" not in head
        if ok:
            calls = [c for c in (re.match(r"assert(?:True|False)\((?:\w+\.)*(\w+)\(.*\)\);$", m) for m in minus) if c]
            ok = bool(calls) and all(re.search(r"\bvoid\s+" + re.escape(c.group(1)) + r"\s*\(", src) for c in calls) and len(minus) - len(plus) <= len(calls)
        if ok: n += 1
        else: keep.append(b)
    return "\n".join(keep), n


def badness(out):
    """how much is still wrong: compile errors count as 1e6 + N (nothing ran), failing tests as their number, 0 when everything passes."""
    m = re.search(r"COMPILE ERRORS: (\d+)", out)
    if m: return 10**6 + int(m.group(1))
    m = re.search(r"TESTS: \d+ run, \d+ passed, (\d+) failed", out)
    return int(m.group(1)) if m else 10**7


def snapshot(ws, dst):
    import shutil
    from dtools import common
    cfg = common.load_config(ws); shutil.rmtree(dst, ignore_errors=True)
    for r in cfg.get("test_roots", ["src/test/java"]):
        if (ws / r).exists(): shutil.copytree(ws / r, dst / r)


def restore(ws, src):
    import shutil
    from dtools import common
    cfg = common.load_config(ws)
    for r in cfg.get("test_roots", ["src/test/java"]):
        if (src / r).exists(): shutil.rmtree(ws / r); shutil.copytree(src / r, ws / r)


def pipeline(ws):
    print("== tools/start"); st = sh(["start.py"], ws); print(st.stdout.strip()[-1500:])
    t = st.stdout
    failed = lambda out: int((re.search(r"(\d+) failed", out) or [0, 10**6])[1]) if "TESTS:" in out else 10**6      # compile errors count as "all failing"
    for rnd in range(1, 4):
        if passed(t): break
        print(f"== delegate round {rnd}"); before = failed(t); b0 = badness(t); snap = ws / ".delegation" / f"snap_round{rnd}"; snapshot(ws, snap); d = sh(["delegate.py"], ws); lines = d.stdout.strip().splitlines(); print("\n".join(lines[-12:]))
        if d.returncode and not d.stdout.strip(): print("delegate failed:", d.stderr[-300:]); break
        no_units = bool(re.search(r"\b0 unit\(s\)", d.stdout))
        t = sh(["run_tests.py"], ws).stdout
        b1 = badness(t); show = lambda b: f"{b - 10**6} compile errors" if b >= 10**6 else f"{b} failing tests"
        if not no_units and not passed(t) and (b1 - (10**6 if b1 >= 10**6 else 0)) > 0.5 * (b0 - (10**6 if b0 >= 10**6 else 0)) and (b1 >= 10**6) == (b0 >= 10**6):   # the round fixed less than half of what was wrong: undo it, the strong agent starts from the state before it (partial, partly wrong edits cost it more than they save)
            restore(ws, snap); t = sh(["run_tests.py"], ws).stdout
            print(f"== ROLLED BACK delegate round {rnd}: {show(b0)} before, {show(b1)} after; fixed less than half; the tests are as before the round"); break
        if no_units or (not passed(t) and failed(t) >= before): print("== no progress this round (nothing left that the delegate can take): stopping the delegate rounds"); break
    else: t = sh(["run_tests.py"], ws).stdout
    print("== tests"); print(t.strip()[-1500:])
    if not passed(t): print("\nESCALATE: tests still fail (above). Repair these yourself or call tools/delegate --units <Class#method>."); return
    rv = sh(["review.py"], ws).stdout if (ws / "tools" / "review").exists() else ""
    flagged = [l for l in rv.splitlines() if any(f in l for f in FLAGS) and not l.lstrip().startswith("flags:")] if rv else []
    if not rv: print("\nACCEPT: all tests pass (no tools/review in this workspace, assertion check not done)."); return
    rv2, n = benign(ws, rv); note = f" ({n} flagged method(s) auto-accepted: only a dropped assertTrue/False(call()) whose method is now void)" if n else ""
    if not any(f in rv2 for f in FLAGS): print("\nACCEPT: all tests pass, review flags nothing" + note + ". Nothing left to do: send the final report."); return
    print("\nESCALATE: tests pass but review flags methods; check these only:" + note + "\n" + rv2[:4000])


if __name__ == "__main__":
    ws = Path(sys.argv[1]); args = sys.argv[2:]
    if args[:1] == ["--pipeline"]: pipeline(ws); sys.exit(0)
    d = ws / ".delegation"; d.mkdir(exist_ok=True); out = d / "repair_stdout.txt"; pidf = d / "repair.pid"; LIMIT = 160

    def alive():
        try: os.kill(int(pidf.read_text()), 0); return True
        except (OSError, ValueError): return False

    if args[:1] != ["--wait"]:
        if pidf.exists() and alive(): print("a repair run is already in progress: call `tools/repair --wait`"); sys.exit(2)
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(ws), "--pipeline"], stdout=open(out, "w"), stderr=subprocess.STDOUT, start_new_session=True); pidf.write_text(str(p.pid))
    t0 = time.time()
    while time.time() - t0 < LIMIT:
        try: os.waitpid(int(pidf.read_text()), os.WNOHANG)
        except (OSError, ValueError, ChildProcessError): pass
        if not alive(): break
        time.sleep(2)
    if alive(): print("REPAIR STILL RUNNING (not an error). Call `tools/repair --wait` and do nothing else until it prints ACCEPT or ESCALATE."); sys.exit(3)
    print(out.read_text(errors="replace").strip())
