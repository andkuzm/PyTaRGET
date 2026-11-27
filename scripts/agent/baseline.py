"""Which tests ALREADY fail in the agent sandbox on the OLD code (before the source change)? Such a failure comes from the environment (no network, restricted folders), not from the change,
and the agent must not rewrite the test to get around it. The old code is rebuilt in a throw-away copy by reversing SOURCE_CHANGES.diff; delegate_worker.py then runs the suite there inside
the same sandbox. This needs the OLD dependency versions in the Maven cache (offline); without them the result is "unavailable", which is reported, never silently treated as "none"."""
import re
import shutil
import subprocess
from pathlib import Path


def reverse_changes(copy):
    """undo SOURCE_CHANGES.diff in `copy` (a directory that may be thrown away). returns (ok, error text)."""
    copy = Path(copy); diff = copy / "SOURCE_CHANGES.diff"
    if not diff.exists(): return False, "no SOURCE_CHANGES.diff"
    shutil.rmtree(copy / ".git", ignore_errors=True)
    subprocess.run(["git", "init", "-q"], cwd=copy, capture_output=True)

    def apply():
        return subprocess.run(["git", "apply", "-R", "--whitespace=nowarn", "SOURCE_CHANGES.diff"], cwd=copy, capture_output=True, text=True)

    r = apply()
    if r.returncode:                                   # the diff may have lost Windows line endings: make the files it touches LF (Java does not care) and try again
        for rel in re.findall(r"^\+\+\+ (?:b/)?(\S+)", diff.read_text(errors="ignore"), flags=re.M):
            f = copy / rel
            if f.is_file() and copy.resolve() in f.resolve().parents: f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
        r = apply()
    return r.returncode == 0, r.stderr.strip()[:300]


def failing_ids(results):
    """[(class, method, status, message)] -> sorted 'pkg.Class#method' of failed or erroring tests; parameter suffixes are removed."""
    return sorted({f"{c}#{re.split(r'[\[({ ]', m or '')[0]}" for c, m, status, _ in results if status in ("fail", "error")})
