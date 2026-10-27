"""
The processed-repositories file: which repositories are done and where an unfinished one stopped.

One repository per line, fields separated by "|":

    owner/name                    completed (old blacklist entries have no commit at all)
    owner/name|<head>             completed; <head> is the commit the search started from
    owner/name|<head>|<marker>    IN PROGRESS; <marker> is the last completely processed commit ("" = none yet)

A repository whose line has three fields is picked up again by the next run, which walks the history from <head>
again (the walk is deterministic) and continues after <marker>. Updates replace the repository's own line in place
and go through a temporary file, under a lock file where fcntl exists, so a kill cannot leave a torn file.
"""
import os
import tempfile
import threading
from collections import namedtuple
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: no locking, fine for a single process
    fcntl = None

Entry = namedtuple("Entry", "name head marker in_progress")

# flock keeps processes apart; this keeps the worker threads of one searcher apart (and is all there is on Windows)
_THREAD_LOCK = threading.Lock()


def parse_line(line):
    fields = line.rstrip("\r\n").split("|")
    name = fields[0].strip()
    if not name:
        return None
    head = fields[1].strip() if len(fields) > 1 else ""
    if len(fields) >= 3:
        return Entry(name, head, fields[2].strip(), True)
    return Entry(name, head, "", False)


def format_entry(entry):
    if entry.in_progress:
        return f"{entry.name}|{entry.head}|{entry.marker}"
    return f"{entry.name}|{entry.head}" if entry.head else entry.name


def read_entries(path):
    """{name: Entry}, in file order; when a name appears twice the later line wins."""
    entries = {}
    path = Path(path)
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = parse_line(line)
            if entry is not None:
                entries[entry.name] = entry
    return entries


def completed_names(path):
    return {name for name, entry in read_entries(path).items() if not entry.in_progress}


def in_progress_entries(path):
    return [entry for entry in read_entries(path).values() if entry.in_progress]


def resume_state(path, name):
    """The in-progress entry of a repository, or None."""
    entry = read_entries(path).get(name)
    return entry if entry is not None and entry.in_progress else None


def _update(path, entry):
    """Replace the line of entry.name (or append one), atomically."""
    path = Path(path)
    lock_path = path.with_name(path.name + ".lock")
    with _THREAD_LOCK:  # released whatever happens below, a stuck lock would block every worker thread
        lock = None
        try:
            if fcntl is not None:
                lock = open(lock_path, "a")
                fcntl.flock(lock, fcntl.LOCK_EX)
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            new_line = format_entry(entry)
            replaced = False
            result = []
            for line in lines:
                parsed = parse_line(line)
                if parsed is not None and parsed.name == entry.name:
                    if not replaced:
                        result.append(new_line)
                        replaced = True
                    continue  # a second line of the same repository would only confuse readers
                result.append(line)
            if not replaced:
                result.append(new_line)

            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(result) + "\n")
            os.replace(tmp, path)
        finally:
            if lock is not None:
                try:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                finally:
                    lock.close()


def mark_started(path, name, head):
    """Record that a repository is being processed, unless it already has an in-progress entry."""
    if resume_state(path, name) is None:
        _update(path, Entry(name, head, "", True))


def mark_progress(path, name, head, marker):
    _update(path, Entry(name, head, marker, True))


def mark_done(path, name, head=""):
    _update(path, Entry(name, head, "", False))
