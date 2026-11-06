"""
Crash-safe writing for everything the evaluation pipeline saves (predictions, statistics, checkpoints).

A file written in place is half written when the job is killed in the middle of the write (Slurm time limit, node
failure, out of memory): the next run then finds a truncated JSON or a broken checkpoint and cannot continue. Here
every file is written to a temporary name next to its destination and moved over it with os.replace, which is atomic
on one filesystem, so the destination is always either the old complete content or the new complete content.
A directory (a model checkpoint) is swapped in the same way, with a recovery step for a kill between the two renames.
"""
import contextlib
import json
import os
import pickle
import shutil
import signal
import threading
from pathlib import Path


def _temporary(path):
    path = Path(path)
    return path.with_name(f"{path.name}.tmp-{os.getpid()}")


def remove_leftovers(path):
    """Delete temporary files / directories that a killed writer left next to `path`."""
    path = Path(path)
    for leftover in path.parent.glob(f"{path.name}.tmp-*"):
        if leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                leftover.unlink()


@contextlib.contextmanager
def atomic_path(path):
    """Yield a temporary path; on a clean exit it replaces `path`, on an error it is removed and `path` is untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        yield temporary
        with open(temporary, "rb") as written:
            os.fsync(written.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def atomic_write_text(path, text, encoding="utf-8"):
    with atomic_path(path) as temporary:
        temporary.write_text(text, encoding=encoding)


def atomic_write_json(path, obj, **dump_kwargs):
    atomic_write_text(path, json.dumps(obj, **dump_kwargs))


def atomic_dataframe_to_json(frame, path, **to_json_kwargs):
    """DataFrame.to_json(path, ...) that never leaves a half written file."""
    with atomic_path(path) as temporary:
        frame.to_json(temporary, **to_json_kwargs)


def atomic_pickle(obj, path):
    with atomic_path(path) as temporary:
        with open(temporary, "wb") as f:
            pickle.dump(obj, f)


def atomic_torch_save(obj, path):
    import torch
    with atomic_path(path) as temporary:
        torch.save(obj, temporary)


def _old(path):
    path = Path(path)
    return path.with_name(f"{path.name}.old")


def recover_directory(path):
    """
    Bring a directory written by atomic_directory back into a consistent state after a kill:
    a swap that stopped between its two renames left only `<path>.old`, which is the last complete version.
    """
    path = Path(path)
    old = _old(path)
    if not path.exists() and old.exists():
        os.replace(old, path)
    elif path.exists() and old.exists():
        shutil.rmtree(old, ignore_errors=True)      # the new version is in place, the old one is not needed
    remove_leftovers(path)


@contextlib.contextmanager
def atomic_directory(path):
    """
    Yield an empty temporary directory to fill; on a clean exit it replaces the directory `path`.
    The previous version is kept as `<path>.old` until the new one is in place (see recover_directory).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recover_directory(path)
    temporary = _temporary(path)
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()
    try:
        yield temporary
        old = _old(path)
        had_previous = path.exists()
        if had_previous:
            os.replace(path, old)
        try:
            os.replace(temporary, path)
        except BaseException:
            if had_previous:
                os.replace(old, path)       # the new version could not be put in place: back to the previous one
            raise
        shutil.rmtree(old, ignore_errors=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def atomic_save_pretrained(model, path):
    """model.save_pretrained(path) as an atomic directory swap."""
    with atomic_directory(path) as temporary:
        model.save_pretrained(temporary)


@contextlib.contextmanager
def sigterm_as_keyboard_interrupt():
    """
    Slurm ends a job with SIGTERM (then SIGKILL some seconds later). Python dies on SIGTERM without running any
    `finally` block; turning it into KeyboardInterrupt lets the code save what it has before the process ends.
    Only possible in the main thread, elsewhere this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
