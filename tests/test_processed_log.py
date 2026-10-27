"""Tests for repository_search/processed_log.py, the format of the processed-repositories file."""
import processed_log as log


def test_parse_the_three_line_formats():
    assert log.parse_line("owner/old") == log.Entry("owner/old", "", "", False)
    assert log.parse_line("owner/done|abc\n") == log.Entry("owner/done", "abc", "", False)
    assert log.parse_line("owner/busy|abc|def") == log.Entry("owner/busy", "abc", "def", True)
    assert log.parse_line("owner/new|abc|") == log.Entry("owner/new", "abc", "", True)
    assert log.parse_line("  \n") is None


def test_old_files_are_read_as_completed(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("legacy/blacklisted\nowner/done|abc\n", encoding="utf-8")
    assert log.completed_names(path) == {"legacy/blacklisted", "owner/done"}
    assert log.in_progress_entries(path) == []
    assert log.read_entries(tmp_path / "missing.txt") == {}


def test_started_progress_done_cycle_replaces_the_line_in_place(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("a/one|h1\nb/two|h2\n", encoding="utf-8")

    log.mark_started(path, "c/three", "h3")
    assert path.read_text(encoding="utf-8").splitlines() == ["a/one|h1", "b/two|h2", "c/three|h3|"]

    log.mark_progress(path, "c/three", "h3", "commit1")
    log.mark_progress(path, "c/three", "h3", "commit2")
    assert path.read_text(encoding="utf-8").splitlines() == ["a/one|h1", "b/two|h2", "c/three|h3|commit2"]
    assert log.resume_state(path, "c/three").marker == "commit2"

    log.mark_done(path, "c/three", "h3")
    assert path.read_text(encoding="utf-8").splitlines() == ["a/one|h1", "b/two|h2", "c/three|h3"]
    assert log.resume_state(path, "c/three") is None


def test_started_never_overwrites_existing_progress(tmp_path):
    path = tmp_path / "p.txt"
    log.mark_progress(path, "a/one", "h1", "commit5")
    log.mark_started(path, "a/one", "other-head")
    assert log.resume_state(path, "a/one") == log.Entry("a/one", "h1", "commit5", True)


def test_other_repositories_are_untouched_and_no_temporary_files_remain(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("a/one|h1\nb/two|h2|m\n", encoding="utf-8")
    log.mark_progress(path, "b/two", "h2", "m2")
    assert path.read_text(encoding="utf-8").splitlines() == ["a/one|h1", "b/two|h2|m2"]
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_a_duplicated_repository_collapses_to_one_line(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("a/one|h1\nb/two|h2\na/one|h1|m\n", encoding="utf-8")
    log.mark_done(path, "a/one", "h1")
    assert path.read_text(encoding="utf-8").splitlines() == ["a/one|h1", "b/two|h2"]


def test_a_failing_file_lock_never_leaves_the_thread_lock_held(tmp_path, monkeypatch):
    import threading

    class BrokenFcntl:
        LOCK_EX, LOCK_UN = 2, 8

        def flock(self, handle, operation):
            if operation == self.LOCK_UN:
                raise OSError("cannot unlock")

    path = tmp_path / "p.txt"
    monkeypatch.setattr(log, "fcntl", BrokenFcntl())
    try:
        log.mark_done(path, "a/one", "h1")
    except OSError:
        pass

    monkeypatch.setattr(log, "fcntl", None)  # the next update must not wait for a lock nobody releases
    worker = threading.Thread(target=log.mark_done, args=(path, "b/two", "h2"))
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    assert "b/two|h2" in path.read_text(encoding="utf-8").splitlines()


def test_concurrent_updates_from_many_threads_are_all_kept(tmp_path):
    import threading
    path = tmp_path / "p.txt"
    threads = [threading.Thread(target=log.mark_done, args=(path, f"o/r{i}", f"h{i}")) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(line.split("|")[0] for line in path.read_text(encoding="utf-8").splitlines()) == sorted(
        f"o/r{i}" for i in range(30))
