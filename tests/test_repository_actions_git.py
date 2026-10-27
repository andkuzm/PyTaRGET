"""Real git-operation tests for repository_search/repository_actions.py.

These exercise clone / checkout / commit-navigation against a small, well-known
repository (benjaminp/six) pinned to stable, immutable commit SHAs.  They are
skipped automatically when git or the network is unavailable.

``clone_repository_last`` bundles a ``pip install .`` step that would mutate the
outer interpreter's environment; we intercept *only* that pip call so the real
git clone + HEAD-resolution logic still runs, without polluting the env.
"""

import subprocess
from pathlib import Path

import pytest

import repository_actions
from conftest import (
    SIX_REPO_NAME,
    SIX_REPO_URL,
    SIX_SHA_GRANDPARENT,
    SIX_SHA_HEAD,
    SIX_SHA_PARENT,
)


def _head_sha(repo_dir):
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_dir),
        capture_output=True, text=True,
    )
    return out.stdout.strip()


@pytest.fixture
def actions(six_clone):
    repository_path, repository_name = six_clone
    a = repository_actions.RepositoryActions(repository_name, repository_path)
    # Deterministic starting point for every test.
    assert a.git_checkout_with_retry(str(a.repo_dir), SIX_SHA_HEAD)
    a.current_hash = SIX_SHA_HEAD
    a.visited_commits = {SIX_SHA_HEAD}
    return a


# --------------------------------------------------------------------------- #
#  clone_repository_last (git parts only; pip install intercepted)            #
# --------------------------------------------------------------------------- #
class TestCloneRepositoryLast:
    def test_clone_sets_repo_dir_and_hash(self, git_available, tmp_path, monkeypatch):
        # Network probe: skip cleanly if we cannot reach the remote.
        probe = subprocess.run(["git", "ls-remote", SIX_REPO_URL, "HEAD"],
                                capture_output=True, text=True)
        if probe.returncode != 0:
            pytest.skip("network unavailable for git ls-remote")

        real_run = subprocess.run

        def fake_run(cmd, *args, **kwargs):
            # Intercept the `pip install .` that clone_repository_last performs
            # so it does not touch the outer environment.
            if any("pip" == str(c) or str(c).endswith("pip") for c in cmd) or "install" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(repository_actions.subprocess, "run", fake_run)

        a = repository_actions.RepositoryActions(SIX_REPO_NAME, str(tmp_path))
        dest = a.clone_repository_last()

        assert Path(dest).exists()
        assert (Path(dest) / ".git").exists()
        assert len(a.current_hash) == 40
        assert a.current_hash in a.visited_commits
        assert a.current_hash == _head_sha(dest)


# --------------------------------------------------------------------------- #
#  checkout_commit / git_checkout_with_retry                                  #
# --------------------------------------------------------------------------- #
class TestCheckout:
    def test_checkout_commit_moves_head(self, actions):
        assert actions.checkout_commit(SIX_SHA_PARENT, str(actions.repo_dir)) is True
        assert _head_sha(actions.repo_dir) == SIX_SHA_PARENT

    def test_checkout_commit_bad_hash_returns_false(self, actions):
        assert actions.checkout_commit("0" * 40, str(actions.repo_dir)) is False

    def test_checkout_with_retry_success(self, actions):
        assert actions.git_checkout_with_retry(str(actions.repo_dir), SIX_SHA_GRANDPARENT) is True
        assert _head_sha(actions.repo_dir) == SIX_SHA_GRANDPARENT

    def test_checkout_with_retry_failure_is_bounded(self, actions, monkeypatch):
        # Don't actually sleep through the exponential backoff.
        monkeypatch.setattr(repository_actions.time, "sleep", lambda *_: None)
        assert actions.git_checkout_with_retry(str(actions.repo_dir), "0" * 40, retries=2) is False


# --------------------------------------------------------------------------- #
#  commit navigation                                                          #
# --------------------------------------------------------------------------- #
class TestCommitNavigation:
    def test_move_to_earlier_commit_follows_parent(self, actions):
        result = actions.move_to_earlier_commit()
        assert result == SIX_SHA_PARENT
        assert actions.current_hash == SIX_SHA_PARENT
        assert actions.previous_hash == SIX_SHA_HEAD
        assert _head_sha(actions.repo_dir) == SIX_SHA_PARENT

    def test_move_to_earlier_detects_cycle(self, actions):
        # Pretend the parent was already visited -> refuse to move.
        actions.visited_commits.add(SIX_SHA_PARENT)
        assert actions.move_to_earlier_commit() == "Error"

    def test_move_to_earlier_respects_max_commits(self, actions):
        actions.commit_counter = 500
        assert actions.move_to_earlier_commit() == "Error"

    def test_move_to_later_returns_to_child(self, actions):
        assert actions.move_to_earlier_commit() == SIX_SHA_PARENT
        assert actions.move_to_later_commit() == SIX_SHA_HEAD
        assert _head_sha(actions.repo_dir) == SIX_SHA_HEAD

    def test_move_to_later_without_previous_is_error(self, actions):
        actions.previous_hash = None
        assert actions.move_to_later_commit() == "Error"


# --------------------------------------------------------------------------- #
#  test discovery against the real checked-out tree                           #
# --------------------------------------------------------------------------- #
class TestDiscoveryOnRealTree:
    def test_has_tests(self, actions):
        actions.checkout_commit(SIX_SHA_PARENT, str(actions.repo_dir))
        assert actions.has_tests() is True

    def test_list_test_files_includes_test_six(self, actions):
        actions.checkout_commit(SIX_SHA_PARENT, str(actions.repo_dir))
        names = {p.name for p in actions.list_test_files()}
        assert "test_six.py" in names

    def test_find_test_methods_on_real_file(self, actions):
        actions.checkout_commit(SIX_SHA_PARENT, str(actions.repo_dir))
        methods = actions.find_test_methods("test_six.py")
        flat = {m[1] for m in methods}
        assert "test_add_doc" in flat
        assert "test_integer_types" in flat

    def test_extract_method_code_on_real_file(self, actions):
        actions.checkout_commit(SIX_SHA_PARENT, str(actions.repo_dir))
        code = actions.extract_method_code("test_six.py", "test_add_doc")
        assert "def test_add_doc():" in code
