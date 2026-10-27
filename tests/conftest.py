"""Shared pytest fixtures and path setup for the repository_search test suite.

The production code mixes two import roots:

* ``repository_actions`` does ``from data_types.Broken_to_repaired import ...``
  (resolved from the project root) and ``from py_parser import ...``
  (resolved from inside ``repository_search/``).
* ``main_repository_miner`` does ``import repository_actions`` (also resolved
  from inside ``repository_search/``).

So both the project root *and* ``repository_search/`` must be importable.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPO_SEARCH = ROOT / "repository_search"

for p in (str(ROOT), str(REPO_SEARCH)):
    if p not in sys.path:
        sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
#  Git / network availability                                                 #
# --------------------------------------------------------------------------- #
def _git_present():
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


GIT_AVAILABLE = _git_present()


@pytest.fixture(scope="session")
def git_available():
    if not GIT_AVAILABLE:
        pytest.skip("git executable not available")
    return True


# A small, stable, well-known repository used to exercise the real git
# operations.  Pinned SHAs taken from its public, immutable history so the
# tests are deterministic.
SIX_REPO_URL = "https://github.com/benjaminp/six.git"
SIX_REPO_NAME = "benjaminp/six"

# Linear chain on master:  HEAD -> PARENT -> GRANDPARENT
SIX_SHA_HEAD = "65486e4383f9f411da95937451205d3c7b61b9e1"
SIX_SHA_PARENT = "1a4f325d55b10541ea94b889cda01e6281ba1689"
SIX_SHA_GRANDPARENT = "64601c7c016227aad81855031d88edb5cba17799"


@pytest.fixture(scope="session")
def six_clone(git_available, tmp_path_factory):
    """Clone the pinned well-known repo once per test session.

    Returns ``(repository_path, repository_name)`` such that the working tree
    lives at ``repository_path/six`` -- exactly the layout
    ``RepositoryActions`` expects (``repo_dir = repository_path/<repo-name>``).
    """
    base = tmp_path_factory.mktemp("six_session")
    dest = base / "six"
    result = subprocess.run(
        ["git", "clone", SIX_REPO_URL, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"network unavailable - git clone failed: {result.stderr.strip()}")
    # Make sure the pinned SHAs are actually present (full clone, not shallow).
    check = subprocess.run(
        ["git", "cat-file", "-e", SIX_SHA_HEAD],
        cwd=str(dest), capture_output=True,
    )
    if check.returncode != 0:
        pytest.skip("pinned commit not present in clone")
    return str(base), SIX_REPO_NAME


# --------------------------------------------------------------------------- #
#  Helpers for the pure / internal repository_actions tests                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_actions(tmp_path):
    """Factory that builds a RepositoryActions over a fake on-disk repo.

    ``files`` maps a repo-relative path -> file contents.  The fake repo is
    created at ``<tmp>/proj`` to match ``repository_name='owner/proj'``.
    """
    import repository_actions

    def _factory(files=None, repository_name="owner/proj"):
        repo_root = tmp_path
        work = repo_root / repository_name.split("/")[-1]
        work.mkdir(parents=True, exist_ok=True)
        for rel, content in (files or {}).items():
            target = work / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return repository_actions.RepositoryActions(
            repository_name=repository_name,
            repository_path=str(repo_root),
        )

    return _factory
