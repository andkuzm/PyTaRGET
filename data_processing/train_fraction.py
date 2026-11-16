"""
Training-set size experiment of the TaRGet paper (section 4.7, RQ3.1): fine-tuning subsets with 20%, 40%, 60% and 80% of the
training data, "achieved by excluding older data, for each project, based on commit dates", with the same validation and test sets.
100% is the full training set (in our per-project split about 80% of every project's cases).

select_recent_per_project(ds, fraction) returns the cases to keep so that the TOTAL kept share is as close as possible to `fraction`:
every project keeps its NEWEST commits (a commit is never split) and always at least its newest commit, so no project disappears
and every test project stays seen at training time. A fixed per-project share does not work for our data (458 small projects: the
one-commit floor alone would turn a request for 20% into 35%), so the per-project share g is calibrated by bisection; the projects
that change between the two neighbouring shares are then switched one at a time (fixed order) until the total is closest to the request. Subsets are nested: the cases kept at 0.4 are also kept at 0.6 (the kept count per project grows
with g). A request below the floor (one newest commit per project) returns that floor.
"""
import hashlib

import pandas as pd


def _projects(ds):
    """Per project: its commits oldest first as a list of (size, [row positions]); positions index the rows of ds."""
    frame = pd.DataFrame({"repo": ds["repository_name"].to_numpy(), "commit": ds["broken_hash"].to_numpy(), "pos": range(len(ds))})
    if "broken_date" in ds.columns:
        frame["time"] = pd.to_datetime(ds["broken_date"].to_numpy(), utc=True)
    else:                                   # no dates: the order of the rows is taken as time order
        frame["time"] = pd.to_datetime(frame["pos"], unit="s", utc=True)
    projects = []
    for _, project in frame.groupby("repo", sort=False):
        commits = (project.groupby("commit", sort=False)
                   .agg(time=("time", "min"), first=("pos", "min"), size=("pos", "size"), positions=("pos", list))
                   .sort_values(["time", "first"], kind="stable"))
        projects.append(list(zip(commits["size"].tolist(), commits["positions"].tolist())))
    return projects


def _dropped_commits(sizes, share):
    """How many of the oldest commits to drop so that the kept count is closest to share * total (at least one commit stays;
    a tie keeps more data)."""
    total = sum(sizes)
    target = share * total
    best_drop, best_gap, remaining = 0, abs(total - target), total
    for drop in range(1, len(sizes)):
        remaining -= sizes[drop - 1]
        gap = abs(remaining - target)
        if gap < best_gap:
            best_drop, best_gap = drop, gap
    return best_drop


def _kept_count(projects, share):
    return sum(sum(size for size, _ in commits[_dropped_commits([s for s, _ in commits], share):]) for commits in projects)


def select_recent_per_project(ds, fraction):
    """
    ds: DataFrame with the columns repository_name, broken_hash and (optional) broken_date, one row per training case, in any order.
    Returns the sorted list of row POSITIONS (0..len(ds)-1) to keep.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return list(range(len(ds)))
    projects = _projects(ds)
    target = fraction * len(ds)
    sizes = [[s_ for s_, _ in commits] for commits in projects]
    # the kept total grows with the per-project share g (a step function): find the two neighbouring shares with a total below
    # and at/above the target
    g_low, g_high = 0.0, 1.0
    for _ in range(50):
        g = (g_low + g_high) / 2
        if _kept_count(projects, g) < target:
            g_low = g
        else:
            g_high = g
    drops_low = [_dropped_commits(sz, g_low) for sz in sizes]       # more commits dropped
    drops_high = [_dropped_commits(sz, g_high) for sz in sizes]     # fewer dropped
    kept_of = lambda commits, drop: sum(size for size, _ in commits[drop:])
    total = sum(kept_of(c, d) for c, d in zip(projects, drops_low))
    # projects that change between the two shares are switched one by one in a fixed pseudo-random order (a hash of the project
    # name, so the order does not depend on the data order) until the total is closest to the target; a tie keeps more data
    flippable = sorted((i for i in range(len(projects)) if drops_low[i] != drops_high[i]),
                       key=lambda i: hashlib.md5(str(i_name(ds, projects, i)).encode()).hexdigest())
    best_flips, best_gap, running = 0, abs(total - target), total
    for number, i in enumerate(flippable, start=1):
        running += kept_of(projects[i], drops_high[i]) - kept_of(projects[i], drops_low[i])
        gap = abs(running - target)
        if gap <= best_gap:
            best_flips, best_gap = number, gap
    drops = list(drops_low)
    for i in flippable[:best_flips]:
        drops[i] = drops_high[i]
    keep = []
    for commits, drop in zip(projects, drops):
        for _, positions in commits[drop:]:
            keep.extend(positions)
    return sorted(keep)


def i_name(ds, projects, i):
    """A stable name for project i: the repository of its first row."""
    return ds["repository_name"].iloc[projects[i][0][1][0]]


def describe(ds, kept_positions):
    """Numbers for the report: cases, projects and the share kept."""
    kept = ds.iloc[kept_positions]
    return {"cases_before": len(ds), "cases_kept": len(kept), "share_kept": round(len(kept) / max(len(ds), 1), 4),
            "projects": int(ds["repository_name"].nunique()), "projects_kept": int(kept["repository_name"].nunique()),
            "commits_before": int(ds["broken_hash"].nunique()), "commits_kept": int(kept["broken_hash"].nunique())}
