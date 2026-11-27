"""Run a build/test command of a NOT-YET-TRUSTED project inside the mount sandbox (scripts/agent/sandbox_inner.sh): the command sees the project folder, its own Maven cache, the JDK and Maven,
and nothing else of the shared file system (no home, no tokens, no reference repair). tools/, .delegation/ and .git/ of the project are read-only for it, because the host-side tools read them.
Without network by default; network=True is only for filling the Maven cache. Used when the workspace config has "jail": true."""
import fcntl
import json
import shlex
import subprocess
import sys
from pathlib import Path

AGENT = Path(__file__).resolve().parents[2] / "agent"
CACHE = Path.home() / "delegation_exp" / "jail_software.json"


def software(java_home, maven_home):
    """folders the JDK, Maven and the agent Python need (about 40 s to compute, so it is cached)."""
    key = f"{java_home}|{maven_home}"
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(CACHE) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        if key not in data:
            sys.path.insert(0, str(AGENT))
            import runtime_paths
            data[key] = list(runtime_paths.needed(java_home, maven_home, str(Path.home() / "venv_agent")))
            CACHE.write_text(json.dumps(data, indent=1))
    return data[key]


def command(ws, cfg, cmd, network=False):
    """(argv, env) that runs cmd (a list) in the sandbox with ws as working directory."""
    ws = Path(ws).resolve(); out = ws.parent / "jail_out"; out.mkdir(parents=True, exist_ok=True); m2 = Path(cfg["m2"]); m2.mkdir(parents=True, exist_ok=True)
    env = {"HOME": str(Path.home()), "PATH": f"{cfg['java_home']}/bin:{cfg['maven_home']}/bin:/usr/bin:/bin", "JAVA_HOME": cfg["java_home"], "MAVEN_OPTS": "-Xmx1g", "LANG": "C.UTF-8", "TERM": "dumb",
           "SANDBOX_NARROW": "1", "SANDBOX_RO": " ".join(str(ws / d) for d in ("tools", ".delegation", ".git"))}              # a CLEAN environment: nothing of the caller's variables
    inner = ["unshare", "-m", str(AGENT / "sandbox_inner.sh"), str(ws), str(out), str(m2), *software(cfg["java_home"], cfg["maven_home"]), "--", *cmd]
    if network: return ["unshare", "-r", *inner], env
    return ["unshare", "-rn", "/usr/bin/bash", "-c", "/usr/sbin/ip link set lo up; exec " + shlex.join(inner)], env


def run(ws, cfg, cmd, network=False, timeout=1500):
    argv, env = command(ws, cfg, cmd, network)
    return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, errors="replace")
