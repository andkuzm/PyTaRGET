"""The cluster software folders the sandboxed agent really needs (fix D1): instead of the two whole software trees, the install folder of the JDK, of Maven and of the agent's Python, plus every folder
their shared libraries (ldd) and their symbolic links point to under /gpfs. usage: runtime_paths.py <java_home> <maven_home> <venv> [...]   prints one folder per line"""
import os
import re
import subprocess
import sys
from pathlib import Path

CLEAN = {"PATH": "/usr/bin:/bin", "LANG": "C"}                                      # ldd must resolve as inside the sandbox, which gets a clean environment
SPACK = re.compile(r"^(.*?/[^/]+-[a-z0-9]{32})(?=/|$)")                              # a spack install folder ends in -<32 character hash>


def shared_objects(root, limit=4000):
    out = []
    for d, _, files in os.walk(root):
        out += [os.path.join(d, f) for f in files if f.endswith(".so") or ".so." in f]
        if len(out) >= limit: break
    return out[:limit]


def prefix_of(path):
    m = SPACK.match(path)
    if m: return m.group(1)
    p = Path(path).parent
    return str(p.parent if p.name in ("lib", "lib64", "bin") else p)


def needed(java_home, maven_home, venv):
    python = os.path.realpath(os.path.join(venv, "bin", "python"))
    homes = [os.path.realpath(java_home), os.path.realpath(maven_home), os.path.dirname(os.path.dirname(python))]
    keep = set(homes) | {java_home.rstrip("/"), maven_home.rstrip("/")}
    binaries = [python, os.path.join(java_home, "bin", "java")] + shared_objects(os.path.join(homes[0], "lib")) + shared_objects(os.path.join(homes[2], "lib")) + shared_objects(os.path.join(venv, "lib"))
    for i in range(0, len(binaries), 200):
        r = subprocess.run(["ldd", *binaries[i:i + 200]], capture_output=True, text=True, env=CLEAN)
        for lib in re.findall(r"=>\s+(/gpfs/\S+)", r.stdout):
            for p in (lib, os.path.realpath(lib)):
                keep.add(prefix_of(p))
    for home in homes:                                                               # symbolic links that leave the install folder
        for d, dirs, files in os.walk(home):
            for n in dirs + files:
                p = os.path.join(d, n)
                if os.path.islink(p):
                    t = os.path.realpath(p)
                    if t.startswith("/gpfs/") and os.path.exists(t) and not any(t == h or t.startswith(h + "/") for h in homes):
                        keep.add(prefix_of(t))
    keep = sorted(p for p in keep if p.startswith("/gpfs/") and os.path.isdir(p) and not p.startswith(str(Path.home()) + "/") and p != str(Path.home()))
    return [p for p in keep if not any(p != q and p.startswith(q + "/") for q in keep)]    # drop folders already inside another one


if __name__ == "__main__":
    print("\n".join(needed(*sys.argv[1:4])))
