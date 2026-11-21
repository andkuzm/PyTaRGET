"""tools/run_tests [Class#method ...]: run the tests offline and print a COMPACT result (a few lines; the full Maven log is in .delegation/last_test.log)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dtools import common


def main(argv):
    ws = Path(argv[0]); tests = argv[1:]
    rc, log = common.mvn(ws, tests or None)
    kind, data = common.result_of(ws, rc, log)
    print(common.compact(ws, kind, data, rc))
    return 0 if kind == "tests" and all(r[2] in ("pass", "skipped") for r in data) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
