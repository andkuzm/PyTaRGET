"""tools/start: the whole opening of a repair run in ONE call, no model: applies the pure identifier renames found in SOURCE_CHANGES.diff to the tests (autorename), runs the tests, prints
what was replaced and the compact test result. usage: tools/start"""
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dtools import autorename, run_tests  # noqa: E402

ws = sys.argv[1]
os.chdir(ws)
buf = io.StringIO()
with redirect_stdout(buf): autorename.main()
print("AUTO-RENAMES (already applied to the tests; do not redo):\n" + buf.getvalue().strip())
print("\nTEST RESULT AFTER THE RENAMES:")
sys.exit(run_tests.main([ws]))
