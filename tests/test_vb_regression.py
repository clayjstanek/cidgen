"""Pinned VB results on a real scenario (slow: about 1-2 minutes; runs only when asked).

    Windows cmd:     set CIDGEN_SLOW=1
                     python -m pytest tests\\test_vb_regression.py -q
    PowerShell:      $env:CIDGEN_SLOW=1; python -m pytest tests\\test_vb_regression.py -q
    bash:            CIDGEN_SLOW=1 python -m pytest tests/test_vb_regression.py -q

Make the golden file first, once, from code you trust:  python tests/vb_golden.py   (see that file).
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vb_golden as VG  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("CIDGEN_SLOW", "").strip() != "1",
                                reason="slow: set CIDGEN_SLOW=1 to run the pinned S12 VB regression")


def test_vb_s12_matches_golden():
    p = VG.golden_path(**VG.DEFAULT)
    if not p.exists():
        pytest.skip(f"no golden file yet: run  python tests/vb_golden.py  to make {p.name}")
    old = json.loads(p.read_text())
    new = VG.compute(**VG.DEFAULT)
    if new["config_fingerprint"] != old["config_fingerprint"]:
        pytest.skip("scenario configuration changed since the golden file was made: re-make it "
                    "(python tests/vb_golden.py) if the change was intended")
    diffs = VG.compare(new, old)
    assert not diffs, "VB results moved:\n  " + "\n  ".join(diffs)
