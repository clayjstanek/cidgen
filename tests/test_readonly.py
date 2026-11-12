"""Regression test: Parquet/Arrow-backed pandas frames hand out read-only arrays. Emulate that and
build every snapshot of a short run.   python tests/test_readonly.py
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Emulate Parquet/Arrow-backed frames: every array pandas hands out is read-only unless explicitly copied
_orig_to_numpy = pd.DataFrame.to_numpy
def ro_to_numpy(self, dtype=None, copy=False, na_value=pd.api.extensions.no_default):
    a = _orig_to_numpy(self, dtype=dtype, copy=True, na_value=na_value)
    if not copy: a.flags.writeable = False
    return a
pd.DataFrame.to_numpy = ro_to_numpy
pd.DataFrame.values = property(lambda self: ro_to_numpy(self))
_s_orig = pd.Series.to_numpy
def ro_s(self, dtype=None, copy=False, na_value=pd.api.extensions.no_default, **kw):
    a = _s_orig(self, dtype=dtype, copy=True, na_value=na_value)
    if not copy: a.flags.writeable = False
    return a
pd.Series.to_numpy = ro_s
pd.Series.values = property(lambda self: ro_s(self))
import tempfile
from cidgen.generate import run_scenario
from cidgen.graph_export import RunGraphs

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        run = run_scenario("S0", seed=5, out_root=td, overrides={"run": {"horizon_s": 300}}, verbose=False)
        rg = RunGraphs(run)
        for t in rg.times:
            rg.snapshot(float(t))
    print(f"READ-ONLY TEST PASSED ({len(rg.times)} snapshots)")
