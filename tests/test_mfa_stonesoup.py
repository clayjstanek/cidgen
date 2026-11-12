"""Stone Soup MFA adapter (skipped when stonesoup / ortools are not installed).

    python -m pytest tests/test_mfa_stonesoup.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip("stonesoup")
pytest.importorskip("ortools")
from cidgen.trackers import make_tracker  # noqa: E402
import test_gnn_tracker as T  # noqa: E402

SENS = {"clutter_rate": 2.0, "r_p_sector_halfwidth_deg": 60.0,
        "radar_types": {"R_S": {"ref_range_km": 75.0, "sector": "full"}}}


@pytest.mark.parametrize("window", [3, 5])
def test_mfa_two_targets_with_clutter(window):
    r = T._reports(seed=1, clutter=5, pd_=0.8)
    r["range_m"] = 50000.0; r["el_deg"] = 1.0
    tr = make_tracker({"tracking": {"method": "mfa", "mfa": {"slide_window": window}}, "sensors": SENS}, "mfa")
    est, rep, tracks = tr.run(r, np.arange(0.0, 120.0, 2.0))
    obj = (r.truth >= 0).values
    assert len(tracks) == 2
    assert (rep.values[~obj] < 0).mean() > 0.95
    ct = pd.crosstab(r.truth.values[obj], rep.values[obj])
    assert (ct.gt(0).sum(axis=1) == 1).all()                 # one track per aircraft, no swaps
    late = est[est.t >= 80.0]
    sp = np.hypot(late.ve, late.vn).groupby(late.eid).mean()
    assert np.allclose(sp, 200.0, atol=20.0)
    assert tr.stats["mfa_calls"] > 0


def test_mfa_on_swarm_bench_runs():
    import tracker_bench as B
    rows = pd.DataFrame(B.run_swarm(4, 25.0, 1, horizon=90.0, methods=("gnn", "mfa")))
    assert set(rows.method) == {"gnn", "mfa"}
    assert rows.set_index("method").loc["mfa", "own_track"] > 0.3
