"""GNN tracker unit tests (no scenario needed).

    python -m pytest tests/test_gnn_tracker.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cidgen.trackers import GNNTracker  # noqa: E402

CFG = {"tracking": {"method": "gnn"}}


def _reports(seed=0, clutter=0, pd_=1.0, horizon=120.0):
    """Two Tier 2 aircraft on crossing paths seen by one radar every 2 s, sigma 100 m, plus clutter."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in np.arange(0.0, horizon, 2.0):
        for k, (p0, v) in enumerate([((-20000, 0, 5000), (200, 0, 0)), ((0, -20000, 6000), (0, 200, 0))]):
            if rng.random() < pd_:
                z = np.array(p0) + np.array(v) * t + rng.normal(0, 100, 3)
                rows.append((t, "R_S-1", "R_S", *z, 10.0, 0.0, k))
        for _ in range(rng.poisson(clutter)):
            z = rng.uniform([-30000, -30000, 0], [30000, 30000, 1000])
            rows.append((t, "R_S-1", "R_S", *z, -20.0, 0.0, -1))
    r = pd.DataFrame(rows, columns=["time", "sensor_id", "sensor_type", "e", "n", "u", "rcs_dbsm", "vr_mps", "truth"])
    r["R"] = [np.eye(3) * 100.0 ** 2] * len(r)
    return r


def test_two_crossing_targets_two_tracks():
    r = _reports()
    est, rep, tracks = GNNTracker(CFG).run(r, np.arange(0.0, 120.0, 2.0))
    assert len(tracks) == 2
    # every report of one object went to one track, and the two objects to different tracks
    m = pd.crosstab(r.truth, rep)
    assert (m.gt(0).sum(axis=1) == 1).all() and m.gt(0).sum(axis=0).eq(1).all()
    # speed estimate close to truth over the last 40 s (Tier 2 filter: q = 30, so it is noisy point to point)
    late = est[est.t >= 80.0]
    sp = np.hypot(late.ve, late.vn).groupby(late.eid).mean()
    assert np.allclose(sp, 200.0, atol=20.0)


def test_clutter_does_not_confirm_and_misses_are_coasted():
    r = _reports(seed=1, clutter=5, pd_=0.8)
    est, rep, tracks = GNNTracker(CFG).run(r, np.arange(0.0, 120.0, 2.0))
    fa = rep[r.truth < 0]
    assert (fa < 0).mean() > 0.95                      # clutter reports almost never end in a confirmed track
    obj = r.truth >= 0
    assert pd.crosstab(r.truth[obj], rep[obj]).gt(0).sum(axis=1).max() <= 2   # at most one break per object


def test_label_mode_is_default():
    from cidgen.config import load_config
    assert (load_config("S12").get("tracking") or {}).get("method", "label") == "label"
