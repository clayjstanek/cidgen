"""Assignment marginals and the soft trackers (JPDA, VB).

    python -m pytest tests/test_soft_trackers.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cidgen.trackers import make_tracker  # noqa: E402
from cidgen.trackers.assoc import bp_marginals, enumerate_marginals, exact_marginals  # noqa: E402
import test_gnn_tracker as T  # noqa: E402

CFG_SENS = {"clutter_rate": 2.0, "r_p_sector_halfwidth_deg": 60.0,
            "radar_types": {"R_S": {"ref_range_km": 75.0, "sector": "full"}}}


def test_exact_marginals_match_enumeration():
    rng = np.random.default_rng(0)
    for N, M in [(1, 1), (1, 4), (2, 3), (3, 4), (4, 3), (5, 5)]:
        W = rng.normal(0, 3, (N, M))
        G = exact_marginals(W)
        assert np.allclose(G, enumerate_marginals(W), atol=1e-10)
        assert np.allclose(G.sum(0), 1.0)
        assert (G[:N].sum(1) <= 1 + 1e-12).all()


def test_bp_close_to_exact():
    rng = np.random.default_rng(1)
    W = rng.normal(0, 3, (6, 7))
    assert np.abs(bp_marginals(W) - exact_marginals(W)).max() < 0.1


def test_soft_trackers_two_targets_with_clutter():
    for method in ("jpda", "vb"):
        r = T._reports(seed=1, clutter=5, pd_=0.8)
        r["range_m"] = 50000.0; r["el_deg"] = 1.0
        tr = make_tracker({"tracking": {"method": method}, "sensors": CFG_SENS}, method)
        est, rep, tracks = tr.run(r, np.arange(0.0, 120.0, 2.0))
        obj = (r.truth >= 0).values
        assert len(tracks) == 2, method
        assert (rep.values[~obj] < 0).mean() > 0.95, method
        late = est[est.t >= 80.0]
        sp = np.hypot(late.ve, late.vn).groupby(late.eid).mean()
        assert np.allclose(sp, 200.0, atol=20.0), method


def test_set_marginals_match_enumeration():
    import itertools
    from cidgen.trackers.assoc import enumerate_set_marginals, exact_set_marginals
    rng = np.random.default_rng(3)
    for _ in range(20):
        N, M = int(rng.integers(1, 5)), int(rng.integers(1, 4))
        ev = []
        for j in range(M):
            e = [(1 << i, rng.normal(0, 2)) for i in range(N)]
            for c in itertools.combinations(range(N), 2):
                if rng.random() < 0.5:
                    e.append((sum(1 << i for i in c), rng.normal(0, 2)))
            ev.append(e)
        P, Pc = exact_set_marginals(ev, N, M)
        Q, Qc = enumerate_set_marginals(ev, N, M)
        assert max(np.abs(p - q).max() for p, q in zip(P, Q)) < 1e-10
        assert np.abs(Pc - Qc).max() < 1e-10


def test_vbm_keeps_unresolved_members_as_separate_tracks():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import tracker_bench as B
    rows = pd.DataFrame(B.run_swarm(4, 60.0, 1, horizon=120.0, methods=("gnn", "vbm")))
    m = rows.set_index("method")
    assert m.loc["vbm", "tracks_per_object"] <= m.loc["gnn", "tracks_per_object"]
    assert m.loc["vbm", "own_track"] >= m.loc["gnn", "own_track"] - 0.02
