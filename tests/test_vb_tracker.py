"""VB tracker tests: the window smoother, causality, and the shared track management.

    python -m pytest tests/test_vb_tracker.py -q

What each group checks:

  pseudo-measurement   _pseudo (the information-weighted soft measurement of one look) has the identities the
                       variational update relies on.
  window smoother      soft_rts (the VB window step for one track: soft Kalman filter from the anchor, then RTS)
                       against an independent batch solution of the same linear-Gaussian problem, solved in
                       information form over all states at once:
                         - certain associations (one-hot beta)        = a standard Kalman filter + RTS smoother
                         - missed looks (beta = 0)                    = no measurement at that look
                         - beta = b on one report                     = that report with noise R / b
                         - beta spread over several reports           = all of them, each with noise R_j / b_j
  causality            what the tracker outputs at a grid time never depends on reports after that time
                       (the window is a fixed-lag re-estimate, not hindsight), and a run is deterministic.
  track management     VB starts, confirms and deletes tracks exactly as the GNN does (the shared code path);
                       isolated tracks are not re-solved by the window (solve_isolated False).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cidgen.trackers import GNNTracker, make_tracker  # noqa: E402
from cidgen.trackers.gnn import _q_block  # noqa: E402
from cidgen.trackers.soft import _pseudo, soft_rts  # noqa: E402
import test_gnn_tracker as G  # noqa: E402

CFG_SENS = {"clutter_rate": 2.0, "r_p_sector_halfwidth_deg": 60.0,
            "radar_types": {"R_S": {"ref_range_km": 75.0, "sector": "full"}}}
H = np.hstack([np.eye(3), np.zeros((3, 3))])


# ---------------------------------------------------------------- helpers
def _spd(rng, scale):
    A = rng.normal(0, 1, (3, 3))
    return scale ** 2 * (A @ A.T / 3 + np.eye(3))


def _problem(seed, n_looks=12, max_reports=3):
    """One track's window: anchor, irregular look times, 1..max_reports reports per look with their own R."""
    rng = np.random.default_rng(seed)
    ta = 100.0
    xa = np.r_[rng.normal(0, 5000, 3), rng.normal(0, 200, 3)]
    Pa = np.diag([300.0 ** 2] * 3 + [40.0 ** 2] * 3)
    times = ta + np.cumsum(rng.uniform(0.5, 4.0, n_looks))
    looks = []
    for t in times:
        m = int(rng.integers(1, max_reports + 1))
        Z = xa[:3] + xa[3:] * (t - ta) + rng.normal(0, 150, (m, 3))
        R = np.stack([_spd(rng, 100.0) for _ in range(m)])
        looks.append((t, Z, R))
    return ta, xa, Pa, looks, 30.0


def _steps(looks, betas):
    return [(t, b, Z, R, np.linalg.inv(R)) for (t, Z, R), b in zip(looks, betas)]


def _batch(ta, xa, Pa, looks, q, meas):
    """Independent solution: the joint Gaussian posterior of the anchor and every look's state, in
    information form. meas[k] is a list of (z, R) used at look k (empty = missed). Returns (means, covs)
    for the look states (not the anchor)."""
    K = len(looks)
    n = 6 * (K + 1)
    L = np.zeros((n, n)); eta = np.zeros(n)
    Pai = np.linalg.inv(Pa)
    L[:6, :6] += Pai; eta[:6] += Pai @ xa
    tp = ta
    for k, (t, _, _) in enumerate(looks, start=1):
        dt = t - tp; tp = t
        F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
        Qi = np.linalg.inv(_q_block(np.array([q]), np.array([dt]))[0])
        A = np.zeros((6, n)); A[:, 6 * (k - 1):6 * k] = -F; A[:, 6 * k:6 * (k + 1)] = np.eye(6)
        L += A.T @ Qi @ A
        for z, R in meas[k - 1]:
            Ri = np.linalg.inv(R)
            L[6 * k:6 * k + 6, 6 * k:6 * k + 6] += H.T @ Ri @ H
            eta[6 * k:6 * k + 6] += H.T @ Ri @ z
    C = np.linalg.inv(L); m = C @ eta
    return [m[6 * k:6 * k + 6] for k in range(1, K + 1)], [C[6 * k:6 * k + 6, 6 * k:6 * k + 6] for k in range(1, K + 1)]


def _compare(new, looks, means, covs):
    for k, (t, _, _) in enumerate(looks):
        x, P = new[t]
        assert np.allclose(x, means[k], rtol=0, atol=1e-6 * max(1.0, np.abs(means[k]).max())), t
        assert np.allclose(P, covs[k], rtol=1e-7, atol=1e-6), t


# ---------------------------------------------------------------- pseudo-measurement
def test_pseudo_measurement_identities():
    rng = np.random.default_rng(0)
    Z = rng.normal(0, 1000, (3, 3)); R = np.stack([_spd(rng, 100.0) for _ in range(3)])
    z, Rt = _pseudo(np.array([0.0, 1.0, 0.0]), Z, R)                 # certain: the report itself
    assert np.allclose(z, Z[1]) and np.allclose(Rt, R[1])
    z, Rt = _pseudo(np.array([0.0, 0.25, 0.0]), Z, R)                # weight b: same report, noise R / b
    assert np.allclose(z, Z[1]) and np.allclose(Rt, R[1] / 0.25)
    Rs = np.stack([R[0]] * 2); Zs = Z[:2]
    z, Rt = _pseudo(np.array([0.5, 0.5]), Zs, Rs)                   # equal split, equal noise: midpoint, noise R
    assert np.allclose(z, Zs.mean(0)) and np.allclose(Rt, R[0])
    _, Rt = _pseudo(np.array([0.3, 0.3, 0.3]), Z, R)                 # symmetric, positive definite
    assert np.allclose(Rt, Rt.T) and (np.linalg.eigvalsh(Rt) > 0).all()


# ---------------------------------------------------------------- window smoother against the batch solution
@pytest.mark.parametrize("seed", range(5))
def test_soft_rts_certain_associations_equal_batch_smoother(seed):
    ta, xa, Pa, looks, q = _problem(seed)
    rng = np.random.default_rng(100 + seed)
    pick = [int(rng.integers(len(Z))) for _, Z, _ in looks]
    betas = [np.eye(len(Z))[j] for (_, Z, _), j in zip(looks, pick)]
    new = soft_rts(ta, xa, Pa, _steps(looks, betas), q)
    means, covs = _batch(ta, xa, Pa, looks, q, [[(Z[j], R[j])] for (_, Z, R), j in zip(looks, pick)])
    _compare(new, looks, means, covs)


def test_soft_rts_missed_looks_are_predictions():
    ta, xa, Pa, looks, q = _problem(7, n_looks=15)
    missed = {2, 3, 4, 9, 14}                                          # includes the last look
    betas = [np.zeros(len(Z)) if k in missed else np.eye(len(Z))[0] for k, (_, Z, _) in enumerate(looks)]
    new = soft_rts(ta, xa, Pa, _steps(looks, betas), q)
    means, covs = _batch(ta, xa, Pa, looks, q,
                         [[] if k in missed else [(Z[0], R[0])] for k, (_, Z, R) in enumerate(looks)])
    _compare(new, looks, means, covs)


@pytest.mark.parametrize("seed", range(3))
def test_soft_rts_fractional_and_spread_beta(seed):
    """beta = b_j over several reports is the same as using every report with noise R_j / b_j."""
    ta, xa, Pa, looks, q = _problem(20 + seed, max_reports=4)
    rng = np.random.default_rng(200 + seed)
    betas, meas = [], []
    for _, Z, R in looks:
        b = rng.dirichlet(np.ones(len(Z) + 1))[:len(Z)]                # last share = "missed or clutter"
        b[rng.random(len(Z)) < 0.3] = 0.0                               # some reports outside the gate
        betas.append(b)
        meas.append([(Z[j], R[j] / b[j]) for j in range(len(Z)) if b[j] > 0])
    new = soft_rts(ta, xa, Pa, _steps(looks, betas), q)
    means, covs = _batch(ta, xa, Pa, looks, q, meas)
    _compare(new, looks, means, covs)


def test_soft_rts_last_look_is_the_filtered_state():
    """The newest look's smoothed state (where the tracker restarts) is the soft filter's own estimate."""
    ta, xa, Pa, looks, q = _problem(3)
    betas = [np.eye(len(Z))[0] for _, Z, _ in looks]
    full = soft_rts(ta, xa, Pa, _steps(looks, betas), q)
    for k in range(1, len(looks)):
        part = soft_rts(ta, xa, Pa, _steps(looks[:k], betas[:k]), q)  # filtering up to look k ...
        t = looks[k - 1][0]
        means, _ = _batch(ta, xa, Pa, looks[:k], q, [[(Z[0], R[0])] for _, Z, R in looks[:k]])
        assert np.allclose(part[t][0], means[-1], atol=1e-6 * np.abs(means[-1]).max())
    assert set(full) == {t for t, _, _ in looks}


# ---------------------------------------------------------------- tracker level: causality and determinism
def _vb(cfg_extra=None):
    cfg = {"tracking": {"method": "vb", **(cfg_extra or {})}, "sensors": CFG_SENS}
    return make_tracker(cfg, "vb")


def _crossing(seed, clutter=5, pd_=0.8, horizon=120.0):
    r = G._reports(seed=seed, clutter=clutter, pd_=pd_, horizon=horizon)
    r["range_m"] = 50000.0; r["el_deg"] = 1.0
    return r


def test_vb_output_never_uses_future_reports():
    T_CUT = 60.0
    r1 = _pair(1)                                                   # close pair: the window is active
    r2 = _pair(2)                                                   # different noise, clutter and misses ...
    r_mix = pd.concat([r1[r1.time <= T_CUT], r2[r2.time > T_CUT]], ignore_index=True)   # ... after T_CUT only
    grid = np.arange(0.0, 120.0, 2.0)
    tr = _vb(); e1, _, _ = tr.run(r1, grid)
    assert tr.stats["window_solves"] > 0
    e2, _, _ = _vb().run(r_mix, grid)
    tr3 = _vb(); e3, _, _ = tr3.run(r1[r1.time <= T_CUT].reset_index(drop=True), grid[grid <= T_CUT])
    assert tr3.stats["window_solves"] > 0                           # the window was re-solving before T_CUT
    a = e1[e1.t <= T_CUT].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, e2[e2.t <= T_CUT].reset_index(drop=True), check_exact=True)
    pd.testing.assert_frame_equal(a, e3.reset_index(drop=True), check_exact=True)
    assert len(a) > 20 and (e1.t > T_CUT).any()                    # the test is not vacuous


def _pair(seed, sep_m=200.0, clutter=2, pd_=0.9, horizon=120.0):
    """Two aircraft side by side, sep_m apart (2 sigma of the radar), flying north at 200 m/s."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in np.arange(0.0, horizon, 2.0):
        for k in range(2):
            if rng.random() < pd_:
                z = np.array([k * sep_m, -12000.0 + 200.0 * t, 5000.0]) + rng.normal(0, 100, 3)
                rows.append((t, "R_S-1", "R_S", *z, 10.0, 0.0, k))
        for _ in range(rng.poisson(clutter)):
            rows.append((t, "R_S-1", "R_S", *rng.uniform([-30000, -30000, 0], [30000, 30000, 1000]), -20.0, 0.0, -1))
    r = pd.DataFrame(rows, columns=["time", "sensor_id", "sensor_type", "e", "n", "u", "rcs_dbsm", "vr_mps", "truth"])
    r["R"] = [np.eye(3) * 100.0 ** 2] * len(r)
    r["range_m"] = 50000.0; r["el_deg"] = 1.0
    return r


def test_vb_window_runs_and_changes_estimates_where_tracks_share_reports():
    """With tracks close enough to share reports, the window solves clusters and moves the estimates;
    iterations 0 (filter step only) gives a different output."""
    r = _pair(1)
    grid = np.arange(0.0, 120.0, 2.0)
    tr = _vb(); e_w, _, _ = tr.run(r, grid)
    e_f, _, _ = _vb({"soft": {"iterations": 0}}).run(r, grid)
    assert tr.stats["window_solves"] > 0
    assert not e_w[["e", "n", "u"]].equals(e_f[["e", "n", "u"]])


def test_vb_is_deterministic():
    r = _pair(4)
    grid = np.arange(0.0, 120.0, 2.0)
    a = _vb().run(r, grid); b = _vb().run(r, grid)
    for x, y in zip(a, b):
        if isinstance(x, pd.DataFrame):
            pd.testing.assert_frame_equal(x, y, check_exact=True)
        else:
            pd.testing.assert_series_equal(x, y, check_exact=True)


# ---------------------------------------------------------------- tracker level: shared track management
def _clean_with_dropout(stop_t=60.0):
    """The two crossing aircraft, no clutter, Pd 1; aircraft 0 stops reporting at stop_t."""
    r = G._reports(seed=0, clutter=0, pd_=1.0)
    r = r[~((r.truth == 0) & (r.time >= stop_t))].reset_index(drop=True)
    r["range_m"] = 50000.0; r["el_deg"] = 1.0
    return r


def test_vb_track_start_confirm_and_delete_match_gnn():
    r = _clean_with_dropout()
    grid = np.arange(0.0, 120.0, 2.0)
    _, rep_g, trk_g = GNNTracker({"tracking": {"method": "gnn"}}).run(r, grid)
    _, rep_v, trk_v = _vb().run(r, grid)
    cols = ["t_first", "t_confirm", "t_last", "filter_tier"]
    assert len(trk_g) == len(trk_v) == 2
    pd.testing.assert_frame_equal(trk_g[cols].sort_values("t_last").reset_index(drop=True),
                                  trk_v[cols].sort_values("t_last").reset_index(drop=True), check_exact=True)
    coast = 12.0                                                      # Tier 2 coast_s (tracking.py defaults)
    t_last = trk_v.t_last.min()
    assert 58.0 <= t_last <= 60.0 + coast                             # the dropped aircraft's track ended on its coast
    # every report went to the same track (by order of appearance) in both trackers
    assert (pd.factorize(rep_g)[0] == pd.factorize(rep_v)[0]).all()


def test_vb_isolated_tracks_are_not_re_solved():
    """Far-apart aircraft never share reports: the window solves nothing and VB equals its filter step."""
    r = G._reports(seed=3, clutter=0, pd_=1.0)
    r["range_m"] = 50000.0; r["el_deg"] = 1.0
    r = r[(r.time < 90.0)].reset_index(drop=True)                    # stop before the two paths meet
    grid = np.arange(0.0, 90.0, 2.0)
    tr = _vb(); e_w, _, _ = tr.run(r, grid)
    e_f, _, _ = _vb({"soft": {"iterations": 0}}).run(r, grid)
    assert tr.stats["window_solves"] == 0
    pd.testing.assert_frame_equal(e_w, e_f, check_exact=True)
