"""Tracker bench: GNN, JPDA and VB on controlled close-target scenarios, outside cidgen.

    python tracker_bench.py                      # all cases, 20 Monte Carlo runs each
    python tracker_bench.py --runs 50 --out bench_results.csv

Why: before trusting a tracker inside the CID chain, see how it behaves where the answer is known.
Every case is one radar (sigma 100 m per axis, a look every 2 s), clutter at a set density, a set Pd:

  cross-<deg>       two aircraft at the same altitude crossing at <deg> degrees
  formation-<k>     N aircraft line abreast, k sigma apart (k = 1 .. 8), flying straight for 90 s, then
                    fanning out at 10 deg per aircraft (do the tracks come apart correctly?)
Measured against truth for each tracker (and the ORACLE: the same Kalman filter fed the true associations):
  tracks_per_object   distinct confirmed tracks that held each object (1 = never broken)
  swaps               times a track's matched object changed to another object
  coverage            share of grid times each object had a matched track
  rmse_m              position error of matched tracks
  nees                mean normalised position error, d2 = e' P^-1 e over 3 axes; 3 = consistent,
                      above 3 = the tracker claims more accuracy than it has (overconfident)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cidgen.trackers import make_tracker  # noqa: E402

SIG = 100.0
DT = 2.0


def truth_paths(case, n_obj, sep_sigma, angle_deg, horizon, speed=200.0):
    """Positions (n_obj, T, 3) at look times."""
    t = np.arange(0.0, horizon, DT)
    P = np.zeros((n_obj, len(t), 3))
    if case == "cross":
        a = np.deg2rad(angle_deg) / 2
        for k, s in enumerate((-1, 1)):
            d = np.array([np.cos(s * a), np.sin(s * a), 0.0])
            P[k] = (t[:, None] - horizon / 2) * speed * d + np.array([0, 0, 5000.0])
    else:
        sep = sep_sigma * SIG
        for k in range(n_obj):
            y0 = (k - (n_obj - 1) / 2) * sep
            ang = np.deg2rad(10.0 * (k - (n_obj - 1) / 2))
            for n, tt in enumerate(t):
                if tt <= 90.0:
                    P[k, n] = [-20000 + speed * tt, y0, 5000.0]
                else:
                    p90 = np.array([-20000 + speed * 90.0, y0, 5000.0])
                    P[k, n] = p90 + speed * (tt - 90.0) * np.array([np.cos(ang), np.sin(ang), 0.0])
    return t, P


def reports(t, P, rng, pd_, clutter_per_look, box=8000.0):
    rows = []
    for n, tt in enumerate(t):
        cen = P[:, n].mean(0)
        for k in range(P.shape[0]):
            if rng.random() < pd_:
                rows.append((tt, *(P[k, n] + rng.normal(0, SIG, 3)), k))
        for _ in range(rng.poisson(clutter_per_look)):
            rows.append((tt, *(cen + rng.uniform(-box, box, 3) * np.array([1, 1, 0.1])), -1))
    r = pd.DataFrame(rows, columns=["time", "e", "n", "u", "truth"])
    r["sensor_id"] = "R_S-1"; r["sensor_type"] = "R_S"; r["rcs_dbsm"] = 10.0; r["vr_mps"] = 0.0
    r["range_m"] = 50000.0; r["el_deg"] = 1.0
    r["R"] = [np.eye(3) * SIG ** 2] * len(r)
    return r


def cfg_for(method, clutter_per_look, box=8000.0):
    # clutter density the soft trackers assume = the bench's true density
    vol = (2 * box) * (2 * box) * (0.2 * box)
    r_ref = 75000.0
    lam_spec = clutter_per_look / vol * (0.95 * r_ref * 2 * np.pi * np.deg2rad(3.0) * 50000.0 ** 2)
    # one radar, no registration bias: no duplicate tracks to drop, so duplicate dropping is off
    # (it also deletes real tracks in formations a few sigma apart; see TRACKERS.md)
    return {"tracking": {"method": method, "gnn": {"merge_checks": 10 ** 9}},
            "sensors": {"clutter_rate": lam_spec, "r_p_sector_halfwidth_deg": 60.0,
                        "radar_types": {"R_S": {"ref_range_km": 75.0, "sector": "full"}}}}


def oracle(r, grid, q=30.0):
    """Kalman filter per object with the true associations (lower bound on error)."""
    rows = []
    for k, g in r[r.truth >= 0].groupby("truth"):
        x = np.r_[g[["e", "n", "u"]].values[0], 0, 0, 0]; P = np.diag([SIG ** 2] * 3 + [100.0 ** 2] * 3)
        tp = g.time.values[0]; gi = np.searchsorted(grid, tp)
        for tt, z in zip(g.time.values[1:], g[["e", "n", "u"]].values[1:]):
            while gi < len(grid) and grid[gi] < tt:
                rows.append(_pred(k, grid[gi], x, P, grid[gi] - tp, q)); gi += 1
            dt = tt - tp
            F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
            Q = q * np.block([[np.eye(3) * dt ** 3 / 3, np.eye(3) * dt ** 2 / 2], [np.eye(3) * dt ** 2 / 2, np.eye(3) * dt]])
            x = F @ x; P = F @ P @ F.T + Q
            S = P[:3, :3] + np.eye(3) * SIG ** 2; K = np.linalg.solve(S, P[:3]).T
            x = x + K @ (z - x[:3]); P = P - K @ S @ K.T; tp = tt
        while gi < len(grid):
            rows.append(_pred(k, grid[gi], x, P, grid[gi] - tp, q)); gi += 1
    return pd.DataFrame(rows, columns=["eid", "t", "e", "n", "u", "P"])


def _pred(k, tg, x, P, dt, q):
    F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
    Q = q * np.block([[np.eye(3) * dt ** 3 / 3, np.eye(3) * dt ** 2 / 2], [np.eye(3) * dt ** 2 / 2, np.eye(3) * dt]])
    xs = F @ x; Ps = F @ P @ F.T + Q
    return (k, tg, *xs[:3], Ps[:3, :3])


def score(est, t_truth, P_truth, with_cov):
    """One-to-one matching per grid time (gate 1 km); per-object metrics."""
    n_obj = P_truth.shape[0]
    held = {k: [] for k in range(n_obj)}; err, nees = [], []; prev = {}; swaps = 0
    for tg, g in est.groupby("t"):
        n = np.searchsorted(t_truth, tg)
        if n >= len(t_truth) or abs(t_truth[n] - tg) > 1e-6:
            continue
        X = g[["e", "n", "u"]].values; D = np.linalg.norm(X[:, None] - P_truth[None, :, n], axis=2)
        rr, cc = linear_sum_assignment(np.where(D < 1000, D, 1e9))
        for a, b in zip(rr, cc):
            if D[a, b] >= 1000:
                continue
            tid = g.eid.values[a]; held[b].append(tid)
            if tid in prev and prev[tid] != b:
                swaps += 1
            prev[tid] = b
            e = X[a] - P_truth[b, n]; err.append(e @ e)
            if with_cov:
                nees.append(float(e @ np.linalg.solve(g.P.values[a], e)))
    T = len(t_truth)
    return dict(tracks_per_object=np.mean([len(set(v)) for v in held.values()]), swaps=swaps,
                coverage=np.mean([len(v) / T for v in held.values()]),
                rmse_m=float(np.sqrt(np.mean(err))) if err else np.nan,
                nees=float(np.mean(nees)) if nees else np.nan)


def run_case(case, n_obj, sep, angle, runs, pd_, clutter, horizon=180.0, methods=("gnn", "jpda", "vb")):
    out = []
    t, P = truth_paths(case, n_obj, sep, angle, horizon)
    grid = t.copy()
    for k in range(runs):
        rng = np.random.default_rng(1000 + k)
        r = reports(t, P, rng, pd_, clutter)
        for method in ("oracle", *methods):
            if method == "oracle":
                est = oracle(r, grid)
                m = score(est, t, P, True)
            else:
                tr = make_tracker(cfg_for(method, clutter), method)
                tr.cov_log = {}
                est, _, _ = tr.run(r, grid)
                cov = _covs(tr, est)
                m = score(est.assign(P=cov), t, P, True)
            out.append(dict(case=f"{case}-{angle if case == 'cross' else sep}", n_obj=n_obj, run=k, method=method, **m))
    return out


def _covs(tr, est):
    """Full position covariance per estimate row (logged by the tracker for the bench)."""
    return [tr.cov_log[(int(e), float(t))] for e, t in zip(est.eid, est.t)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--pd", type=float, default=0.9)
    ap.add_argument("--clutter", type=float, default=2.0, help="false alarms per look in a 16 x 16 x 1.6 km box")
    ap.add_argument("--methods", nargs="+", default=["gnn", "jpda", "vb"],
                    help="trackers to compare: gnn jpda vb vbm mfa (mfa needs stonesoup + ortools)")
    ap.add_argument("--swarm", action="store_true",
                    help="run the unresolved-swarm cases (merged reports) instead of crossing/formation")
    ap.add_argument("--protect", action="store_true",
                    help="swarm cases: protect resolved tracks from duplicate dropping (30 s) for every tracker")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global PROTECT_S
    PROTECT_S = 30.0 if a.protect else 0.0
    rows = []
    pd.set_option("display.width", 160)
    if a.swarm:
        for n, sp in ((4, 60.0), (4, 25.0), (8, 25.0)):
            rows += run_swarm(n, sp, a.runs, a.pd, a.clutter, methods=tuple(a.methods))
        df = pd.DataFrame(rows)
        tab = df.groupby(["case", "method"], sort=False)[["own_track", "tracks_per_object", "tracks_alive",
                                                          "rmse_m", "merged_share"]].mean()
    else:
        for ang in (10, 30, 90):
            rows += run_case("cross", 2, 0, ang, a.runs, a.pd, a.clutter, methods=tuple(a.methods))
        for sep in (1, 2, 3, 5, 8):
            rows += run_case("formation", 4, sep, 0, a.runs, a.pd, a.clutter, methods=tuple(a.methods))
        df = pd.DataFrame(rows)
        tab = df.groupby(["case", "method"], sort=False)[["tracks_per_object", "swaps", "coverage", "rmse_m", "nees"]].mean()
    print(tab.round(3).to_string())
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
        tab.round(4).to_csv(Path(a.out).with_name(Path(a.out).stem + "_summary.csv"))
        print(f"wrote {a.out} and its _summary.csv")


# ---------------------------------------------------------------------- unresolved groups (merged reports)
RC = dict(sigma_r_m=3.0, sigma_az_deg=0.5, sigma_el_deg=0.5, res_r_m=10.0, res_az_deg=2.0, update_s=0.5,
          ref_range_km=5.0)


def swarm_paths(n_obj, spread_m, horizon, seed):
    """n drones around a common path 4-6 km from a counter-UAS radar at the origin; each wanders on its
    own slow loop of about spread_m, so the radar merges and resolves them intermittently."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, horizon, RC["update_s"])
    base = np.stack([-3000 + 30.0 * t, 4000 + 0 * t, 150 + 0 * t], 1)
    P = np.zeros((n_obj, len(t), 3))
    for k in range(n_obj):
        ph = rng.uniform(0, 2 * np.pi, 2); w = rng.uniform(0.02, 0.05, 2)
        off = np.stack([spread_m * np.cos(w[0] * t + ph[0]), spread_m * np.sin(w[1] * t + ph[1]),
                        0.2 * spread_m * np.sin(w[0] * t + ph[1])], 1)
        P[k] = base + off + rng.normal(0, spread_m * 0.3, 3)
    return t, P


def merged_reports(t, P, rng, pd_, clutter_per_look):
    """Detections merged greedily in range order inside one resolution cell (the generator's rule);
    a merged report sits at an echo-weighted centroid of its members."""
    from cidgen.geometry import enu_to_polar, polar_cov_to_enu, polar_to_enu
    rows = []
    for n, tt in enumerate(t):
        det = [k for k in range(P.shape[0]) if rng.random() < pd_]
        if det:
            rg, az, el = enu_to_polar(P[det, n])
            w = rng.exponential(1.0, len(det))
            order = np.argsort(rg); clusters, cur = [], []
            for k in order:
                if cur and abs(rg[k] - rg[cur[0]]) < RC["res_r_m"] and \
                        abs((az[k] - az[cur[0]] + 180) % 360 - 180) < RC["res_az_deg"] and \
                        abs(el[k] - el[cur[0]]) < 1.5 * RC["res_az_deg"]:
                    cur.append(k); continue
                if cur:
                    clusters.append(cur)
                cur = [k]
            clusters.append(cur)
            for c in clusters:
                cen = (P[[det[k] for k in c], n] * (w[c] / w[c].sum())[:, None]).sum(0)
                r0, a0, e0 = enu_to_polar(cen)
                rows.append((tt, r0 + rng.normal(0, RC["sigma_r_m"]), a0 + rng.normal(0, RC["sigma_az_deg"]),
                             e0 + rng.normal(0, RC["sigma_el_deg"]), det[c[int(np.argmax(w[c]))]], len(c)))
        for _ in range(rng.poisson(clutter_per_look)):
            rows.append((tt, rng.uniform(250, 5000), rng.uniform(0, 360), rng.uniform(0, 3), -1, 1))
    r = pd.DataFrame(rows, columns=["time", "range_m", "az_deg", "el_deg", "truth", "n_merged"])
    xyz = polar_to_enu(r.range_m.values, r.az_deg.values, r.el_deg.values)
    r["e"], r["n"], r["u"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r["R"] = list(polar_cov_to_enu(r.range_m.values, r.az_deg.values, r.el_deg.values,
                                   np.full(len(r), RC["sigma_r_m"]), np.full(len(r), RC["sigma_az_deg"]),
                                   np.full(len(r), RC["sigma_el_deg"])))
    r["sensor_id"] = "R_C-1"; r["sensor_type"] = "R_C"; r["rcs_dbsm"] = -20.0; r["vr_mps"] = 0.0
    r["site_e"] = r["site_n"] = r["site_u"] = 0.0
    return r


PROTECT_S = 0.0          # --protect: protect resolved tracks from duplicate dropping for every tracker


def cfg_rc(method, clutter_per_look, protect=None):
    p = PROTECT_S if protect is None else protect
    return {"tracking": {"method": method, "gnn": {"merge_checks": 3, "protect_resolved_s": p}},
            "sensors": {"clutter_rate": clutter_per_look, "r_p_sector_halfwidth_deg": 60.0,
                        "radar_types": {"R_C": {"ref_range_km": RC["ref_range_km"], "sector": "full",
                                                "res_r_m": RC["res_r_m"], "res_az_deg": RC["res_az_deg"]}}}}


def run_swarm(n_obj, spread_m, runs, pd_=0.9, clutter=2.0, horizon=240.0, methods=("gnn", "jpda", "vb", "vbm")):
    out = []
    for k in range(runs):
        t, P = swarm_paths(n_obj, spread_m, horizon, 500 + k)
        r = merged_reports(t, P, np.random.default_rng(2000 + k), pd_, clutter)
        grid = np.arange(0.0, horizon, 2.0)
        n_idx = np.round(grid / RC["update_s"]).astype(int)
        for method in methods:
            tr = make_tracker(cfg_rc(method, clutter), method)
            tr.cov_log = {}
            est, _, _ = tr.run(r, grid)
            m = score_swarm(est, grid, P[:, n_idx], gate=max(3 * spread_m, 150.0))
            m["merged_share"] = float((r.n_merged > 1).mean())
            m["stats"] = str(getattr(tr, "stats", ""))
            out.append(dict(case=f"swarm{n_obj}-{int(spread_m)}m", run=k, method=method, **m))
    return out


def score_swarm(est, grid, Pg, gate):
    n_obj = Pg.shape[0]
    held = {k: [] for k in range(n_obj)}; err = []
    for n, tg in enumerate(grid):
        g = est[np.isclose(est.t, tg)]
        if g.empty:
            continue
        X = g[["e", "n", "u"]].values; D = np.linalg.norm(X[:, None] - Pg[None, :, n], axis=2)
        rr, cc = linear_sum_assignment(np.where(D < gate, D, 1e9))
        for a, b in zip(rr, cc):
            if D[a, b] < gate:
                held[b].append(g.eid.values[a]); err.append(D[a, b] ** 2)
    return dict(own_track=np.mean([len(v) / len(grid) for v in held.values()]),
                tracks_per_object=np.mean([len(set(v)) for v in held.values()]),
                tracks_alive=float(est.groupby("t").size().mean()),
                rmse_m=float(np.sqrt(np.mean(err))) if err else np.nan)


if __name__ == "__main__":
    main()
